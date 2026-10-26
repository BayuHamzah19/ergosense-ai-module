"""
make_dataset.py - Landmark (.npz) -> tensor jendela untuk ST-GCN.

Jalankan dari folder ERGO-SENSE (setelah run_all.py):
    python src/make_dataset.py                    # subjek 32 dikeluarkan (default)
    python src/make_dataset.py --calibrate        # kurangi pose netral subjek (dari video alert)
    python src/make_dataset.py --window 15        # jendela 15 detik (ablation panjang jendela)

Pemrosesan:
  1. Ambil ulang sampel ke grid waktu SERAGAM 15 FPS (interpolasi linear antar frame). Penting:
     frame hasil run_all.py bisa berjarak tidak seragam (mis. video 25 FPS: 40 ms dan 80 ms bergantian).
     Celah > 0.5 detik tetap NaN.
  2. Normalisasi PER FRAME: pusat = titik tengah kedua mata, skala = jarak antar pusat mata.
  3. (opsional) kalibrasi: kurangi pose netral. Tiga mode:
       --calibrate alert    median pose SELURUH video alert. OPTIMISTIS: jendela alert dikalibrasi dengan
                            statistiknya sendiri (in-sample), jendela drowsy dengan rekaman lain.
       --calibrate alert30  30 detik pertama video alert; jendela alert yang tumpang tindih segmen itu dibuang.
       --calibrate self30   tiap rekaman dikalibrasi dari 30 detik pertamanya sendiri (tanpa label),
                            jendela di 30 detik pertama dibuang. Paling mirip aplikasi (kalibrasi awal sesi).
  4. Kanal tiap node: x, y, dx, dy (selisih dengan frame sebelumnya).
  5. Jendela = window detik x 15 FPS frame, geser 1 detik. Jendela dengan > 20% frame hilang dibuang.

Keluaran: data/stgcn_windows_w<window>[_cal].npz
    X: (N, 4, T, 20) float32   y: (N,) 0=alert 1=drowsy   subject: (N,)   video: (N,)   t_end: (N,)
Hanya angka koordinat ternormalisasi yang disimpan, tidak ada gambar wajah.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from check_signal import fill_short_gaps

ROOT = Path(__file__).resolve().parent.parent
FPS = 15
MAX_NAN = 0.20


def resample_uniform(pts, t, max_gap=0.5):
    """pts (T,20,2) pada timestamp t (tidak seragam) -> grid seragam 15 FPS (interpolasi linear).
    Titik grid menjadi NaN bila sampel valid terdekat lebih jauh dari max_gap detik."""
    tg = np.arange(t[0], t[-1] + 1e-9, 1.0 / FPS)
    out = np.full((len(tg), pts.shape[1], 2), np.nan)
    for n in range(pts.shape[1]):
        for c in range(2):
            y = pts[:, n, c].astype(np.float64)
            ok = ~np.isnan(y)
            if ok.sum() < 2:
                continue
            tv = t[ok].astype(np.float64)
            out[:, n, c] = np.interp(tg, tv, y[ok])
            k = np.searchsorted(tv, tg)
            lo, hi = tv[np.clip(k - 1, 0, len(tv) - 1)], tv[np.clip(k, 0, len(tv) - 1)]
            out[np.minimum(np.abs(tg - lo), np.abs(hi - tg)) > max_gap, n, c] = np.nan
    return out, tg


def normalize_video(pts):
    """pts (T,20,2) piksel -> (T,20,2) ternormalisasi per frame; NaN dipertahankan."""
    out = pts.astype(np.float64)
    right, left = out[:, 0:6].mean(axis=1), out[:, 6:12].mean(axis=1)
    center = (right + left) / 2
    iod = np.linalg.norm(right - left, axis=1)
    iod = np.where(iod > 1e-6, iod, np.nan)
    return (out - center[:, None, :]) / iod[:, None, None]


def channels(norm):
    """(T,20,2) -> (4,T,20) dengan x, y, dx, dy."""
    d = np.zeros_like(norm)
    d[1:] = norm[1:] - norm[:-1]
    d[0] = np.nan if np.isnan(norm[0]).any() else 0.0
    d[np.isnan(d)] = np.nan
    x = np.concatenate([norm, d], axis=2)              # (T,20,4)
    return x.transpose(2, 0, 1)                         # (4,T,20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=["32"])
    ap.add_argument("--window", type=float, default=5.0, help="panjang jendela (detik)")
    ap.add_argument("--stride", type=float, default=1.0, help="geser (detik)")
    ap.add_argument("--tag", default="", help="akhiran nama file, mis. matched")
    ap.add_argument("--calibrate", nargs="?", const="alert", default="none",
                    choices=["none", "alert", "alert30", "self30"],
                    help="alert=pose rata-rata SELURUH video alert (optimistis, in-sample); "
                         "alert30=30 detik pertama video alert; self30=30 detik pertama tiap rekaman")
    args = ap.parse_args()

    L, S = int(round(args.window * FPS)), int(round(args.stride * FPS))
    files = sorted(p for p in (ROOT / "landmarks").glob("*.npz") if not p.name.startswith("test_"))
    vids = {}
    for f in files:
        subj, lab = f.stem.split("_", 1)
        if subj in args.exclude:
            continue
        d = np.load(f)
        t = d["t"]
        eff = (len(t) - 1) / (t[-1] - t[0])                # FPS efektif rata-rata
        if eff < 14.5:
            print(f"[lewati] {f.name}: FPS efektif {eff:.1f} < 15 (akan mengarang frame saat di-resample)")
            continue
        pts_u, tg = resample_uniform(d["pts"], t)           # grid seragam 15 FPS
        vids.setdefault(subj, {})[lab] = (normalize_video(pts_u), tg)

    Xs, ys, ss, ff, te = [], [], [], [], []
    for subj, v in sorted(vids.items()):
        drowsy = [k for k in v if k.startswith("10")]
        if "0" not in v or not drowsy:
            print(f"[lewati] subjek {subj}: butuh video 0 dan 10")
            continue
        CAL_F = 30 * FPS                                  # 30 detik pertama = segmen kalibrasi
        for k in ["0"] + drowsy:
            norm, t = v[k]
            start_min = 0                                  # jendela yang dimulai sebelum ini dibuang
            if args.calibrate == "alert":
                pose = np.nanmedian(v["0"][0], axis=0)
            elif args.calibrate == "alert30":
                pose = np.nanmedian(v["0"][0][:CAL_F], axis=0)
                start_min = CAL_F if k == "0" else 0
            elif args.calibrate == "self30":
                pose = np.nanmedian(norm[:CAL_F], axis=0)
                start_min = CAL_F
            else:
                pose = None
            if pose is not None:
                norm = norm - pose[None]
            ch = channels(norm)                             # (4,T,20)
            nanf = np.isnan(ch).any(axis=(0, 2))            # frame yang mengandung NaN
            for a in range(start_min, ch.shape[1] - L + 1, S):
                if nanf[a:a + L].mean() > MAX_NAN:
                    continue
                w = ch[:, a:a + L].copy()
                m = np.nanmean(w, axis=1, keepdims=True)    # sisa NaN -> rata-rata jendela
                w = np.where(np.isnan(w), m, w)
                Xs.append(w.astype(np.float32))
                ys.append(0 if k == "0" else 1)
                ss.append(subj); ff.append(k); te.append(float(t[a + L - 1]))

    X = np.stack(Xs)
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    suffix = {"none": "", "alert": "_cal", "alert30": "_cal_alert30", "self30": "_cal_self30"}[args.calibrate]
    if abs(args.stride - 1.0) > 1e-6:
        suffix += f"_s{args.stride:g}"
    if args.tag:
        suffix += f"_{args.tag}"
    name = f"stgcn_windows_w{int(args.window)}{suffix}.npz"
    np.savez_compressed(out / name, X=X, y=np.array(ys), subject=np.array(ss),
                        video=np.array(ff), t_end=np.array(te))
    print(f"Jendela: {len(X)}  bentuk X: {X.shape}  (N, kanal, waktu, node)")
    for s in sorted(set(ss)):
        yy = np.array(ys)[np.array(ss) == s]
        print(f"  subjek {s}: alert {(yy == 0).sum()}, drowsy {(yy == 1).sum()}")
    print(f"Disimpan: {out / name}  ({(out / name).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()