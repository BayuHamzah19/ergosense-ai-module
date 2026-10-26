"""
check_signal.py - Cek cepat: apakah sinyal video drowsy memang beda dari alert?

Jalankan dari folder ERGO-SENSE (setelah run_all.py selesai):
    python src/check_signal.py

Untuk tiap file landmarks/<subjek>_<label>.npz dihitung:
    eff_fps      FPS efektif setelah resampling (dari timestamp)
    ear_med      median EAR
    perclos20    fraksi waktu EAR < 0.20 * baseline   (mata tertutup >= 80%)
    perclos50    fraksi waktu EAR < 0.50 * baseline
    blink_per_min, blink_dur_ms   dari penurunan EAR di bawah 0.70 * baseline
    mar_gt50     fraksi waktu MAR > 0.50 (indikasi mulut terbuka lebar)

baseline = persentil ke-90 EAR pada video ALERT (label 0) subjek yang sama.
CATATAN: ini memakai label alert untuk kalibrasi, sebagaimana kalibrasi 30 detik
di aplikasi (pengguna dalam keadaan segar). Tulis eksplisit di paper.
Ini cek awal, BUKAN hasil evaluasi model.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from extract import ear_mar_from_pts

ROOT = Path(__file__).resolve().parent.parent


def fill_short_gaps(x, max_gap):
    """Interpolasi NaN yang pendek (<= max_gap frame); celah panjang tetap NaN."""
    x = x.copy()
    isn = np.isnan(x)
    if not isn.any() or isn.all():
        return x
    idx = np.arange(len(x))
    interp = np.interp(idx, idx[~isn], x[~isn])
    # tandai celah panjang
    run_start = None
    for i in range(len(x) + 1):
        cur = isn[i] if i < len(x) else False
        if cur and run_start is None:
            run_start = i
        elif not cur and run_start is not None:
            if i - run_start <= max_gap:
                x[run_start:i] = interp[run_start:i]
            run_start = None
    return x


def blinks(e, t, thr):
    """Kembalikan daftar durasi (detik) tiap episode EAR < thr."""
    below = (e < thr) & ~np.isnan(e)
    durs, start = [], None
    for i in range(len(e)):
        if below[i] and start is None:
            start = i
        elif not below[i] and start is not None:
            durs.append(t[i] - t[start])
            start = None
    return durs


def load(path):
    d = np.load(path)
    pts, t = d["pts"], d["t"]
    e, m = ear_mar_from_pts(pts)
    fps = float(d["target_fps"])
    gap = int(round(0.5 * fps))                 # isi celah <= 0.5 detik
    return fill_short_gaps(e, gap), fill_short_gaps(m, gap), t


def main():
    files = sorted(p for p in (ROOT / "landmarks").glob("*.npz") if not p.name.startswith("test_"))
    if not files:
        sys.exit("Tidak ada file di landmarks/. Jalankan run_all.py dulu.")

    data = defaultdict(dict)                    # subjek -> {nama_label: (e, m, t)}
    for f in files:
        subj, label = f.stem.split("_", 1)
        data[subj][label] = load(f)

    rows = []
    for subj, vids in sorted(data.items()):
        if "0" not in vids:
            print(f"[lewati] subjek {subj} tidak punya video alert (0)")
            continue
        e0 = vids["0"][0]
        baseline = np.nanpercentile(e0, 90)
        for label, (e, m, t) in sorted(vids.items()):
            valid = ~np.isnan(e)
            dur_min = (t[-1] - t[0]) / 60 if len(t) > 1 else np.nan
            d = blinks(e, t, 0.70 * baseline)
            rows.append({
                "subject": subj,
                "label": "alert" if label == "0" else "drowsy",
                "file": label,
                "eff_fps": round((len(t) - 1) / (t[-1] - t[0]), 1),
                "minutes": round(dur_min, 1),
                "baseline": round(float(baseline), 3),
                "ear_med": round(float(np.nanmedian(e)), 3),
                "perclos20": round(float(np.nanmean(e[valid] < 0.20 * baseline)), 3),
                "perclos50": round(float(np.nanmean(e[valid] < 0.50 * baseline)), 3),
                "blink_per_min": round(len(d) / dur_min, 1),
                "blink_dur_ms": round(1000 * float(np.mean(d)), 0) if d else np.nan,
                "mar_gt50": round(float(np.nanmean(m[~np.isnan(m)] > 0.50)), 3),
            })

    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "video_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.drop(columns=["file"]).to_string(index=False))

    # ringkasan arah perbedaan drowsy - alert per subjek
    print("\nSelisih (drowsy - alert) per subjek:")
    cols = ["ear_med", "perclos20", "perclos50", "blink_per_min", "blink_dur_ms", "mar_gt50"]
    g = df.groupby(["subject", "label"])[cols].mean()
    diffs = []
    for subj in sorted(df.subject.unique()):
        if (subj, "alert") in g.index and (subj, "drowsy") in g.index:
            diffs.append((g.loc[(subj, "drowsy")] - g.loc[(subj, "alert")]).rename(subj))
    if diffs:
        dd = pd.DataFrame(diffs).round(3)
        print(dd.to_string())
        print("\nJumlah subjek di mana tiap fitur NAIK pada drowsy:")
        print((dd > 0).sum().to_string())
    print(f"\nDisimpan: {out / 'video_stats.csv'}")


if __name__ == "__main__":
    main()