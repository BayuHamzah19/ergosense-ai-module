"""
baseline_loso.py - Jendela waktu + baseline rule-based + evaluasi Leave-One-Subject-Out (LOSO).

Jalankan dari folder ERGO-SENSE (setelah run_all.py):
    python src/baseline_loso.py                       # subjek 32 dikeluarkan (default)
    python src/baseline_loso.py --exclude 32 31       # keluarkan 31 juga
    python src/baseline_loso.py --train-exclude 31    # 31 tidak dipakai menyetel ambang, tetap diuji

Langkah:
  1. Jendela 5 detik, geser 1 detik, berdasarkan TIMESTAMP (bukan jumlah frame).
     Jendela dengan > 20% frame hilang dibuang.
  2. Fitur dihitung dari konteks yang berakhir di jendela. Konteks 30 detik mengikuti paper (Bab 3.4);
     versi _5 (hanya jendela 5 detik) setara dengan skala waktu input ST-GCN, dan _15 sebagai perantara:
        p50_30      fraksi waktu EAR < 0.5 * baseline
        p20_30      fraksi waktu EAR < 0.2 * baseline   (mata tertutup >= 80%)
        maxclose_30 durasi terpanjang mata tertutup terus-menerus (detik, EAR < 0.5 * baseline)
     Jendela pertama yang konteksnya belum 30 detik dibuang.
  3. Tiga aturan satu-fitur: "drowsy jika fitur >= ambang". Ambang disetel HANYA pada subjek latih
     (maksimalkan balanced accuracy), lalu diuji pada satu subjek yang ditahan.
  4. Dilaporkan per subjek uji: AUC (tanpa ambang), balanced accuracy, recall & precision kelas drowsy.

Label: 0 = alert (file 0), 1 = drowsy (file 10, 10_1, 10_2). Label berlaku untuk seluruh rekaman,
jadi label jendela hanya perkiraan. JANGAN melabel ulang jendela berdasarkan fitur.
Baseline EAR (--calib): alert = persentil ke-90 seluruh video alert (optimistis, in-sample);
alert30 / self30 = dari 30 detik pertama saja, jendela di segmen kalibrasi dibuang.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from check_signal import load

ROOT = Path(__file__).resolve().parent.parent
WIN, STEP, CTX, MAX_NAN = 5.0, 1.0, 30.0, 0.20
FEATURES = ["p50_5", "maxclose_5", "p50_15", "p50_30", "p20_30", "maxclose_30"]


def longest_run(c, tt):
    """Durasi (detik) run True terpanjang pada boolean c dengan timestamp tt."""
    if len(c) < 2 or not c.any():
        return 0.0
    d = np.diff(np.concatenate([[0], c.astype(int), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return float((tt[e - 1] - tt[s] + np.median(np.diff(tt))).max())


def windows_for(e, t, base, subj, label_name, y, cal_skip=0.0):
    r = e / base
    nan = np.isnan(r)
    c50 = np.where(nan, np.nan, (r < 0.5).astype(float))
    c20 = np.where(nan, np.nan, (r < 0.2).astype(float))
    rows = []
    t_end = t[-1]
    w_end = t[0] + cal_skip + CTX                   # konteks penuh 30 detik, setelah segmen kalibrasi
    while w_end <= t_end + 1e-6:
        w_start = w_end - WIN
        i0, i1 = np.searchsorted(t, w_start), np.searchsorted(t, w_end)
        j0 = np.searchsorted(t, w_end - CTX)
        if i1 - i0 >= 2 and nan[i0:i1].mean() <= MAX_NAN:
            ctx = slice(j0, i1)
            valid = ~nan[ctx]
            k0 = np.searchsorted(t, w_end - 15.0)
            if valid.sum() > 0:
                rows.append({
                    "subject": subj, "file": label_name, "y": y, "t_end": round(float(w_end), 1),
                    "p50_5": float(np.nanmean(c50[i0:i1])),
                    "maxclose_5": longest_run((r[i0:i1] < 0.5) & ~nan[i0:i1], t[i0:i1]),
                    "p50_15": float(np.nanmean(c50[k0:i1])),
                    "p50_30": float(np.nanmean(c50[ctx])),
                    "p20_30": float(np.nanmean(c20[ctx])),
                    "maxclose_30": longest_run((r[ctx] < 0.5) & valid, t[ctx]),
                })
        w_end += STEP
    return rows


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bal_acc(y, pred):
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else np.nan
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return float((tpr + tnr) / 2)


def best_threshold(y, s):
    cands = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    scores = [bal_acc(y, (s >= th).astype(int)) for th in cands]
    k = int(np.argmax(scores))
    return float(cands[k]), float(scores[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=["32"], help="subjek dikeluarkan sepenuhnya")
    ap.add_argument("--calib", choices=["alert", "alert30", "self30"], default="alert",
                    help="baseline EAR dari: alert=seluruh video alert (optimistis), alert30=30 dtk pertama "
                         "video alert, self30=30 dtk pertama tiap rekaman. Selain 'alert', 60 dtk pertama dibuang.")
    ap.add_argument("--train-exclude", nargs="*", default=[], help="subjek tidak dipakai menyetel ambang")
    args = ap.parse_args()

    files = sorted(p for p in (ROOT / "landmarks").glob("*.npz") if not p.name.startswith("test_"))
    data = {}
    for f in files:
        subj, lab = f.stem.split("_", 1)
        if subj not in args.exclude:
            data.setdefault(subj, {})[lab] = load(f)

    rows = []
    for subj, vids in sorted(data.items()):
        drowsy = [k for k in vids if k.startswith("10")]
        if "0" not in vids or not drowsy:
            print(f"[lewati] subjek {subj}: butuh video 0 dan 10")
            continue
        e0, t0v = vids["0"][0], vids["0"][2]
        for k in ["0"] + drowsy:
            e, _, t = vids[k]
            first30 = lambda ee, tt: ee[tt < tt[0] + 30]
            if args.calib == "alert":
                base, skip = np.nanpercentile(e0, 90), 0.0
            elif args.calib == "alert30":
                base, skip = np.nanpercentile(first30(e0, t0v), 90), (30.0 if k == "0" else 0.0)
            else:
                base, skip = np.nanpercentile(first30(e, t), 90), 30.0
            rows += windows_for(e, t, base, subj, k, 0 if k == "0" else 1, cal_skip=skip)
    W = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    W.to_csv(out / "windows.csv", index=False)
    subjects = sorted(W.subject.unique())
    print(f"Jendela: {len(W)} dari {len(subjects)} subjek {subjects}")
    print(W.groupby(["subject", "y"]).size().unstack().rename(columns={0: "alert", 1: "drowsy"}).to_string())

    res = []
    for test_s in subjects:
        train = W[(W.subject != test_s) & (~W.subject.isin(args.train_exclude))]
        test = W[W.subject == test_s]
        for feat in FEATURES:
            th, tr_ba = best_threshold(train.y.values, train[feat].values)
            pred = (test[feat].values >= th).astype(int)
            y = test.y.values
            tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
            res.append({
                "test_subject": test_s, "rule": feat, "threshold": round(th, 3),
                "train_bal_acc": round(tr_ba, 3),
                "AUC": round(auc(y, test[feat].values), 3),
                "bal_acc": round(bal_acc(y, pred), 3),
                "recall_drowsy": round(tp / max((y == 1).sum(), 1), 3),
                "precision_drowsy": round(tp / (tp + fp), 3) if (tp + fp) else np.nan,
            })
    R = pd.DataFrame(res)
    R.to_csv(out / "baseline_loso.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nHasil LOSO per subjek uji:")
    print(R.to_string(index=False))
    print("\nAUC per subjek uji (baris = fitur; _5 = jendela 5 dtk saja, _15/_30 = konteks 15/30 dtk):")
    piv = R.pivot(index="rule", columns="test_subject", values="AUC").reindex(FEATURES)
    piv["rata2"] = piv.mean(axis=1)
    piv["rata2_tanpa31"] = piv.drop(columns=["rata2"]).drop(columns=["31"], errors="ignore").mean(axis=1)
    print(piv.round(3).to_string())
    print("\nRata-rata atas subjek uji (tiap subjek bobot sama).")
    print("Catatan: precision hanya dirata-rata pada subjek yang punya prediksi drowsy (NaN dilewati).")
    print(R.groupby("rule")[["AUC", "bal_acc", "recall_drowsy", "precision_drowsy"]].mean().round(3).to_string())
    print("\nTanpa subjek 31 pada rata-rata:")
    print(R[R.test_subject != "31"].groupby("rule")[["AUC", "bal_acc"]].mean().round(3).to_string())
    print(f"\nDisimpan: {out / 'windows.csv'}, {out / 'baseline_loso.csv'}")


if __name__ == "__main__":
    main()