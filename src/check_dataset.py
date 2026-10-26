"""
check_dataset.py - Diagnosis cepat tensor ST-GCN (hitungan detik, tanpa pelatihan).

Jalankan dari folder ERGO-SENSE:
    python src/check_dataset.py                                  # data tanpa kalibrasi
    python src/check_dataset.py --data data/stgcn_windows_w5_cal_self30.npz

Tujuan 1: apakah tensor X masih memuat sinyal kantuk?
    EAR dihitung ulang dari koordinat di X (EAR tidak berubah oleh pergeseran dan skala), lalu
    AUC per subjek. Pada data TANPA kalibrasi, p50_5 harus mendekati hasil baseline_loso
    (mode alert): 31=0.94 33=0.94 34=0.81 35=0.93 36=0.40. Jika jauh berbeda, ada bug di make_dataset.
Tujuan 2: adakah ciri tingkat-rekaman yang memisahkan kelas tanpa ada hubungan dengan kantuk?
    'jitter' (rata-rata |dx|+|dy|, getaran landmark) dan 'mar_mean' (bukaan mulut rata-rata).
    AUC yang jauh dari 0.5 (mendekati 0 atau 1) pada banyak subjek = ada jalan pintas yang bisa
    dihafal model (kamera, FPS, resolusi, posisi duduk), bukan kantuk.
AUC: 0.5 = tidak memisahkan; > 0.5 = nilai fitur lebih tinggi pada drowsy; < 0.5 = lebih rendah.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from baseline_loso import auc
from extract import ear, mar

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stgcn_windows_w5.npz")
    args = ap.parse_args()
    path = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    D = np.load(path)
    X, y, subj = D["X"], D["y"].astype(int), D["subject"].astype(str)
    N, C, T, V = X.shape
    print(f"Data: {path.name} | {N} jendela | subjek {sorted(set(subj.tolist()))}\n")

    pts = X[:, :2].transpose(0, 2, 3, 1).reshape(N * T, V, 2).astype(np.float64)
    e = ((ear(pts[:, 0:6]) + ear(pts[:, 6:12])) / 2).reshape(N, T)
    m = mar(pts[:, 12:20]).reshape(N, T)
    jitter = np.abs(X[:, 2:4]).mean(axis=(1, 2, 3))

    feats = {}
    rows = {k: {} for k in ["p50_5 (mata tertutup)", "ear_mean (dibalik)", "jitter", "mar_mean"]}
    for s in sorted(set(subj.tolist())):
        mk = subj == s
        base = np.nanpercentile(e[mk & (y == 0)], 90)              # baseline dari video alert
        p50 = (e[mk] < 0.5 * base).mean(axis=1)
        ys = y[mk]
        rows["p50_5 (mata tertutup)"][s] = auc(ys, p50)
        rows["ear_mean (dibalik)"][s] = auc(ys, -e[mk].mean(axis=1))
        rows["jitter"][s] = auc(ys, jitter[mk])
        rows["mar_mean"][s] = auc(ys, m[mk].mean(axis=1))
    df = pd.DataFrame(rows).T
    df["rata2"] = df.mean(axis=1)
    pd.set_option("display.width", 200)
    print("AUC per subjek (drowsy vs alert), satu fitur sederhana per baris:")
    print(df.round(3).to_string())
    print("\nCara membaca:")
    print(" - p50_5 harus ~ baseline_loso (31=0.94 33=0.94 34=0.81 35=0.93 36=0.40) pada data tanpa kalibrasi.")
    print(" - jitter / mar_mean jauh dari 0.5 pada banyak subjek = ciri rekaman yang bisa dihafal model.")


if __name__ == "__main__":
    main()