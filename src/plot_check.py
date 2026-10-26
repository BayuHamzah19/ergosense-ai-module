"""
plot_check.py - Grafik EAR sepanjang video, satu gambar per subjek (alert di atas, drowsy di bawah).

Jalankan dari folder ERGO-SENSE:
    python src/plot_check.py            # semua subjek
    python src/plot_check.py 31 36      # subjek tertentu

Hasil: results/signal_<subjek>.png
Isi tiap panel:
    garis biru    EAR dibagi baseline subjek (1.0 = mata terbuka normal)
    garis merah   0.5  (mata tertutup separuh)
    garis oranye  fraksi waktu EAR < 0.5 baseline dalam 60 detik terakhir (sumbu kanan)
Gambar ini hanya berisi angka, tidak ada wajah.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from check_signal import load

ROOT = Path(__file__).resolve().parent.parent


def main():
    only = set(sys.argv[1:])
    files = sorted(p for p in (ROOT / "landmarks").glob("*.npz") if not p.name.startswith("test_"))
    data = {}
    for f in files:
        subj, label = f.stem.split("_", 1)
        data.setdefault(subj, {})[label] = load(f)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    for subj, vids in sorted(data.items()):
        if only and subj not in only:
            continue
        if "0" not in vids:
            continue
        base = np.nanpercentile(vids["0"][0], 90)
        order = ["0"] + sorted(k for k in vids if k != "0")
        fig, axes = plt.subplots(len(order), 1, figsize=(12, 2.6 * len(order)), squeeze=False)
        for ax, key in zip(axes[:, 0], order):
            e, _, t = vids[key]
            r = e / base
            ax.plot(t / 60, r, lw=0.5, color="tab:blue")
            ax.axhline(0.5, color="tab:red", lw=0.8)
            ax.set_ylim(0, 1.6)
            ax.set_ylabel("EAR / baseline")
            closed = np.where(np.isnan(r), np.nan, (r < 0.5).astype(float))
            roll = pd.Series(closed, index=pd.to_timedelta(t, unit="s")).rolling("60s").mean()
            ax2 = ax.twinx()
            ax2.plot(t / 60, roll.values, color="tab:orange", lw=1.2)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("fraksi tertutup (60 dtk)", color="tab:orange")
            name = "ALERT" if key == "0" else f"DROWSY ({key})"
            ax.set_title(f"Subjek {subj} - {name}", fontsize=10, loc="left")
        axes[-1, 0].set_xlabel("menit")
        fig.tight_layout()
        path = out / f"signal_{subj}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("disimpan:", path)


if __name__ == "__main__":
    main()