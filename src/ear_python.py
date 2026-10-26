"""
ear_python.py - Hitung deret EAR dari sebuah klip video dengan jalur Python (extract.py), simpan ke CSV (t,ear).

Dipakai untuk tes kecocokan Python vs browser: klip yang SAMA diproses di sini dan di aplikasi browser.

    python src/ear_python.py klip.mp4 --seconds 60 --out ear_python.csv
    python src/ear_python.py landmarks/test_31_0.npz --out ear_python.csv    # dari .npz yang sudah ada

Kolom: t (detik sejak awal klip), ear (kosong bila wajah tidak terdeteksi).
Catatan privasi: yang disimpan hanya angka. Klip rekaman anggota tim sendiri (dengan persetujuan), bukan video dataset.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from extract import ear_mar_from_pts


def npz_to_csv(npz_path, csv_path):
    d = np.load(npz_path)
    ear, _ = ear_mar_from_pts(d["pts"])
    t = d["t"].astype(float)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("t,ear\n")
        for ti, ei in zip(t, ear):
            fh.write(f"{float(ti)!r},{'' if not np.isfinite(ei) else repr(float(ei))}\n")
    return len(t), float(np.mean(~np.isfinite(ear)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="klip video (.mp4 dll) atau file landmark .npz")
    ap.add_argument("--out", default="ear_python.csv")
    ap.add_argument("--seconds", type=float, default=60)
    args = ap.parse_args()
    if args.input.lower().endswith(".npz"):
        n, miss = npz_to_csv(args.input, args.out)
    else:
        from extract import extract
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "clip.npz"
            extract(args.input, npz, max_seconds=args.seconds)
            n, miss = npz_to_csv(npz, args.out)
    print(f"{n} baris ditulis ke {args.out}; frame tanpa wajah: {miss:.1%}")


if __name__ == "__main__":
    main()
