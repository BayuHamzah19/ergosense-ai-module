"""
inventory.py - Daftar semua video dataset (subjek, label, FPS, durasi, resolusi).

Jalankan dari folder ERGO-SENSE:
    python src/inventory.py                       # otomatis cari Fold3_part1 / Fold3_part2
    python src/inventory.py Fold3_part2 Fold3_part1
    python src/inventory.py /path/lain/Fold1_part1

Struktur yang diharapkan:  <folder>/<subjek>/<label>.mp4   (label: 0, 5, atau 10)
Keluaran: inventory.csv + ringkasan di layar.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def main():
    args = sys.argv[1:]
    if args:
        folders = [Path(a) if Path(a).is_absolute() else ROOT / a for a in args]
    else:
        folders = [p for p in sorted(ROOT.glob("Fold*")) if p.is_dir()]
    if not folders:
        sys.exit("Tidak ada folder Fold* di proyek. Beri path folder sebagai argumen.")

    rows = []
    for folder in folders:
        if not folder.is_dir():
            print(f"[lewati] bukan folder: {folder}")
            continue
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                print(f"[gagal dibuka] {f}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            minutes = round(n / fps / 60, 2) if fps else ""
            rows.append({
                "subject": f.parent.name,
                "label": f.stem,                 # "0", "5", atau "10"
                "fps": round(fps, 2),
                "frames": int(n),
                "minutes": minutes,
                "resolution": f"{w}x{h}",
                "path": str(f),
            })

    if not rows:
        sys.exit("Tidak ada video ditemukan. Cek path dan ekstensi file.")

    out = ROOT / "inventory.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---- ringkasan ----
    by_subj = defaultdict(set)
    for r in rows:
        by_subj[r["subject"]].add(r["label"])

    print(f"\nTotal video : {len(rows)}")
    print(f"Total subjek: {len(by_subj)}\n")
    print(f"{'subjek':<8}{'label tersedia':<18}{'0 & 10?'}")
    usable = []
    for s in sorted(by_subj, key=lambda x: (len(x), x)):
        labels = sorted(by_subj[s], key=int) if all(l.isdigit() for l in by_subj[s]) else sorted(by_subj[s])
        both = "0" in by_subj[s] and "10" in by_subj[s]
        if both:
            usable.append(s)
        print(f"{s:<8}{','.join(labels):<18}{'YA' if both else '-'}")

    fps_vals = sorted({r['fps'] for r in rows})
    print(f"\nSubjek dengan video 0 dan 10 (bisa dipakai alert vs drowsy): {len(usable)}")
    print(f"FPS asli yang ditemukan: {fps_vals}")
    print(f"Disimpan ke: {out}")
    unknown = {r['label'] for r in rows} - {"0", "5", "10"}
    if unknown:
        print(f"PERINGATAN: nama file tak terduga (bukan 0/5/10): {sorted(unknown)}")


if __name__ == "__main__":
    main()
