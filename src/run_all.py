"""
run_all.py - Ekstraksi landmark untuk SEMUA video di inventory.csv (default: label 0 dan 10).

Jalankan dari folder ERGO-SENSE, SETELAH inventory.py dan test_one.py berhasil:
    python src/run_all.py
    python src/run_all.py --labels 0 10 5      # termasuk label 5
    python src/run_all.py --limit 2            # uji: hanya 2 video pertama

Aman dihentikan (Ctrl+C) dan dilanjutkan: video yang sudah punya .npz dilewati.
Keluaran: landmarks/<subjek>_<label>.npz  dan  landmarks/summary.csv
"""
import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from extract import extract

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "landmarks"


def read_inventory():
    inv = ROOT / "inventory.csv"
    if not inv.exists():
        sys.exit("inventory.csv belum ada. Jalankan dulu: python src/inventory.py")
    with open(inv, newline="") as fh:
        return list(csv.DictReader(fh))


def write_summary():
    """Ringkasan dari SEMUA .npz yang ada (termasuk hasil run sebelumnya)."""
    rows = []
    for f in sorted(OUT_DIR.glob("*.npz")):
        if f.name.startswith("test_"):
            continue
        d = np.load(f)
        pts = d["pts"]
        subj, label = f.stem.split("_", 1)
        missing = float(np.isnan(pts[:, 0, 0]).mean()) if len(pts) else 1.0
        rows.append({
            "subject": subj, "label": label,
            "native_fps": round(float(d["native_fps"]), 2),
            "frames_15fps": len(pts),
            "minutes": round(len(pts) / float(d["target_fps"]) / 60, 2),
            "missing_pct": round(missing * 100, 1),
        })
    if rows:
        with open(OUT_DIR / "summary.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["0", "10"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all-subjects", action="store_true",
                    help="proses juga subjek yang tidak punya pasangan video 0 dan 10")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    inv = read_inventory()
    if not args.all_subjects:
        # hanya subjek yang punya video alert (0) DAN drowsy (10)
        have = {}
        for r in inv:
            have.setdefault(r["subject"], set()).add(r["label"])
        ok = {s for s, l in have.items() if "0" in l and any(x.startswith("10") for x in l)}
        skipped = sorted(set(have) - ok)
        if skipped:
            print(f"Subjek dilewati (tidak punya 0 dan 10): {', '.join(skipped)}")
        inv = [r for r in inv if r["subject"] in ok]
    wanted = set(args.labels)
    todo = [r for r in inv if r["label"] in wanted or ("10" in wanted and r["label"].startswith("10_"))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Video yang akan diproses: {len(todo)} (label {args.labels})")

    failed, t_start = [], time.time()
    for n, r in enumerate(todo, 1):
        out = OUT_DIR / f"{r['subject']}_{r['label']}.npz"
        if out.exists():
            print(f"[{n}/{len(todo)}] lewati (sudah ada): {out.name}")
            continue
        print(f"[{n}/{len(todo)}] proses {out.name} ...", flush=True)
        t0 = time.time()
        try:
            info = extract(r["path"], out)
            print(f"    selesai {time.time() - t0:.0f} dtk | FPS asli {info['native_fps']:.1f} "
                  f"| frame hilang {info['missing_frac'] * 100:.1f}%", flush=True)
        except KeyboardInterrupt:
            if out.exists():
                out.unlink()          # buang file setengah jadi
            print("\nDihentikan. Jalankan lagi untuk melanjutkan.")
            break
        except Exception:
            failed.append(r["path"])
            print(f"    GAGAL: {r['path']}")
            traceback.print_exc()

    rows = write_summary()
    print(f"\nTotal waktu: {(time.time() - t_start) / 60:.1f} menit")
    print(f"File landmark tersedia: {len(rows)}  -> {OUT_DIR / 'summary.csv'}")
    high = [x for x in rows if x["missing_pct"] > 20]
    if high:
        print("PERINGATAN: frame hilang > 20% pada:", ", ".join(f"{x['subject']}_{x['label']}" for x in high))
    if failed:
        print("Video gagal diproses:", *failed, sep="\n  ")


if __name__ == "__main__":
    main()
