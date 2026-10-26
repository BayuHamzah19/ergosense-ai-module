"""
test_one.py - Uji SATU video: ekstraksi landmark -> grafik EAR & MAR.

Jalankan dari folder ERGO-SENSE:
    python src/test_one.py                      # default: subjek 31, label 0
    python src/test_one.py --subject 31 --label 10
    python src/test_one.py --video Fold3_part2/31/0.mp4 --seconds 60
    python src/test_one.py --subject 31 --label 0 --debug-frame   # simpan 1 gambar titik (cek manual)

Cek hasil:
  - grafik EAR stabil ~0.25-0.35 dengan turun tajam sesekali (kedipan)  -> benar
  - frame hilang tinggi (>20%)                                          -> wajah sering tak terdeteksi
  - grafik datar/kacau                                                  -> cek indeks landmark (--debug-frame)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extract import IDX, extract, ear_mar_from_pts, make_landmarker

ROOT = Path(__file__).resolve().parent.parent


def find_video(args):
    if args.video:
        p = Path(args.video)
        return p if p.is_absolute() else ROOT / p
    for folder in sorted(ROOT.glob("Fold*")):
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            p = folder / str(args.subject) / f"{args.label}{ext}"
            if p.exists():
                return p
    sys.exit(f"Video tidak ditemukan untuk subjek {args.subject}, label {args.label}.")


def save_debug_frame(video, out):
    """Simpan SATU frame dengan 20 titik digambar, untuk cek visual indeks landmark.
    Gambar ini memuat wajah: pakai hanya untuk debugging lokal, lalu hapus."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Gagal membaca frame debug.")
        return
    landmarker, mp = make_landmarker()
    h, w = frame.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = landmarker.detect_for_video(mp_img, 1)
    landmarker.close()
    if not res.face_landmarks:
        print("Wajah tidak terdeteksi di frame debug.")
        return
    lm = res.face_landmarks[0]
    for n, k in enumerate(IDX):
        x, y = int(lm[k].x * w), int(lm[k].y * h)
        color = (0, 255, 0) if n < 12 else (0, 0, 255)   # hijau = mata, merah = bibir
        cv2.circle(frame, (x, y), 3, color, -1)
        cv2.putText(frame, str(n), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(str(out), frame)
    print(f"Frame debug disimpan: {out}  (hapus setelah dicek)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="31")
    ap.add_argument("--label", default="0")
    ap.add_argument("--video", default=None, help="path video langsung (menimpa subject/label)")
    ap.add_argument("--seconds", type=float, default=60, help="durasi yang diproses (detik)")
    ap.add_argument("--debug-frame", action="store_true")
    args = ap.parse_args()

    video = find_video(args)
    tag = f"{video.parent.name}_{video.stem}"
    (ROOT / "landmarks").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    npz = ROOT / "landmarks" / f"test_{tag}.npz"

    print(f"Video : {video}")
    t0 = time.time()
    info = extract(video, npz, max_seconds=args.seconds)
    dt = time.time() - t0
    print(f"FPS asli       : {info['native_fps']:.2f}")
    print(f"Frame diproses : {info['frames']} (resampling 15 FPS)")
    print(f"Frame hilang   : {info['missing_frac'] * 100:.1f} %")
    print(f"Waktu proses   : {dt:.1f} detik untuk {args.seconds:.0f} detik video")

    d = np.load(npz)
    pts, t = d["pts"], d["t"]
    e, m = ear_mar_from_pts(pts)
    print(f"EAR  min/median/maks : {np.nanmin(e):.3f} / {np.nanmedian(e):.3f} / {np.nanmax(e):.3f}"
          if np.isfinite(e).any() else "EAR: semua NaN (wajah tidak terdeteksi)")

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(t, e, lw=0.9)
    ax[0].set_ylabel("EAR")
    ax[0].set_title(f"{tag}  (hilang {info['missing_frac'] * 100:.1f}%)")
    ax[1].plot(t, m, lw=0.9, color="tab:red")
    ax[1].set_ylabel("MAR")
    ax[1].set_xlabel("detik")
    fig.tight_layout()
    out = ROOT / "results" / f"ear_{tag}.png"
    fig.savefig(out, dpi=150)
    print(f"Grafik disimpan: {out}")

    if args.debug_frame:
        save_debug_frame(video, ROOT / "results" / f"debug_{tag}.jpg")


if __name__ == "__main__":
    main()
