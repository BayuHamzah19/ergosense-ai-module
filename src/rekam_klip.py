"""
rekam_klip.py - Rekam klip webcam singkat untuk tes kecocokan Python vs browser, dengan aba-aba di layar.

    python src/rekam_klip.py                       # rekam 60 detik dari webcam -> ~/uji_paritas/klip_bayu.mp4
    python src/rekam_klip.py --out ~/uji_paritas/klip_2.mp4
    python src/rekam_klip.py --detik 30            # klip lebih pendek (uji coba)

Yang terjadi:
  1. Jendela pratinjau terbuka. Ada hitung mundur 5 detik (BELUM direkam).
  2. Rekaman 60 detik dimulai. Lihat layar seperti biasa.
  3. Tiga kali (sekitar detik 15, 30, dan 45) muncul aba-aba "PEJAMKAN MATA". Pejamkan mata 2 detik, lalu buka.
  4. Rekaman berhenti sendiri dan berkas disimpan. Tekan Q kapan saja untuk membatalkan.
Teks aba-aba hanya tampil di pratinjau, TIDAK ikut tersimpan di video.
Berkas disimpan di luar folder proyek supaya tidak ikut ke zip cadangan. Hapus setelah tes selesai.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import cv2

OUT_DEFAULT = Path.home() / "uji_paritas" / "klip_bayu.mp4"


def aba_aba(t, total):
    """Teks aba-aba untuk detik ke-t (hanya tampilan)."""
    marks = [total * 0.25, total * 0.5, total * 0.75]
    for m in marks:
        if m - 3 <= t < m:
            return f"BERSIAP... {math.ceil(m - t)}", (0, 200, 255)
        if m <= t < m + 2:
            return "PEJAMKAN MATA SEKARANG", (0, 0, 255)
        if m + 2 <= t < m + 3:
            return "buka mata", (0, 200, 0)
    return "lihat layar seperti biasa", (255, 255, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--detik", type=float, default=60)
    ap.add_argument("--sumber", default="0", help="nomor webcam (default 0) atau berkas video (hanya untuk uji)")
    ap.add_argument("--tanpa-pratinjau", action="store_true", help="tanpa jendela (hanya untuk uji)")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    src = int(args.sumber) if args.sumber.isdigit() else args.sumber
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("Webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera (Zoom, browser) lalu coba lagi.")
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Tidak ada gambar dari kamera.")
    h, w = frame.shape[:2]
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        sys.exit("Gagal membuat berkas video.")
    show = not args.tanpa_pratinjau

    if show:   # hitung mundur 5 detik, belum merekam
        t0 = time.time()
        while time.time() - t0 < 5:
            ok, frame = cap.read()
            if not ok:
                break
            view = frame.copy()
            cv2.putText(view, f"Rekaman mulai dalam {5 - (time.time() - t0):.0f} detik. Hadap kamera, jarak sekitar 50-60 cm.",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("rekam_klip (Q = batal)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                cap.release(); writer.release(); cv2.destroyAllWindows(); out.unlink(missing_ok=True)
                sys.exit("Dibatalkan.")

    start, written, grabbed = time.time(), 0, 0
    cancelled = False
    while True:
        elapsed = time.time() - start
        if elapsed >= args.detik:
            break
        ok, frame = cap.read()
        if not ok:
            break
        grabbed += 1
        # tulis frame sesuai waktu nyata supaya durasi video benar walau webcam lebih lambat dari fps nominal
        target = int(elapsed * args.fps) + 1
        while written < target:
            writer.write(frame)
            written += 1
        if show:
            msg, col = aba_aba(elapsed, args.detik)
            view = frame.copy()
            cv2.putText(view, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
            cv2.putText(view, f"{elapsed:4.0f} / {args.detik:.0f} detik", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("rekam_klip (Q = batal)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                cancelled = True
                break
    cap.release(); writer.release()
    if show:
        cv2.destroyAllWindows()
    if cancelled:
        out.unlink(missing_ok=True)
        sys.exit("Dibatalkan. Berkas tidak disimpan.")
    dur = time.time() - start
    print(f"Selesai. Disimpan: {out}")
    print(f"Durasi {dur:.1f} detik, {grabbed} frame diterima kamera (sekitar {grabbed / max(dur, 1e-9):.0f} frame per detik), "
          f"ukuran {w}x{h}, {out.stat().st_size / 1e6:.1f} MB")
    if grabbed / max(dur, 1e-9) < 12:
        print("PERINGATAN: kamera hanya memberi < 12 frame per detik. Tutup aplikasi lain dan perbaiki cahaya, lalu rekam ulang.")
    print("Ingat: kirim berkas ini ke Ardhi sebagai DOKUMEN di WhatsApp, dan hapus setelah tes selesai.")


if __name__ == "__main__":
    main()