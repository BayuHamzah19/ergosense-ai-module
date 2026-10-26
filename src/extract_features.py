"""Ekstraksi fitur per frame (EAR, MAR, pose kepala) dari video dataset.

Input : manifest.csv dengan kolom  video,subject,label
Output: features/<subject>__<nama_video>.csv  (kolom t,ear,mar,pitch,yaw,roll)

Contoh:
  python src/extract_features.py --manifest data/manifest.csv \
      --model models/face_landmarker.task --out features --target-fps 15
"""
import argparse
import csv
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

# PENTING: samakan indeks dan rumus EAR dengan src/ear_python.py
RIGHT = [33, 160, 158, 133, 153, 144]
LEFT = [362, 385, 387, 263, 373, 380]
# Mulut: sudut kiri, 2 titik atas, sudut kanan, 2 titik bawah (bibir dalam)
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]


def _d(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_ear(pts):
    return (_d(pts[1], pts[5]) + _d(pts[2], pts[4])) / (2 * _d(pts[0], pts[3]))


def frame_features(lm, w, h, matrix):
    px = lambda idx: [(lm[i].x * w, lm[i].y * h) for i in idx]
    ear = (eye_ear(px(RIGHT)) + eye_ear(px(LEFT))) / 2
    m = px(MOUTH)
    mar = (_d(m[1], m[7]) + _d(m[2], m[6]) + _d(m[3], m[5])) / (2 * _d(m[0], m[4]))
    pitch = yaw = roll = np.nan
    if matrix is not None:
        R = np.array(matrix)[:3, :3]
        R = R / np.linalg.norm(R, axis=0, keepdims=True)  # buang skala
        sy = np.hypot(R[0, 0], R[1, 0])
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return ear, mar, pitch, yaw, roll


def process(video_path, model_path, target_fps):
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / target_fps)))
    rows, idx, last_ts = [], 0, -1
    with vision.FaceLandmarker.create_from_options(opts) as lmk:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = idx / fps
                ts = int(t * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = lmk.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                if res.face_landmarks:
                    mats = res.facial_transformation_matrixes
                    f = frame_features(res.face_landmarks[0], w, h,
                                       mats[0] if mats else None)
                    rows.append([round(t, 4), *[round(float(x), 5) for x in f]])
                else:
                    rows.append([round(t, 4), "", "", "", "", ""])
            idx += 1
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="features")
    ap.add_argument("--target-fps", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.manifest, newline="") as f:
        items = list(csv.DictReader(f))
    for n, it in enumerate(items, 1):
        name = f"{it['subject']}__{os.path.splitext(os.path.basename(it['video']))[0]}.csv"
        out_path = os.path.join(args.out, name)
        if os.path.exists(out_path):
            print(f"[{n}/{len(items)}] lewati (sudah ada): {name}")
            continue
        try:
            rows = process(it["video"], args.model, args.target_fps)
        except Exception as e:  # satu video rusak tidak menghentikan semuanya
            print(f"[{n}/{len(items)}] GAGAL {it['video']}: {e}")
            continue
        with open(out_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "ear", "mar", "pitch", "yaw", "roll"])
            wr.writerows(rows)
        noface = sum(1 for r in rows if r[1] == "") / max(1, len(rows))
        print(f"[{n}/{len(items)}] {name}: {len(rows)} frame, tanpa wajah {noface:.1%}")


if __name__ == "__main__":
    main()
