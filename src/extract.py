"""
extract.py - Video -> 20 landmark (12 mata + 8 bibir dalam) per frame.

Keluaran per video: file .npz berisi
    pts        : (T, 20, 2) float32, koordinat piksel; NaN jika wajah tidak terdeteksi
    t          : (T,) detik, timestamp tiap frame yang dipakai
    native_fps : FPS asli video
    target_fps : FPS hasil resampling (15)

Urutan 20 titik:
    0-5   mata kanan  p1..p6
    6-11  mata kiri   p1..p6
    12-19 bibir dalam m1..m8
Hanya angka landmark yang disimpan, TIDAK ada frame/gambar wajah.
"""
from pathlib import Path
import urllib.request

import cv2
import numpy as np

# --- indeks landmark MediaPipe Face Mesh (478 titik) -------------------------
RIGHT_EYE = [33, 160, 158, 133, 153, 144]        # p1..p6
LEFT_EYE = [362, 385, 387, 263, 373, 380]        # p1..p6
LIPS = [78, 81, 13, 311, 308, 402, 14, 178]      # m1..m8
IDX = RIGHT_EYE + LEFT_EYE + LIPS                # 20 titik
TARGET_FPS = 15

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Pastikan file model ada. Coba unduh otomatis jika belum ada."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh model Face Landmarker ke {path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:
        if path.exists():
            path.unlink()
        raise RuntimeError(
            "Gagal mengunduh model otomatis.\n"
            f"Unduh manual dari:\n  {MODEL_URL}\n"
            f"lalu simpan sebagai:\n  {path}\n"
            f"(error asli: {e})"
        )
    return path


def make_landmarker(model_path: Path = MODEL_PATH):
    """Buat FaceLandmarker (Tasks API) untuk mode video."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = ensure_model(model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options), mp


def extract(video_path, out_path, target_fps=TARGET_FPS, max_seconds=None):
    """Proses satu video dan simpan landmark ke out_path (.npz).

    max_seconds: batasi durasi yang diproses (berguna untuk uji cepat).
    Mengembalikan dict ringkasan.
    """
    video_path, out_path = Path(video_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video tidak bisa dibuka: {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS)
    if not native or native <= 0:
        native = 30.0

    landmarker, mp = make_landmarker()
    pts, times = [], []
    next_t, i, last_ts_ms = 0.0, 0, -1

    try:
        # grab() murah; retrieve() (decode penuh) hanya untuk frame yang dipilih
        while cap.grab():
            t = i / native
            i += 1
            if max_seconds is not None and t > max_seconds:
                break
            if t < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            next_t += 1.0 / target_fps

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = max(int(t * 1000), last_ts_ms + 1)   # harus naik terus
            last_ts_ms = ts_ms
            res = landmarker.detect_for_video(mp_img, ts_ms)

            if res.face_landmarks:
                lm = res.face_landmarks[0]
                pts.append([[lm[k].x * w, lm[k].y * h] for k in IDX])
            else:
                pts.append(np.full((len(IDX), 2), np.nan))   # hilang, bukan 0
            times.append(t)
    finally:
        cap.release()
        landmarker.close()

    pts = np.array(pts, dtype=np.float32).reshape(-1, len(IDX), 2)
    np.savez_compressed(out_path, pts=pts, t=np.array(times, dtype=np.float32),
                        native_fps=native, target_fps=target_fps)
    missing = float(np.isnan(pts[:, 0, 0]).mean()) if len(pts) else 1.0
    return {"frames": len(pts), "native_fps": native, "missing_frac": missing}


# --- fitur per frame ----------------------------------------------------------
def ear(eye):
    """eye: (T, 6, 2) berurutan p1..p6 -> EAR (T,)"""
    v = (np.linalg.norm(eye[:, 1] - eye[:, 5], axis=1)
         + np.linalg.norm(eye[:, 2] - eye[:, 4], axis=1))
    h = np.linalg.norm(eye[:, 0] - eye[:, 3], axis=1)
    return v / (2.0 * h)


def mar(mouth):
    """mouth: (T, 8, 2) berurutan m1..m8 -> MAR (T,)"""
    v = (np.linalg.norm(mouth[:, 1] - mouth[:, 7], axis=1)
         + np.linalg.norm(mouth[:, 2] - mouth[:, 6], axis=1)
         + np.linalg.norm(mouth[:, 3] - mouth[:, 5], axis=1))
    h = np.linalg.norm(mouth[:, 0] - mouth[:, 4], axis=1)
    return v / (2.0 * h)


def ear_mar_from_pts(pts):
    """pts: (T, 20, 2) -> (EAR rata-rata dua mata, MAR), masing-masing (T,)"""
    e = (ear(pts[:, 0:6]) + ear(pts[:, 6:12])) / 2.0
    m = mar(pts[:, 12:20])
    return e, m
