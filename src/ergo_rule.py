"""
ergo_rule.py - Mesin aturan EAR ERGO-SENSE (implementasi referensi, v0.1).

Input : deretan (t detik, EAR rata-rata dua mata atau None bila wajah tidak terdeteksi).
Output: satu dict per detik (lihat RuleEngine.update) berisi status, skor IKM, dan fitur.

Alur:
  1. Kalibrasi 30 detik pertama -> baseline = persentil ke-90 EAR valid. Sah bila >= 70% waktu valid.
  2. Warm-up 30 detik (jendela fitur harus terisi data pasca-kalibrasi), status 'uncertain'.
  3. Tiap detik: p50_30 = porsi waktu valid dengan EAR/baseline < 0.5 dalam 30 detik terakhir (berbobot waktu),
     maxclose_30_s = penutupan terus-menerus terpanjang, valid_frac_30 = waktu valid / 30 detik.
  4. IKM = clamp(100 * p50_30 / (ikm_divisor * k), 0, 100). Status: green < 33 <= yellow < 66 <= red.
     'uncertain' bila belum kalibrasi, warm-up, wajah hilang > 10 detik, atau valid_frac_30 < 0.8.
  5. Peringatan (alert): IKM merah bertahan >= 10 detik, lalu jeda >= 5 menit; aktif lagi setelah IKM < 50.

Semua waktu memakai timestamp (detik), bukan jumlah frame. Parameter ada di ergo_rule_config.json.
Parameter bertanda 'provisional' adalah tebakan desain, bukan hasil data.
"""
import json
import math
import sys
from collections import deque
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "ergo_rule_config.json"

# Indeks landmark MediaPipe Face Mesh (478 titik): p1..p6 tiap mata
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ear_one_eye(p):
    """p: 6 titik (x, y) dalam PIKSEL, urutan p1..p6 -> EAR satu mata."""
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]))


def ear_two_eyes(right, left):
    """Rata-rata EAR mata kanan dan kiri (titik dalam piksel)."""
    return (ear_one_eye(right) + ear_one_eye(left)) / 2.0


def ear_from_landmarks(landmarks, width, height):
    """landmarks: daftar objek/dict/tuple dengan x,y ternormalisasi 0..1 (hasil MediaPipe). Mengembalikan EAR atau None."""
    def pt(i):
        l = landmarks[i]
        x, y = (l["x"], l["y"]) if isinstance(l, dict) else (getattr(l, "x", None), getattr(l, "y", None))
        if x is None:
            x, y = l[0], l[1]
        return (x * width, y * height)          # piksel, bukan koordinat ternormalisasi
    try:
        e = ear_two_eyes([pt(i) for i in RIGHT_EYE], [pt(i) for i in LEFT_EYE])
    except ZeroDivisionError:
        return None
    return e if math.isfinite(e) else None


def percentile(values, q):
    """Persentil dengan interpolasi linear (sama dengan numpy default)."""
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class RuleEngine:
    def __init__(self, cfg=None, sensitivity=None):
        self.cfg = cfg or load_config()
        self.set_sensitivity(sensitivity or self.cfg["default_sensitivity"])
        self.reset()

    def set_sensitivity(self, name):
        self.sensitivity = name
        self.k = self.cfg["sensitivity"][name]

    def reset(self):
        self.t0 = None                 # awal fase kalibrasi
        self.baseline = None
        self.calibrated_at = None
        self.calib_ears = []
        self.calib_valid_time = 0.0
        self.prev = None               # sampel sebelumnya (t, ear atau None)
        self.win = deque()             # (t, dt, status_sampel, rasio) ; status_sampel: 'closed'|'open'|'invalid'
        self.next_tick = None
        self.last_valid_t = None
        self.last_ear = None
        self.last_ratio = None
        self.red_since = None
        self.armed = True
        self.last_alert_t = -math.inf

    def recalibrate(self, t):
        keep = self.sensitivity
        self.reset()
        self.sensitivity = keep
        self.t0 = t

    # ------------------------------------------------------------------
    def update(self, t, ear):
        """Masukkan satu sampel. Mengembalikan dict (satu per detik) atau None bila belum waktunya."""
        cfg = self.cfg
        events = []
        if self.t0 is None:
            self.t0 = t
            self.next_tick = t + cfg["tick_s"]
        if ear is not None and not (isinstance(ear, (int, float)) and math.isfinite(ear)):
            ear = None
        if ear is not None:
            self.last_valid_t, self.last_ear = t, ear

        # selesaikan sampel sebelumnya: bobot waktunya = selisih ke sampel ini, dibatasi dt_cap_s
        if self.prev is not None:
            pt, pe = self.prev
            dt = min(t - pt, cfg["dt_cap_s"])
            if dt > 0:
                self._finalize(pt, dt, pe)
        self.prev = (t, ear)

        # akhir fase kalibrasi?
        if self.baseline is None and t - self.t0 >= cfg["calibration_s"]:
            valid_frac = self.calib_valid_time / cfg["calibration_s"]
            if valid_frac >= cfg["calibration_min_valid_frac"] and self.calib_ears:
                self.baseline = percentile(self.calib_ears, cfg["baseline_percentile"])
                self.calibrated_at = t
                self.win.clear()
                events.append({"type": "calibration_done", "baseline_ear": self.baseline, "valid_frac": min(valid_frac, 1.0)})
            else:
                events.append({"type": "calibration_failed", "valid_frac": min(valid_frac, 1.0)})
                keep_next = self.next_tick
                self.recalibrate(t)
                self.next_tick = keep_next
                self.prev = (t, ear)
        if self.baseline is not None and ear is not None:
            self.last_ratio = ear / self.baseline

        if t >= self.next_tick:
            out = self._tick(t, events)
            while self.next_tick <= t:
                self.next_tick += cfg["tick_s"]
            return out
        if events:                     # event tanpa tick: keluarkan sebagai tick khusus
            return self._tick(t, events)
        return None

    def _finalize(self, pt, dt, pe):
        if self.baseline is None:
            if self.t0 is not None and pt >= self.t0:
                if pe is not None:
                    self.calib_ears.append(pe)
                    self.calib_valid_time += dt
            return
        if pe is None:
            self.win.append((pt, dt, "invalid", None))
        else:
            r = pe / self.baseline
            self.win.append((pt, dt, "closed" if r < self.cfg["closed_ratio"] else "open", r))

    def _tick(self, now, events):
        cfg = self.cfg
        W = cfg["window_s"]
        while self.win and self.win[0][0] <= now - W:
            self.win.popleft()
        out = {"t": now, "calibrated": self.baseline is not None, "baseline_ear": self.baseline,
               "ear": self.last_ear, "ear_ratio": self.last_ratio, "sensitivity": self.sensitivity,
               "p50_30": None, "maxclose_30_s": None, "valid_frac_30": None, "ikm": None,
               "status": "uncertain", "reason": None, "alert": False, "events": events}
        if self.baseline is None:
            out["reason"] = "calibrating"
            self.red_since = None
            return out

        valid_time = closed_time = run = longest = 0.0
        for (_, dt, st, _r) in self.win:
            if st == "invalid":
                run = 0.0
                continue
            valid_time += dt
            if st == "closed":
                closed_time += dt
                run += dt
                longest = max(longest, run)
            else:
                run = 0.0
        valid_frac = min(valid_time / W, 1.0)
        out["valid_frac_30"] = valid_frac
        out["maxclose_30_s"] = longest
        out["p50_30"] = (closed_time / valid_time) if valid_time > 0 else None

        if now - self.calibrated_at < cfg["warmup_s"]:
            out["reason"] = "warming_up"
        elif self.last_valid_t is None or now - self.last_valid_t > cfg["no_face_uncertain_s"]:
            out["reason"] = "no_face"
        elif valid_frac < cfg["min_valid_frac"] or out["p50_30"] is None:
            out["reason"] = "low_valid"
        else:
            ikm = max(0.0, min(100.0, 100.0 * out["p50_30"] / (cfg["ikm_divisor"] * self.k)))
            out["ikm"] = ikm
            out["status"] = "red" if ikm >= cfg["ikm_red"] else ("yellow" if ikm >= cfg["ikm_yellow"] else "green")

        # peringatan
        a = cfg["alert"]
        if out["status"] == "red":
            if self.red_since is None:
                self.red_since = now
            if self.armed and now - self.red_since >= a["red_hold_s"] and now - self.last_alert_t >= a["cooldown_s"]:
                out["alert"] = True
                self.armed = False
                self.last_alert_t = now
        else:
            self.red_since = None
        if out["ikm"] is not None and out["ikm"] < a["rearm_ikm"]:
            self.armed = True
        return out


def run_csv(path, cfg=None, sensitivity=None):
    eng = RuleEngine(cfg, sensitivity)
    ticks = []
    with open(path, encoding="utf-8") as fh:
        next(fh)                                   # header: t,ear
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t, _, e = line.partition(",")
            out = eng.update(float(t), float(e) if e.strip() != "" else None)
            if out:
                ticks.append(out)
    return ticks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("pemakaian: python ergo_rule.py input.csv   (kolom: t,ear ; ear kosong = wajah tidak terdeteksi)")
    for tk in run_csv(sys.argv[1], sensitivity=sys.argv[2] if len(sys.argv) > 2 else None):
        print(json.dumps(tk))
