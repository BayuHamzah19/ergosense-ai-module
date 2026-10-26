"""
test_ergo_rule.py - Uji perilaku mesin aturan dan uji kecocokan Python vs JavaScript.

    python test_ergo_rule.py            # semua uji (uji JS dilewati bila node tidak ada)

Data uji SINTETIS, jadi hanya membuktikan logika dan kecocokan dua implementasi,
BUKAN akurasi deteksi. Akurasi diuji pada data nyata lewat eval_rule.py.
"""
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ergo_rule import RuleEngine, load_config, run_csv  # noqa: E402


def make_series(seed=0, dur=240.0, drowsy=(100.0, 200.0), no_face=((60.0, 64.0), (210.0, 225.0)), fail_calib=False):
    """Deret (t, ear atau None) tak seragam ~15 FPS. Fase drowsy: mata tertutup sekitar 30% waktu."""
    rnd = random.Random(seed)
    t, out, next_blink, closed_until = 0.0, [], 3.0, -1.0
    while t < dur:
        t += rnd.uniform(0.05, 0.09)
        in_nf = any(a <= t < b for a, b in no_face) or (fail_calib and t < 25.0 and rnd.random() < 0.9)
        if in_nf:
            out.append((t, None)); continue
        ear = 0.30 + rnd.gauss(0, 0.004)
        if t >= next_blink:
            if drowsy[0] <= t < drowsy[1] and rnd.random() < 0.5:
                closed_until = t + rnd.uniform(1.0, 3.0)            # penutupan lama
                next_blink = closed_until + rnd.uniform(2.0, 4.0)
            else:
                closed_until = t + 0.15                              # kedipan biasa
                next_blink = t + rnd.uniform(4.5, 6.5)
        if t < closed_until:
            ear = 0.08 + rnd.gauss(0, 0.004)
        out.append((t, ear))
    return out


def run_engine(series, **kw):
    eng = RuleEngine(load_config(), kw.get("sensitivity"))
    ticks = []
    for t, e in series:
        o = eng.update(t, e)
        if o:
            ticks.append(o)
    return ticks


def check(cond, msg):
    print(("  OK   " if cond else "  GAGAL ") + msg)
    if not cond:
        check.failed += 1
check.failed = 0


def test_behaviour():
    print("Uji perilaku:")
    ticks = run_engine(make_series())
    done = [e for tk in ticks for e in tk["events"] if e["type"] == "calibration_done"]
    check(len(done) == 1 and 0.29 < done[0]["baseline_ear"] < 0.33, f"kalibrasi sekali, baseline wajar ({done[0]['baseline_ear']:.3f})")
    early = [tk for tk in ticks if tk["t"] < 60]
    check(all(tk["status"] == "uncertain" for tk in early), "status uncertain selama kalibrasi dan warm-up (t < 60 detik)")
    fresh = [tk for tk in ticks if 66 <= tk["t"] < 98]
    check(fresh and all(tk["status"] == "green" for tk in fresh), "hijau di fase segar (kecuali jeda wajah hilang pendek)")
    late = [tk for tk in ticks if 150 <= tk["t"] < 200 and tk["status"] != "uncertain"]
    check(late and sum(tk["status"] == "red" for tk in late) / len(late) > 0.5, "mayoritas merah di fase drowsy")
    alerts = [tk["t"] for tk in ticks if tk["alert"]]
    check(len(alerts) == 1, f"peringatan muncul tepat sekali (cooldown 5 menit): t = {[round(a) for a in alerts]}")
    nf = [tk for tk in ticks if 221 <= tk["t"] < 224.5]   # wajah kembali di t=225, jadi cek sebelum itu
    check(nf and all(tk["status"] == "uncertain" and tk["reason"] == "no_face" for tk in nf), "wajah hilang > 10 detik -> uncertain (no_face)")
    check(all(not (tk["status"] == "uncertain" and tk["alert"]) for tk in ticks), "status uncertain tidak pernah memicu peringatan")

    tk2 = run_engine(make_series(fail_calib=True))
    ev = [e["type"] for tk in tk2 for e in tk["events"]]
    check("calibration_failed" in ev and "calibration_done" in ev and ev.index("calibration_failed") < ev.index("calibration_done"),
          "kalibrasi gagal (wajah kurang), lalu diulang dan berhasil")

    hi = run_engine(make_series(), sensitivity="high")
    lo = run_engine(make_series(), sensitivity="low")
    r = lambda tks: sum(tk["status"] == "red" for tk in tks)
    check(r(hi) >= r(lo), f"sensitivitas tinggi memberi merah >= sensitivitas rendah ({r(hi)} vs {r(lo)})")


def test_parity():
    print("Uji kecocokan Python vs JavaScript:")
    node = shutil.which("node")
    if not node:
        print("  (dilewati: node tidak ditemukan)"); return
    for name, kw in [("normal", {}), ("kalibrasi_gagal", {"fail_calib": True}), ("sensitivitas_tinggi", {})]:
        series = make_series(seed=3, **{k: v for k, v in kw.items()})
        sens = "high" if name == "sensitivitas_tinggi" else None
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "in.csv"
            csv.write_text("t,ear\n" + "\n".join(f"{t!r},{'' if e is None else repr(e)}" for t, e in series), encoding="utf-8")
            py = run_csv(csv, sensitivity=sens)
            args = [node, str(HERE / "ergo_rule.js"), str(csv)] + ([sens] if sens else [])
            res = subprocess.run(args, capture_output=True, text=True, check=True)
            js = [json.loads(l) for l in res.stdout.splitlines()]
        ok = len(py) == len(js)
        maxdiff, bad = 0.0, 0
        for a, b in zip(py, js):
            for key in ("t", "baseline_ear", "ear", "ear_ratio", "p50_30", "maxclose_30_s", "valid_frac_30", "ikm"):
                x, y = a[key], b[key]
                if (x is None) != (y is None):
                    bad += 1
                elif x is not None:
                    maxdiff = max(maxdiff, abs(x - y))
            for key in ("status", "reason", "alert", "calibrated", "sensitivity"):
                if a[key] != b[key]:
                    bad += 1
            if [e["type"] for e in a["events"]] != [e["type"] for e in b["events"]]:
                bad += 1
        check(ok and bad == 0 and maxdiff < 1e-9, f"{name}: {len(py)} tick, selisih numerik maks {maxdiff:.2e}, beda status/event: {bad}")


if __name__ == "__main__":
    test_behaviour()
    test_parity()
    print("\nSEMUA UJI LOLOS" if check.failed == 0 else f"\n{check.failed} UJI GAGAL")
    sys.exit(1 if check.failed else 0)
