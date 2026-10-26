"""
eval_rule.py - Jalankan mesin aturan (ergo_rule.py) pada landmark NYATA (landmarks/*.npz) dan ukur hasilnya.

Taruh di folder src/ bersama ergo_rule.py dan ergo_rule_config.json, lalu dari folder ERGO-SENSE:
    python src/eval_rule.py                 # 11 subjek, subjek 32 dikeluarkan
    python src/eval_rule.py --sens high     # sensitivitas lain

Tiap rekaman diproses seperti di aplikasi: kalibrasi dari 30 detik pertamanya sendiri (tanpa label), warm-up 30 detik,
lalu satu keluaran per detik. Yang diukur:
  1. AUC p50_30 per subjek (alert vs drowsy). Harus mendekati baseline_loso --calib self30 (kolom 'ref').
     Jika jauh berbeda, ada perbedaan implementasi yang perlu dicari.
  2. Perilaku pada ambang SEMENTARA saat ini: persen detik berstatus kuning-atau-lebih dan merah pada rekaman alert
     (alarm palsu) dan pada rekaman drowsy (terdeteksi), serta jumlah peringatan.
Detik berstatus 'uncertain' tidak dihitung (dilaporkan terpisah).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from baseline_loso import auc
from ergo_rule import RuleEngine, load_config
from extract import ear_mar_from_pts

ROOT = Path(__file__).resolve().parent.parent
REF = {"25": .805, "26": .960, "27": .155, "28": .850, "29": .561, "30": .987,
       "31": .998, "33": 1.000, "34": .984, "35": .968, "36": .387}   # baseline_loso --calib self30, p50_30


def run_file(path, cfg, sens):
    d = np.load(path)
    ear, _ = ear_mar_from_pts(d["pts"])
    eng = RuleEngine(cfg, sens)
    ticks = []
    for t, e in zip(d["t"].astype(float), ear):
        o = eng.update(float(t), None if not np.isfinite(e) else float(e))
        if o:
            ticks.append(o)
    return ticks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=["32"])
    ap.add_argument("--sens", default=None, choices=[None, "high", "medium", "low"])
    args = ap.parse_args()
    cfg = load_config()

    rec = {}
    for f in sorted((ROOT / "landmarks").glob("*.npz")):
        if f.name.startswith("test_"):
            continue
        subj, lab = f.stem.split("_", 1)
        if subj in args.exclude or lab not in ("0", "10"):
            continue
        rec.setdefault(subj, {})[lab] = run_file(f, cfg, args.sens)

    rows, aucs = [], []
    print(f"{'subjek':<7}{'AUC':>7}{'ref':>7} | {'ALERT: kuning+':>15}{'merah':>8}{'peringatan':>11} | "
          f"{'DROWSY: kuning+':>16}{'merah':>8}{'peringatan':>11}{'peringatan pertama (dtk)':>26} | {'uncertain a/d':>14}")
    for s, v in sorted(rec.items()):
        if "0" not in v or "10" not in v:
            continue
        def valid(tks):
            return [tk for tk in tks if tk["status"] != "uncertain"]
        a, d = valid(v["0"]), valid(v["10"])
        if not a or not d:
            print(f"{s:<7} (tidak ada detik valid)"); continue
        y = np.r_[np.zeros(len(a)), np.ones(len(d))]
        sc = np.r_[[tk["p50_30"] for tk in a], [tk["p50_30"] for tk in d]]
        A = auc(y, sc)
        frac = lambda tks, st: float(np.mean([tk["status"] in st for tk in tks]))
        al_a = sum(tk["alert"] for tk in v["0"]); al_d = [tk["t"] for tk in v["10"] if tk["alert"]]
        first = (al_d[0] - v["10"][0]["t"] + 0.0) if al_d else float("nan")
        unc = lambda tks: float(np.mean([tk["status"] == "uncertain" for tk in tks]))
        aucs.append(A)
        rows.append((s, A, frac(a, ("yellow", "red")), frac(a, ("red",)), frac(d, ("yellow", "red")), frac(d, ("red",)), al_a, len(al_d)))
        print(f"{s:<7}{A:7.3f}{REF.get(s, float('nan')):7.3f} | {frac(a, ('yellow', 'red')):15.1%}{frac(a, ('red',)):8.1%}{al_a:11d} | "
              f"{frac(d, ('yellow', 'red')):16.1%}{frac(d, ('red',)):8.1%}{len(al_d):11d}{first:26.0f} | {unc(v['0']):6.0%}/{unc(v['10']):.0%}")

    if not rows:
        sys.exit("Tidak ada subjek lengkap di landmarks/. Jalankan run_all.py dulu.")
    R = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
    print(f"\nRata-rata atas {len(rows)} subjek (bobot sama):")
    print(f"  AUC p50_30                     : {R[:, 0].mean():.3f}   (referensi baseline_loso self30: 0.787)")
    print(f"  Subjek dengan AUC >= 0.8       : {int((R[:, 0] >= 0.8).sum())} dari {len(rows)}   (referensi: 8 dari 11)")
    print(f"  Rekaman ALERT  - kuning atau lebih (alarm palsu): {R[:, 1].mean():.1%}   merah: {R[:, 2].mean():.1%}")
    print(f"  Rekaman DROWSY - kuning atau lebih (terdeteksi) : {R[:, 3].mean():.1%}   merah: {R[:, 4].mean():.1%}")
    print("\nCatatan: ambang kuning (IKM 33) punya dasar data, tetapi batas merah (IKM 66) dan pembagi skor 0,15 masih tebakan sementara.")
    print("Jika alarm palsu pada rekaman ALERT tinggi, itu bahan untuk menyesuaikan ambang (lewat uji coba relawan, bukan lewat menyetel ke data ini).")


if __name__ == "__main__":
    main()
