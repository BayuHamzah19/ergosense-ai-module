"""
compare_ear.py - Bandingkan deret EAR dari browser dan dari Python untuk klip yang sama.

    python src/compare_ear.py ear_browser.csv ear_python.csv

Kedua CSV berkolom  t,ear  (ear kosong = wajah tidak terdeteksi). t boleh berawal dari angka berbeda
(mis. performance.now() di browser): tiap deret dinormalkan ke awalnya sendiri, lalu selisih waktu
(lag, paling jauh 3 detik) dicari otomatis.

Yang dinilai (nilai ambang adalah USULAN, belum teruji; ubah bila hasil nyata menunjukkan hal lain):
  1. rasio skala: median browser / median Python, harus 0.95 sampai 1.05.
  2. korelasi setelah dihaluskan 0.4 detik, harus >= 0.95. Korelasi mentah tetap dilaporkan, tetapi kedipan
     0.1-0.2 detik tersampel pada waktu berbeda di dua jalur, jadi korelasi mentah wajar lebih rendah.
  3. baseline (persentil ke-90 EAR) kedua jalur, selisih relatif < 5%.
  4. porsi waktu mata tertutup (EAR < 0.5 x baseline) kedua jalur, selisih < 0.02. Ini besaran yang
     dipakai aplikasi, jadi ini ukuran kecocokan yang terpenting.
Rasio skala jauh dari 1 (mis. 0.56) hampir pasti berarti koordinat belum dikonversi ke piksel (x*lebar, y*tinggi).
Catatan: skala yang seragam sebagian diserap kalibrasi (EAR dibagi baseline), tetapi angka tetap harus cocok supaya
nilai yang tampil dan pengujian konsisten.
"""
import sys

import numpy as np

GRID = 1 / 15
SMOOTH_S = 0.4


def read_csv(path):
    t, e = [], []
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a, _, b = line.partition(",")
            t.append(float(a)); e.append(float(b) if b.strip() != "" else np.nan)
    t, e = np.array(t), np.array(e)
    return t - t[0], e


def resample(t, e, grid):
    ok = np.isfinite(e)
    out = np.interp(grid, t[ok], e[ok], left=np.nan, right=np.nan)
    # grid yang jauh (> 0.5 dtk) dari sampel valid dianggap tidak ada wajah
    idx = np.searchsorted(t[ok], grid)
    tv = t[ok]
    lo, hi = tv[np.clip(idx - 1, 0, len(tv) - 1)], tv[np.clip(idx, 0, len(tv) - 1)]
    out[np.minimum(np.abs(grid - lo), np.abs(hi - grid)) > 0.5] = np.nan
    return out


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan, m
    return float(np.corrcoef(a[m], b[m])[0, 1]), m


def main():
    if len(sys.argv) < 3:
        sys.exit("pemakaian: python compare_ear.py ear_browser.csv ear_python.csv")
    tb, eb = read_csv(sys.argv[1])
    tp, ep = read_csv(sys.argv[2])
    dur = min(tb[-1], tp[-1])
    grid = np.arange(0, dur, GRID)
    best = (-2, 0)
    for lag in np.arange(-3, 3 + 1e-9, GRID):                       # geser deret browser sebesar lag
        c, _ = corr(resample(tb + lag, eb, grid), resample(tp, ep, grid))
        if np.isfinite(c) and c > best[0]:
            best = (c, lag)
    c, lag = best
    if c == -2:
        sys.exit("Tidak cukup data yang bisa dibandingkan (periksa isi CSV).")
    b, p = resample(tb + lag, eb, grid), resample(tp, ep, grid)
    _, m = corr(b, p)
    mae = float(np.mean(np.abs(b[m] - p[m])))
    ratio = float(np.nanmedian(b[m]) / np.nanmedian(p[m]))
    k = max(1, int(round(SMOOTH_S / GRID)))
    sm = lambda x: np.convolve(np.where(np.isfinite(x), x, np.nanmedian(x)), np.ones(k) / k, mode="same")
    c_s = float(np.corrcoef(sm(b)[m], sm(p)[m])[0, 1])
    base_b, base_p = float(np.nanpercentile(b[m], 90)), float(np.nanpercentile(p[m], 90))
    clo_b, clo_p = float(np.mean(b[m] < 0.5 * base_b)), float(np.mean(p[m] < 0.5 * base_p))
    agree = float(np.mean(np.isfinite(b) == np.isfinite(p)))
    d_base = abs(base_b - base_p) / base_p
    print(f"Titik dibandingkan          : {int(m.sum())} (grid {1/GRID:.0f} Hz)")
    print(f"Selisih waktu (lag)         : {lag:+.2f} detik")
    print(f"Rasio skala (browser/Python): {ratio:.3f}   (target 0.95 sampai 1.05)")
    print(f"Korelasi mentah             : {c:.3f}")
    print(f"Korelasi dihaluskan 0.4 dtk : {c_s:.3f}   (target >= 0.95)")
    print(f"Selisih absolut rata-rata   : {mae:.4f}")
    print(f"Baseline browser / Python   : {base_b:.3f} / {base_p:.3f}   (selisih relatif {d_base:.1%}, target < 5%)")
    print(f"Mata tertutup browser / Py  : {clo_b:.3f} / {clo_p:.3f}   (selisih {abs(clo_b - clo_p):.3f}, target < 0.02)")
    print(f"Kecocokan wajah terdeteksi  : {agree:.1%}")
    checks = [("rasio skala", 0.95 <= ratio <= 1.05), ("korelasi dihaluskan", c_s >= 0.95),
              ("baseline", d_base < 0.05), ("porsi mata tertutup", abs(clo_b - clo_p) < 0.02)]
    bad = [n for n, ok in checks if not ok]
    print("\nHASIL:", "LOLOS (sesuai target usulan)" if not bad else "PERLU DICEK: " + ", ".join(bad))
    if "rasio skala" in bad or "baseline" in bad:
        print(" - Skala berbeda: cek apakah koordinat sudah dikonversi ke piksel (x*lebar, y*tinggi) sebelum menghitung EAR.")
    if "korelasi dihaluskan" in bad:
        print(" - Bentuk sinyal berbeda: cek indeks landmark mata (33,160,158,133,153,144 dan 362,385,387,263,373,380) dan urutan p1..p6.")


if __name__ == "__main__":
    main()
