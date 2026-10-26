"""
make_slide_assets.py - Tabel dan grafik siap tempel untuk slide bagian AI ERGO-SENSE.

Taruh di folder src/ (bersama eval_rule.py, ergo_rule.py, baseline_loso.py), lalu dari folder ERGO-SENSE:
    python src/make_slide_assets.py

Membaca (tidak mengubah apa pun):
  landmarks/           landmark 11 subjek data awal (subjek 32 dikeluarkan)
  landmarks_holdout/   landmark 11 subjek uji murni (subjek 18 dikeluarkan)
  results/baseline_11.txt          keluaran 'baseline_loso.py --calib self30' (untuk grafik panjang konteks)
  results/colab/**/stgcn_loso_*.csv   hasil ST-GCN (4 konfigurasi x 3 seed)

Menulis ke results/slides/:
  fig1_panjang_konteks.png   fig2_auc_per_subjek.png   fig3_stgcn_vs_aturan.png   fig4_sensitivitas.png
  tabel_slide.md             tabel_per_subjek.csv      catatan_slide.md
Semua gambar hanya berisi angka (tanpa wajah). Semua angka dihitung dari berkas di atas, bukan diketik.
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from baseline_loso import auc
from ergo_rule import load_config
from eval_rule import run_file

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "slides"
C_AWAL, C_BARU, C_ABU, C_GELAP = "#4C78A8", "#F58518", "#8C8C8C", "#333333"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 17, "axes.labelsize": 14, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})
TAGS = [("A_all_none", "Mata + bibir\ntanpa kalibrasi"), ("B_all_self30", "Mata + bibir\ndengan kalibrasi"),
        ("C_eyes_none", "Hanya mata\ntanpa kalibrasi"), ("D_eyes_self30", "Hanya mata\ndengan kalibrasi")]
SENS = [("low", "Rendah"), ("medium", "Sedang"), ("high", "Tinggi")]


def fmt(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def collect(lm_dir, exclude, cfg, sens):
    """Jalankan mesin aturan pada tiap subjek; kembalikan DataFrame statistik per subjek."""
    rec = {}
    for f in sorted(Path(lm_dir).glob("*.npz")):
        subj, lab = f.stem.split("_", 1)
        if subj in exclude or lab not in ("0", "10"):
            continue
        rec.setdefault(subj, {})[lab] = run_file(f, cfg, sens)
    rows = []
    for s, v in rec.items():
        if "0" not in v or "10" not in v:
            continue
        a = [t for t in v["0"] if t["status"] != "uncertain"]
        d = [t for t in v["10"] if t["status"] != "uncertain"]
        if not a or not d:
            continue
        y = np.r_[np.zeros(len(a)), np.ones(len(d))]
        sc = np.r_[[t["p50_30"] for t in a], [t["p50_30"] for t in d]]
        fr = lambda tk, st: float(np.mean([t["status"] in st for t in tk]))
        rows.append({"subjek": int(s), "auc": auc(y, sc),
                     "alert_kuning": fr(a, ("yellow", "red")), "alert_merah": fr(a, ("red",)),
                     "drowsy_kuning": fr(d, ("yellow", "red")), "drowsy_merah": fr(d, ("red",))})
    return pd.DataFrame(rows).sort_values("subjek").reset_index(drop=True)


def parse_baseline(path):
    """Ambil tabel 'AUC per subjek uji' dari keluaran teks baseline_loso.py -> DataFrame (baris = fitur)."""
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("AUC per subjek uji"))
    header = lines[i + 1].split()[1:]
    rows = {}
    for ln in lines[i + 3:i + 3 + 6]:
        t = ln.split()
        rows[t[0]] = [float(x) for x in t[1:]]
    return pd.DataFrame(rows, index=header).T


def stgcn_table():
    out = {}
    for tag, _ in TAGS:
        means = []
        for s in ["", "_seed1", "_seed2"]:
            f = glob.glob(str(ROOT / "results" / "colab" / "**" / f"stgcn_loso_{tag}{s}.csv"), recursive=True)
            if len(f) == 1:
                means.append(pd.read_csv(f[0])["AUC"].mean())
        if means:
            out[tag] = means
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    dev_dir, ho_dir = ROOT / "landmarks", ROOT / "landmarks_holdout"
    if not dev_dir.exists():
        sys.exit("Folder landmarks/ tidak ada.")

    print("Menjalankan mesin aturan pada landmark (beberapa menit)...")
    stats = {}
    for sens, _ in SENS:
        stats[("awal", sens)] = collect(dev_dir, ["32"], cfg, sens)
        stats[("baru", sens)] = collect(ho_dir, ["18"], cfg, sens) if ho_dir.exists() else pd.DataFrame()
    dev, new = stats[("awal", "medium")], stats[("baru", "medium")]

    # ---------------- Fig 1: panjang konteks ----------------
    b_path = ROOT / "results" / "baseline_11.txt"
    base = None
    if b_path.exists():
        try:
            base = parse_baseline(b_path)
        except Exception as e:  # noqa: BLE001
            print("  (gagal membaca baseline_11.txt:", e, ")")
    if base is not None:
        vals = [base.loc[k, "rata2"] for k in ("p50_5", "p50_15", "p50_30")]
        fig, ax = plt.subplots(figsize=(8, 5.2))
        bars = ax.bar(["5 detik", "15 detik", "30 detik"], vals, color=C_AWAL, width=0.55)
        ax.axhline(0.5, color=C_ABU, ls="--", lw=1.5)
        ax.text(2.45, 0.505, "tebakan acak (0,5)", ha="right", va="bottom", color=C_ABU, fontsize=12)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, fmt(v), ha="center", fontsize=15, fontweight="bold")
        ax.set_ylim(0.45, 0.9)
        ax.set_ylabel("AUC rata-rata (11 subjek awal)")
        ax.set_xlabel("Berapa detik terakhir yang dilihat aturan")
        ax.set_title("Konteks lebih panjang, AUC lebih tinggi", loc="left")
        fig.text(0.01, 0.01, "Sumbu mulai dari 0,45 supaya selisih terlihat. Aturan EAR, kalibrasi 30 detik pertama.", fontsize=10, color=C_ABU)
        fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "fig1_panjang_konteks.png", dpi=200); plt.close(fig)
    else:
        print("  (fig1 dilewati: results/baseline_11.txt tidak ada. Buat dengan: python src/baseline_loso.py --calib self30 > results/baseline_11.txt)")

    # ---------------- Fig 2: AUC per subjek ----------------
    allsub = pd.concat([dev.assign(grup="awal"), new.assign(grup="baru")]) if len(new) else dev.assign(grup="awal")
    allsub = allsub.sort_values("auc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [C_AWAL if g == "awal" else C_BARU for g in allsub.grup]
    ax.barh(range(len(allsub)), allsub.auc, color=colors, height=0.72)
    ax.set_yticks(range(len(allsub)))
    ax.set_yticklabels([f"S{s}" for s in allsub.subjek], fontsize=11)
    ax.axvline(0.5, color=C_ABU, ls="--", lw=1.5); ax.axvline(0.8, color=C_GELAP, ls=":", lw=1.5)
    ax.set_ylim(-0.7, len(allsub) + 0.9)
    ax.text(0.5, len(allsub) + 0.05, "tebakan", ha="center", va="bottom", color=C_ABU, fontsize=11)
    ax.text(0.8, len(allsub) + 0.05, "0,8", ha="center", va="bottom", color=C_GELAP, fontsize=11)
    ax.set_xlim(0, 1.05); ax.set_xlabel("AUC aturan EAR (konteks 30 detik) per subjek")
    ax.set_title("Bekerja baik pada sebagian besar orang, tidak pada semuanya", loc="left")
    leg = [Patch(color=C_AWAL, label=f"Data awal (n={len(dev)}), rata-rata {fmt(dev.auc.mean())}")]
    if len(new):
        leg.append(Patch(color=C_BARU, label=f"Uji murni (n={len(new)}), rata-rata {fmt(new.auc.mean())}"))
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(leg), frameon=False, fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig2_auc_per_subjek.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---------------- Fig 3: ST-GCN vs aturan ----------------
    st = stgcn_table()
    if st:
        labels = [l for t, l in TAGS if t in st]
        m = [np.mean(st[t]) for t, _ in TAGS if t in st]
        sd = [np.std(st[t], ddof=1) if len(st[t]) > 1 else 0 for t, _ in TAGS if t in st]
        r5 = base.loc["p50_5", "rata2"] if base is not None else None
        r30 = base.loc["p50_30", "rata2"] if base is not None else None
        fig, ax = plt.subplots(figsize=(10, 5.6))
        bars = ax.bar(labels, m, yerr=sd, capsize=6, color=C_ABU, width=0.55, error_kw={"lw": 2})
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, 0.462, fmt(v), ha="center", fontsize=15, fontweight="bold", color="white")
        ax.axhline(0.5, color=C_GELAP, ls=":", lw=1.2)
        if r5 is not None:
            ax.axhline(r5, color=C_AWAL, ls="--", lw=2); ax.text(len(m) - 0.45, r5 + 0.008, f"aturan EAR, 5 detik: {fmt(r5)}", ha="right", color=C_AWAL, fontsize=12)
            ax.axhline(r30, color=C_BARU, ls="--", lw=2); ax.text(len(m) - 0.45, r30 + 0.008, f"aturan EAR, 30 detik: {fmt(r30)}", ha="right", color=C_BARU, fontsize=12)
        ax.set_ylim(0.45, 0.9); ax.set_ylabel("AUC rata-rata (11 subjek)")
        ax.set_title("ST-GCN tidak jelas lebih baik daripada aturan sederhana", loc="left")
        fig.text(0.01, 0.01, "Batang = rata-rata 3 percobaan acak, garis tegak = simpangan baku antar percobaan. Sumbu mulai dari 0,45.", fontsize=10, color=C_ABU)
        fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "fig3_stgcn_vs_aturan.png", dpi=200); plt.close(fig)
    else:
        print("  (fig3 dilewati: results/colab/**/stgcn_loso_*.csv tidak ditemukan)")

    # ---------------- Fig 4: sensitivitas ----------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    allx, ally = [], []
    offs = {"Rendah": (10, -18), "Sedang": (10, 8), "Tinggi": (10, -18)}
    for grup, warna, nama in (("awal", C_AWAL, "Data awal"), ("baru", C_BARU, "Uji murni")):
        pts = []
        for sens, lab in SENS:
            df = stats[(grup, sens)]
            if len(df):
                pts.append((df.alert_kuning.mean() * 100, df.drowsy_kuning.mean() * 100, lab))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=warna, lw=2.5, ms=9, label=nama)
            for x, y, lab in pts:
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=offs[lab], fontsize=11, color=warna)
            allx += [p[0] for p in pts]; ally += [p[1] for p in pts]
    if allx:
        ax.set_xlim(0, max(allx) * 1.25 + 4); ax.set_ylim(max(0, min(ally) - 12), min(100, max(ally) + 10))
    ax.set_xlabel("Alarm palsu: rekaman segar berstatus kuning+ (%)")
    ax.set_ylabel("Terdeteksi: rekaman ngantuk berstatus kuning+ (%)")
    ax.set_title("Sensitivitas menukar deteksi dengan alarm palsu", loc="left", pad=22)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig4_sensitivitas.png", dpi=200); plt.close(fig)

    # ---------------- Tabel dan catatan ----------------
    allsub.sort_values("subjek")[["subjek", "grup", "auc", "alert_kuning", "alert_merah", "drowsy_kuning", "drowsy_merah"]].round(3).to_csv(OUT / "tabel_per_subjek.csv", index=False)
    n8 = lambda df: int((df.auc >= 0.8).sum())
    md = ["# Tabel untuk slide (dihitung dari berkas hasil)\n"]
    if base is not None:
        md += ["## AUC aturan EAR menurut panjang konteks (11 subjek awal)\n", "| Konteks | AUC rata-rata |", "|---|---|"]
        md += [f"| {k} | {fmt(base.loc[kk, 'rata2'])} |" for k, kk in (("5 detik", "p50_5"), ("15 detik", "p50_15"), ("30 detik", "p50_30"))] + [""]
    md += ["## Data awal vs uji murni (sensitivitas Sedang, konteks 30 detik)\n", "| | Data awal | Uji murni |", "|---|---|---|"]
    if len(new):
        md += [f"| Jumlah subjek | {len(dev)} | {len(new)} |", f"| AUC rata-rata | {fmt(dev.auc.mean())} | {fmt(new.auc.mean())} |",
               f"| Subjek dengan AUC di atas 0,8 | {n8(dev)} | {n8(new)} |",
               f"| Rekaman segar: kuning atau lebih | {dev.alert_kuning.mean():.0%} | {new.alert_kuning.mean():.0%} |",
               f"| Rekaman segar: merah | {dev.alert_merah.mean():.0%} | {new.alert_merah.mean():.0%} |",
               f"| Rekaman ngantuk: kuning atau lebih | {dev.drowsy_kuning.mean():.0%} | {new.drowsy_kuning.mean():.0%} |",
               f"| Rekaman ngantuk: merah | {dev.drowsy_merah.mean():.0%} | {new.drowsy_merah.mean():.0%} |", ""]
    else:
        md += [f"| AUC rata-rata | {fmt(dev.auc.mean())} | (landmarks_holdout/ tidak ada) |", ""]
    if st:
        md += ["## ST-GCN (11 subjek, 3 percobaan acak)\n", "| Konfigurasi | AUC rata-rata | Simpangan baku antar percobaan |", "|---|---|---|"]
        md += [f"| {l.replace(chr(10), ' ')} | {fmt(np.mean(st[t]))} | {fmt(np.std(st[t], ddof=1) if len(st[t]) > 1 else 0)} |" for t, l in TAGS if t in st] + [""]
    (OUT / "tabel_slide.md").write_text("\n".join(md), encoding="utf-8")

    new_auc = fmt(new.auc.mean()) if len(new) else "-"
    new_n8 = n8(new) if len(new) else "-"
    note = f"""# Catatan untuk slide AI (usulan 3 slide, sekitar 2 menit dari 7 menit tim)

Semua angka di bawah dihitung dari berkas hasil. Tempel hanya yang Anda pahami asalnya.

## Slide 1: Alur dan keputusan
- Alur: webcam -> MediaPipe (titik mata) -> EAR -> seberapa lama mata tertutup dalam 30 detik terakhir -> skor 0 sampai 100 -> status hijau, kuning, merah.
- Kalibrasi 30 detik pertama per pengguna. Semua diproses di perangkat, tanpa menyimpan frame.
- Keputusan: aplikasi memakai aturan EAR, bukan ST-GCN. Alasannya di slide 3.

## Slide 2: Hasil aturan EAR (fig1 dan fig2)
- Diuji dengan Leave-One-Subject-Out pada data awal ({len(dev)} subjek) dan diuji murni pada {len(new)} subjek lain yang tidak dipakai merancang aturan.
- AUC rata-rata {fmt(dev.auc.mean())} (data awal) dan {new_auc} (uji murni). AUC 0,5 = tebakan, 1,0 = sempurna.
- Bekerja baik (AUC di atas 0,8) pada {n8(dev)} dari {len(dev)} subjek awal dan {new_n8} dari {len(new)} subjek uji murni.
- Jujur: tidak bekerja pada sebagian orang. Konteks 5 detik jauh lebih lemah daripada 30 detik.

## Slide 3: ST-GCN dan keterbatasan (fig3 dan fig4)
- Kami juga melatih ST-GCN (graf titik wajah). Hasilnya setara dengan aturan sederhana di konfigurasi terbaik dan sangat bervariasi, jadi dipilih yang sederhana dan ringan.
- Sensitivitas menukar deteksi dengan alarm palsu (fig4). Karena itu peringatan dibuat opsional.
- Keterbatasan: {len(dev) + len(new)} subjek, dua rekaman per subjek (segar dan ngantuk), label dari pengakuan peserta, bukan pekerja kantoran. AUC bukan akurasi aplikasi.

## Jangan dikatakan
- Akurasi tunggal tanpa konteks, atau 100% akurat.
- Mendeteksi stres, emosi, mikro-ekspresi, atau menguap.
- Target 30 sampai 60 FPS sebagai hasil, sebelum diukur.
"""
    (OUT / "catatan_slide.md").write_text(note, encoding="utf-8")
    print(f"\nSelesai. Hasil di {OUT}:")
    for f in sorted(OUT.iterdir()):
        print("  ", f.name)
    print("\nPeriksa fig2 dan fig4: jika grup uji murni kosong, landmarks_holdout/ belum ada.")


if __name__ == "__main__":
    main()
