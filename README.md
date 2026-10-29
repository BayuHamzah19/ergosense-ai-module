# ERGO-SENSE: arsip eksperimen (bukan paket serah-terima)

Branch ini menyimpan bukti pembanding EAR vs ST-GCN. Paket yang diserahkan ke tim lain ada di branch `main` (tag `v0.1.1-ear-rule`). Jangan digabung ke `main`.

## Isi
- `src/rekap_stgcn.py`: rekap ST-GCN LOSO 11 fold
- `src/tabel_perbandingan.py`: tabel EAR (`p50_30`, `maxclose_30`) vs ST-GCN per subjek
- `src/uji_gabungan.py`: uji gabungan skor EAR + ST-GCN (regresi logistik, LOSO)
- `src/eval_rule.py`: evaluasi mesin pada landmark nyata
- `results/`: keluaran skrip di atas, salinan hasil kalibrasi `self30`

## Hasil utama (LOSO 11 subjek, kalibrasi self30)

| Metode | AUC | Balanced acc |
|---|---|---|
| EAR `p50_30` | 0,787 | 0,697 |
| ST-GCN (node mata) | 0,685 | 0,621 |
| Gabungan `p50_30` + ST-GCN | 0,771 | 0,631 |

Uji holdout (11 subjek, aturan dibekukan sebelum dilihat): AUC `p50_30` 0,784.

## Menjalankan ulang

Jalankan dari folder proyek:

    python src/rekap_stgcn.py
    python src/tabel_perbandingan.py
    python src/uji_gabungan.py
    python src/eval_rule.py

Jangan menjalankan ulang `src/baseline_loso.py` tanpa mengamankan `results/baseline_loso.csv` dan `results/windows.csv`, karena skrip itu menimpa keduanya.

Data (video UTA-RLDD dan landmark) tidak disertakan.
