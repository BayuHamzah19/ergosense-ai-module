# ERGO-SENSE: AI module

EAR-based eye-fatigue rule engine (Python and JavaScript) and evaluation scripts.

- `ergo_rule_v0.1/`: rule engine (config, JS, Python, README for integration)
- `src/`: landmark extraction, dataset building, LOSO baseline, ST-GCN comparison, parity test
- `results/`: metrics (CSV and text)

Data (UTA-RLDD videos, extracted landmarks) is NOT included. Download the dataset from its authors and the MediaPipe Face Landmarker model (`face_landmarker.task`) from the MediaPipe site.

## Catatan tentang ST-GCN

Folder `src/` memuat skrip dan hasil eksperimen ST-GCN yang dibuat sebagai pembanding. Pada evaluasi LOSO 11 subjek, ST-GCN (AUC 0,685) tidak lebih baik daripada rule engine EAR (AUC 0,787), sehingga ST-GCN tidak termasuk paket yang diserahkan. Paket yang dipakai tim lain hanyalah `ergo_rule_v0.1/`. Bobot model ST-GCN tidak disimpan di repositori ini.

Skrip perbandingan dan uji gabungan ada di branch `eksperimen-gabungan`.

## Paket yang diserahkan

Gunakan folder `ergo_rule_v0.1/ergo_rule/` (panduan integrasi ada di README di dalamnya). Versi yang dipakai: tag `v0.1.1-ear-rule`.
Parameter yang bertanda `provisional` di `ergo_rule_config.json` (batas merah, pembagi skor, aturan peringatan) belum tervalidasi.
