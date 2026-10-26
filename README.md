# ERGO-SENSE: AI module

EAR-based eye-fatigue rule engine (Python and JavaScript) and evaluation scripts.

- `ergo_rule_v0.1/`: rule engine (config, JS, Python, README for integration)
- `src/`: landmark extraction, dataset building, LOSO baseline, ST-GCN comparison, parity test
- `results/`: metrics (CSV and text)

Data (UTA-RLDD videos, extracted landmarks) is NOT included. Download the dataset from its authors and the MediaPipe Face Landmarker model (`face_landmarker.task`) from the MediaPipe site.
