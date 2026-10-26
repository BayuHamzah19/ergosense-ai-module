"""Fitur per jendela waktu, pelatihan, evaluasi per subjek, dan ekspor model.

Input : manifest.csv (video,subject,label) + features/*.csv dari extract_features.py
Output: results/metrics.txt, results/confusion_*.png, models/ergo_rf.joblib,
        models/ergo_mlp.keras (jika TensorFlow terpasang), models/model_spec.json

Contoh:
  python src/train_model.py --manifest data/manifest.csv --features features
  python src/train_model.py --manifest data/manifest.csv --features features --ablation
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MAR_YAWN = 0.6     # ambang mulut terbuka lebar; sesuaikan dengan data Anda
YAWN_MIN_S = 1.5   # durasi minimum menguap (detik)
BLINK_MIN_S, BLINK_MAX_S = 0.05, 1.0

FEATURES = [
    "ear_mean", "ear_std", "ear_min_rel", "perclos80", "perclos50",
    "blink_rate", "blink_dur_mean", "closure_max", "n_long_closure",
    "mar_mean", "mar_max", "yawn_count",
    "pitch_std", "yaw_std", "roll_std", "pitch_dev",
]
GROUPS = {  # untuk ablation
    "EAR saja": ["ear_mean", "ear_std", "ear_min_rel"],
    "EAR+PERCLOS+kedip": ["ear_mean", "ear_std", "ear_min_rel", "perclos80", "perclos50",
                          "blink_rate", "blink_dur_mean", "closure_max", "n_long_closure"],
    "+MAR": ["ear_mean", "ear_std", "ear_min_rel", "perclos80", "perclos50",
             "blink_rate", "blink_dur_mean", "closure_max", "n_long_closure",
             "mar_mean", "mar_max", "yawn_count"],
    "semua (+pose)": FEATURES,
}


def runs(mask, t, dt):
    """Durasi (detik) tiap rangkaian True berurutan."""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append(t[i - 1] - t[s] + dt)
            s = None
    if s is not None:
        out.append(t[-1] - t[s] + dt)
    return out


def window_features(w, base, dt, pitch_med):
    t, ear, mar = w[:, 0], w[:, 1], w[:, 2]
    pit, yaw, rol = w[:, 3], w[:, 4], w[:, 5]
    rel = ear / base
    dur = t[-1] - t[0] + dt
    closed = rel < 0.5
    cl = runs(closed, t, dt)
    blinks = [c for c in cl if BLINK_MIN_S <= c <= BLINK_MAX_S]
    yawns = [c for c in runs(mar > MAR_YAWN, t, dt) if c >= YAWN_MIN_S]
    ok = ~np.isnan(ear)
    return {
        "ear_mean": np.nanmean(ear), "ear_std": np.nanstd(ear),
        "ear_min_rel": np.nanmin(rel),
        "perclos80": np.mean(rel[ok] < 0.2), "perclos50": np.mean(rel[ok] < 0.5),
        "blink_rate": len(blinks) / dur * 60,
        "blink_dur_mean": float(np.mean(blinks)) if blinks else 0.0,
        "closure_max": max(cl) if cl else 0.0,
        "n_long_closure": sum(c > BLINK_MAX_S for c in cl),
        "mar_mean": np.nanmean(mar), "mar_max": np.nanmax(mar),
        "yawn_count": len(yawns),
        "pitch_std": np.nanstd(pit), "yaw_std": np.nanstd(yaw), "roll_std": np.nanstd(rol),
        "pitch_dev": abs(np.nanmedian(pit) - pitch_med),
    }


def build_windows(manifest, feat_dir, win_s, step_s, min_face):
    rows = []
    for _, it in manifest.iterrows():
        name = f"{it['subject']}__{os.path.splitext(os.path.basename(it['video']))[0]}.csv"
        p = os.path.join(feat_dir, name)
        if not os.path.exists(p):
            print("lewati (fitur tidak ada):", name)
            continue
        d = pd.read_csv(p).to_numpy(dtype=float)
        if len(d) < 10 or np.isnan(d[:, 1]).all():
            continue
        t = d[:, 0]
        dt = float(np.median(np.diff(t)))
        base = np.nanpercentile(d[:, 1], 90)  # EAR mata terbuka per video
        pitch_med = np.nanmedian(d[:, 3])
        t0 = t[0]
        while t0 + win_s <= t[-1] + dt:
            w = d[(t >= t0) & (t < t0 + win_s)]
            if len(w) > 0 and np.mean(~np.isnan(w[:, 1])) >= min_face:
                f = window_features(w, base, dt, pitch_med)
                f.update(video=it["video"], subject=str(it["subject"]),
                         label=int(it["label"]), t0=t0)
                rows.append(f)
            t0 += step_s
    return pd.DataFrame(rows)


# ---------- model ----------
def make_keras(n_in, n_out):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_in,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(n_out, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def class_weights(y):
    cls, cnt = np.unique(y, return_counts=True)
    return {int(c): len(y) / (len(cls) * n) for c, n in zip(cls, cnt)}


def fit_predict(name, Xtr, ytr, Xte, n_classes, cols):
    if name == "rule":  # ambang yang dipelajari pada PERCLOS
        idx = [cols.index("perclos80"), cols.index("perclos50")]
        m = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=0)
        return m.fit(Xtr[:, idx], ytr).predict(Xte[:, idx])
    if name == "rf":
        m = RandomForestClassifier(300, class_weight="balanced", random_state=0, n_jobs=-1)
        return m.fit(Xtr, ytr).predict(Xte)
    if name == "gboost":
        m = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
        return m.fit(Xtr, ytr).predict(Xte)
    if name == "mlp":
        sc = StandardScaler().fit(Xtr)
        m = make_keras(Xtr.shape[1], n_classes)
        m.fit(sc.transform(Xtr), ytr, epochs=60, batch_size=64, verbose=0,
              class_weight=class_weights(ytr))
        return m.predict(sc.transform(Xte), verbose=0).argmax(1)
    raise ValueError(name)


def cross_validate(name, X, y, groups, n_classes, cols, n_splits):
    pred = np.full(len(y), -1)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):  # per subjek
        pred[te] = fit_predict(name, X[tr], y[tr], X[te], n_classes, cols)
    return pred


def report(name, y, pred, class_names, out):
    labels = list(range(len(class_names)))
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    line = (f"{name:8s} acc={accuracy_score(y, pred):.3f} "
            f"F1-macro={f1_score(y, pred, average='macro'):.3f} "
            f"recall/kelas=" + ", ".join(f"{c}:{r:.2f}" for c, r in zip(class_names, rec)))
    cm = confusion_matrix(y, pred, labels=labels)
    out.append(line)
    out.append("confusion matrix (baris=asli, kolom=prediksi):\n" + str(cm) + "\n")
    print(line)
    return cm


def save_cm(cm, class_names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30)
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("prediksi")
    ax.set_ylabel("asli")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features", default="features")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--min-face", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--class-names", default="Fresh,Mild Fatigue,Severe Fatigue")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--no-mlp", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    man = pd.read_csv(a.manifest)
    df = build_windows(man, a.features, a.window, a.step, a.min_face)
    if df.empty:
        raise SystemExit("Tidak ada jendela terbentuk. Cek manifest dan folder fitur.")
    df.to_csv("results/windows.csv", index=False)
    X = df[FEATURES].fillna(0).to_numpy(float)
    y = df["label"].to_numpy(int)
    g = df["subject"].to_numpy()
    n_classes = int(y.max()) + 1
    names = a.class_names.split(",")[:n_classes]
    names += [f"kelas{i}" for i in range(len(names), n_classes)]
    folds = min(a.folds, len(set(g)))
    if folds < 2:
        raise SystemExit("Butuh minimal 2 subjek untuk pembagian per subjek.")

    out = [f"jendela={len(df)} subjek={len(set(g))} kelas={n_classes} "
           f"window={a.window}s step={a.step}s folds={folds}",
           "distribusi kelas: " + str(dict(zip(*np.unique(y, return_counts=True)))), ""]
    print(out[0])
    try:
        import tensorflow  # noqa: F401
        have_tf = not a.no_mlp
    except ImportError:
        have_tf = False
        print("TensorFlow tidak terpasang: MLP dilewati.")

    models = ["rule", "rf", "gboost"] + (["mlp"] if have_tf else [])
    for name in models:
        pred = cross_validate(name, X, y, g, n_classes, FEATURES, folds)
        cm = report(name, y, pred, names, out)
        save_cm(cm, names, f"results/confusion_{name}.png")

    if a.ablation:
        out.append("== ABLATION (Random Forest, per subjek) ==")
        for gname, cols in GROUPS.items():
            Xs = df[cols].fillna(0).to_numpy(float)
            pred = cross_validate("rf", Xs, y, g, n_classes, cols, folds)
            out.append(f"{gname:20s} acc={accuracy_score(y, pred):.3f} "
                       f"F1-macro={f1_score(y, pred, average='macro'):.3f}")
            print(out[-1])

    # ---- model akhir pada semua data ----
    rf = RandomForestClassifier(300, class_weight="balanced", random_state=0, n_jobs=-1).fit(X, y)
    joblib.dump(rf, "models/ergo_rf.joblib")
    sc = StandardScaler().fit(X)
    spec = {
        "features": FEATURES, "classes": names,
        "window_s": a.window, "step_s": a.step,
        "mar_yawn": MAR_YAWN, "yawn_min_s": YAWN_MIN_S,
        "blink_s": [BLINK_MIN_S, BLINK_MAX_S],
        "baseline": "persentil-90 EAR (per sesi/kalibrasi)",
        "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
    }
    if have_tf:
        m = make_keras(X.shape[1], n_classes)
        m.fit(sc.transform(X), y, epochs=60, batch_size=64, verbose=0, class_weight=class_weights(y))
        m.save("models/ergo_mlp.keras")
    json.dump(spec, open("models/model_spec.json", "w"), indent=2)
    open("results/metrics.txt", "w").write("\n".join(out))
    print("Selesai. Lihat results/metrics.txt dan models/")


if __name__ == "__main__":
    main()
