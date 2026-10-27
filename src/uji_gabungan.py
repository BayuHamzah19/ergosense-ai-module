import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

st = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/stgcn_preds_eyes_*.csv"))])
ew = pd.read_csv("results/windows.csv").rename(columns={"file": "video"})
st["subject"] = st["subject"].astype(int); ew["subject"] = ew["subject"].astype(int)
st = st.sort_values("t_end"); ew = ew.sort_values("t_end")

m = pd.merge_asof(st, ew, on="t_end", by=["subject", "video"],
                  tolerance=0.6, direction="nearest", suffixes=("", "_ear"))
m = m.dropna(subset=["p50_30"]).reset_index(drop=True)
print("Jendela ST-GCN:", len(st), "| berhasil dicocokkan:", len(m))
print("Subjek:", sorted(m.subject.unique()))
print("Label y sama dengan y_ear:", (m["y"] == m["y_ear"]).mean().round(4))
print()

sets = {
    "EAR p50_30": ["p50_30"],
    "EAR maxclose_30": ["maxclose_30"],
    "ST-GCN": ["p_drowsy"],
    "Gabungan p50_30 + ST-GCN": ["p50_30", "p_drowsy"],
}
rows = []
for s in sorted(m.subject.unique()):
    tr, te = m[m.subject != s], m[m.subject == s]
    for name, cols in sets.items():
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(tr[cols], tr["y"])
        p = clf.predict_proba(te[cols])[:, 1]
        rows.append(dict(subject=s, model=name,
                         AUC=roc_auc_score(te["y"], p),
                         bal_acc=balanced_accuracy_score(te["y"], p > 0.5)))
r = pd.DataFrame(rows)
print("Per subjek (bal_acc):")
print(r.pivot(index="subject", columns="model", values="bal_acc").round(3))
print("\nRata-rata (bobot sama per subjek):")
print(r.groupby("model")[["AUC", "bal_acc"]].mean().round(3))
