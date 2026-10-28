import glob
import pandas as pd

st = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/stgcn_loso_eyes_*.csv"))])
st["test_subject"] = st["test_subject"].astype(str)
cols = ["test_subject", "AUC", "bal_acc", "recall_drowsy", "precision_drowsy"]
print("ST-GCN per subjek (11 fold):")
print(st[cols].to_string(index=False))
print("\nRata-rata ST-GCN, semua subjek:")
print(st[cols[1:]].mean().round(3))

ear = pd.read_csv("results/baseline_loso.csv")
ear["test_subject"] = ear["test_subject"].astype(str)
common = sorted(set(ear["test_subject"]) & set(st["test_subject"]))
print("\nSubjek yang sama dengan EAR:", common)
print("ST-GCN:")
print(st[st.test_subject.isin(common)][cols[1:]].mean().round(3))
print("EAR, per aturan:")
print(ear[ear.test_subject.isin(common)].groupby("rule")[["AUC", "bal_acc", "recall_drowsy"]].mean().round(3))
print("\nPer subjek: ST-GCN vs EAR terbaik (bal_acc)")
e = ear.groupby("test_subject")["bal_acc"].max()
t = st.set_index("test_subject")["bal_acc"]
print(pd.DataFrame({"stgcn": t, "ear_terbaik": e}).loc[common].round(3))
