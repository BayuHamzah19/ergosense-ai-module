import glob
import pandas as pd

ear = pd.read_csv("results/baseline_loso_self30.csv")
ear["test_subject"] = ear["test_subject"].astype(str)
st = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/stgcn_loso_eyes_*.csv"))])
st["test_subject"] = st["test_subject"].astype(str)

def pick(rule):
    d = ear[ear.rule == rule].set_index("test_subject")
    return d[["AUC", "bal_acc", "recall_drowsy"]].add_prefix(rule + "_")

t = pick("p50_30").join(pick("maxclose_30"))
s = st.set_index("test_subject")[["AUC", "bal_acc", "recall_drowsy"]].add_prefix("stgcn_")
t = t.join(s).sort_index()
t["menang_bal_acc"] = (t["stgcn_bal_acc"] > t["p50_30_bal_acc"]).map({True: "ST-GCN", False: "EAR"})

pd.set_option("display.width", 200)
print("Perbandingan per subjek (LOSO, kalibrasi self30, 11 subjek)")
print(t.round(3).to_string())
print("\nRata-rata (bobot sama per subjek):")
print(t.drop(columns="menang_bal_acc").mean().round(3).to_string())
print("\nJumlah subjek dimenangkan (bal_acc):")
print(t["menang_bal_acc"].value_counts().to_string())
t.round(3).to_csv("results/perbandingan_per_subjek.csv")
