"""
train_stgcn.py - ST-GCN kecil (20 node) dengan evaluasi Leave-One-Subject-Out.

Jalankan dari folder ERGO-SENSE (setelah make_dataset.py):
    python src/train_stgcn.py --epochs 3 --only-test 31     # uji cepat dulu (beberapa menit)
    python src/train_stgcn.py                               # semua fold, hingga 50 epoch
    python src/train_stgcn.py --data data/stgcn_windows_w5_cal.npz --tag cal
    python src/train_stgcn.py --seed 1                      # ulangi dengan seed lain

Protokol (mengikuti Bab 3.4 paper, disesuaikan dengan jumlah subjek):
  - Tiap fold: 1 subjek UJI ditahan. Dari 4 subjek sisanya, 1 subjek jadi VALIDASI (untuk early
    stopping, F1 kelas drowsy, sabar 8 epoch), 3 subjek untuk LATIH. Subjek tidak pernah muncul
    di dua bagian. (--val none: tanpa validasi, latih penuh dengan jumlah epoch tetap.)
  - Model: 3 blok [graph conv spasial -> conv temporal kernel 9], kanal 32/64/64, global average
    pooling, 2 keluaran. Cross-entropy berbobot kelas, Adam lr 1e-3, batch 64.
  - Ambang keputusan tetap 0.5 (TIDAK disetel pada subjek uji). AUC dilaporkan terpisah.
Keluaran: results/stgcn_loso_<tag>.csv dan results/stgcn_preds_<tag>.csv
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from baseline_loso import auc, bal_acc

ROOT = Path(__file__).resolve().parent.parent
V = 20


def build_adj(v=V):
    A = np.eye(v)
    cycles = [(0, 6), (6, 6), (12, 8)] if v == 20 else [(0, 6), (6, 6)]   # mata kanan, mata kiri, (bibir dalam)
    for base, n in cycles:
        for i in range(n):
            a, b = base + i, base + (i + 1) % n
            A[a, b] = A[b, a] = 1
    d = A.sum(1)
    return torch.tensor(A / np.sqrt(np.outer(d, d)), dtype=torch.float32)


class STBlock(nn.Module):
    def __init__(self, cin, cout, A, k=9):
        super().__init__()
        self.register_buffer("A", A)
        self.sp = nn.Conv2d(cin, cout, 1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.tc = nn.Conv2d(cout, cout, (k, 1), padding=(k // 2, 0))
        self.bn2 = nn.BatchNorm2d(cout)
        self.res = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))
        self.act = nn.ReLU()

    def forward(self, x):                                # x: (N, C, T, V)
        y = torch.einsum("nctv,vw->nctw", self.sp(x), self.A)
        y = self.act(self.bn1(y))
        y = self.bn2(self.tc(y))
        return self.act(y + self.res(x))


class STGCN(nn.Module):
    def __init__(self, cin=4, chans=(32, 64, 64), classes=2, v=V):
        super().__init__()
        A = build_adj(v)
        self.data_bn = nn.BatchNorm1d(cin * v)
        dims = (cin,) + tuple(chans)
        self.blocks = nn.Sequential(*[STBlock(dims[i], dims[i + 1], A) for i in range(len(chans))])
        self.fc = nn.Linear(chans[-1], classes)

    def forward(self, x):                                # x: (N, C, T, V)
        N, C, T, Vn = x.shape
        x = self.data_bn(x.permute(0, 3, 1, 2).reshape(N, Vn * C, T))
        x = x.reshape(N, Vn, C, T).permute(0, 2, 3, 1)
        return self.fc(self.blocks(x).mean(dim=(2, 3)))


def predict(model, X, dev, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(X[i:i + bs].to(dev)), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def f1_pos(y, pred):
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def fit(Xtr, ytr, Xva, yva, args, dev):
    model = STGCN(v=Xtr.shape[-1]).to(dev)
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2)), dtype=torch.float32).to(dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx].to(dev)), ytr_t[idx].to(dev))
            loss.backward(); opt.step(); tot += loss.item() * len(idx)
        msg = f"    epoch {ep:2d} loss {tot / len(perm):.4f}"
        if Xva is not None:
            pv = predict(model, torch.tensor(Xva), dev)
            f1, va_auc = f1_pos(yva, (pv >= 0.5).astype(int)), auc(yva, pv)
            msg += f" | val F1 {f1:.3f} AUC {va_auc:.3f}"
            score = f1 if args.monitor == "f1" else va_auc
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        print(msg + f" | {time.time() - t0:.1f} dtk", flush=True)
        if Xva is not None and bad >= args.patience:
            print("    early stopping")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stgcn_windows_w5.npz")
    ap.add_argument("--tag", default="w5")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay Adam")
    ap.add_argument("--monitor", choices=["f1", "auc"], default="f1",
                    help="metrik validasi untuk early stopping (paper: f1; auc tidak bergantung ambang 0.5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", choices=["subject", "none"], default="subject")
    ap.add_argument("--nodes", choices=["all", "eyes"], default="all",
                    help="all = 20 node (mata+bibir); eyes = 12 node (hanya mata). Ablation yang sudah direncanakan di paper (Bab 3.7)")
    ap.add_argument("--only-test", nargs="*", default=None, help="hanya fold dengan subjek uji ini")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    D = np.load(path)
    X, y, subj = D["X"], D["y"].astype(int), D["subject"].astype(str)
    subjects = sorted(set(subj.tolist()))
    if args.nodes == "eyes":
        X = np.ascontiguousarray(X[..., :12])
    print(f"Perangkat: {dev} | data {X.shape} | node: {args.nodes} | subjek {subjects}")
    n_params = sum(p.numel() for p in STGCN(v=X.shape[-1]).parameters())
    print(f"Parameter: {n_params:,} (~{n_params * 4 / 1e6:.2f} MB float32)\n")

    rows, preds = [], []
    for ti, ts in enumerate(subjects):
        if args.only_test and ts not in args.only_test:
            continue
        others = [s for s in subjects if s != ts]
        if args.val == "subject":
            vs = others[ti % len(others)]
            tr = [s for s in others if s != vs]
        else:
            vs, tr = None, others
        m_tr, m_te = np.isin(subj, tr), subj == ts
        Xva = yva = None
        if vs is not None:
            m_va = subj == vs
            Xva, yva = X[m_va], y[m_va]
        print(f"Fold uji={ts} | latih={tr} | validasi={vs}")
        model = fit(X[m_tr], y[m_tr], Xva, yva, args, dev)
        p = predict(model, torch.tensor(X[m_te]), dev)
        yt, pred = y[m_te], (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
        rows.append({
            "test_subject": ts, "val_subject": vs, "AUC": round(auc(yt, p), 3),
            "bal_acc": round(bal_acc(yt, pred), 3),
            "recall_drowsy": round(tp / max((yt == 1).sum(), 1), 3),
            "precision_drowsy": round(tp / (tp + fp), 3) if (tp + fp) else np.nan,
            "F1_drowsy": round(f1_pos(yt, pred), 3),
        })
        print(f"  -> {rows[-1]}\n")
        preds.append(pd.DataFrame({"subject": ts, "video": D["video"][m_te], "t_end": D["t_end"][m_te],
                                   "y": yt, "p_drowsy": p}))

    if not rows:
        sys.exit(f"Tidak ada fold yang dijalankan. --only-test {args.only_test} tidak ada di data. Subjek tersedia: {subjects}")
    R = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    R.to_csv(out / f"stgcn_loso_{args.tag}.csv", index=False)
    pd.concat(preds).to_csv(out / f"stgcn_preds_{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print("Hasil ST-GCN LOSO:")
    print(R.to_string(index=False))
    num = ["AUC", "bal_acc", "recall_drowsy", "precision_drowsy", "F1_drowsy"]
    print("\nRata-rata (bobot sama per subjek):\n", R[num].mean().round(3).to_string())
    if len(R) > 1:
        print("\nRata-rata tanpa subjek 31:\n", R[R.test_subject != "31"][num].mean().round(3).to_string())
    print(f"\nDisimpan: {out / f'stgcn_loso_{args.tag}.csv'}")


if __name__ == "__main__":
    main()