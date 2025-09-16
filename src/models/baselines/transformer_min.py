# transformer.py  —  minimal Transformer baseline with ROC-AUC / PR-AUC
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score 
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


WINDOW, PATCH, STRIDE = 5, 30, 5
RATIO, GAP, EPS = 0.005, 5, 1e-8
DATA = "/home/kitsch/VQShape/GEM1h.csv"
SEED = 42
BATCH = 128
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path):
    df = pd.read_csv(path)
    if "Time" in df.columns: df = df.drop(columns=["Time"])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(" µA","",regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)

def build_one_sensor(signal):
    Xp, yp = [], []
    step_max = len(signal) - (PATCH + WINDOW) + 1
    if step_max <= 0: return Xp, yp
    for start in range(0, step_max, STRIDE):
        seg = signal[start:start + WINDOW + PATCH]
        region = seg[WINDOW:]
        last_fire = -10**9; sparks=[]
        for t in range(WINDOW, len(seg)):
            mu = np.mean(seg[t-WINDOW:t]); val = seg[t]
            is_spark = abs(val - mu) / (abs(mu) + EPS) > RATIO
            rel_t = t - WINDOW
            if is_spark and (rel_t - last_fire) >= GAP:
                sparks.append(1); last_fire = rel_t
            else:
                sparks.append(0)
        yp.append(int(any(sparks[-PATCH:])))
        Xp.append(region.astype(np.float32))
    return Xp, yp

def prepare_dataset(df):
    X_all, y_all = [], []
    for col in df.columns:
        X, y = build_one_sensor(df[col].values.astype(float))
        if len(y) == 0: continue
        X_all.append(np.stack(X)); y_all.append(np.array(y, int))
    X = np.concatenate(X_all, 0)  # [N,30]
    y = np.concatenate(y_all, 0)  # [N]
    return X, y

def find_best_thr(y, s, lo=0.05, hi=0.95, step=0.01):
    best = (0.5, (0,0,0))
    thr = lo
    while thr <= hi + 1e-12:
        yhat = (s >= thr).astype(int)
        p,r,f1,_ = precision_recall_fscore_support(y,yhat,average="binary",zero_division=0)
        if f1 > best[1][2]: best = (float(thr),(float(p),float(r),float(f1)))
        thr += step
    return best

class TinyTransformerMinimal(nn.Module):
    def __init__(self, d_model=64, nhead=2, nlayers=1):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, 1)  # 只用均值池化，简洁稳定
    def forward(self, x):                  # x: [B,L,1]
        h = self.proj(x)                   # [B,L,D]
        h = self.encoder(h)                # [B,L,D]
        rep = h.mean(dim=1)                # Avg Pooling over time
        return self.head(rep).squeeze(-1)  # [B]

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = load_data(DATA)
    X, y = prepare_dataset(df)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    mean = X_tr.mean(axis=0, keepdims=True); std = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr-mean)/std; X_va = (X_va-mean)/std; X_te = (X_te-mean)/std

    X_tr = torch.tensor(X_tr[:, :, None], dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.float32)
    X_va = torch.tensor(X_va[:, :, None], dtype=torch.float32)
    y_va = torch.tensor(y_va, dtype=torch.float32)
    X_te = torch.tensor(X_te[:, :, None], dtype=torch.float32)
    y_te = torch.tensor(y_te, dtype=torch.float32)

    dl_tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)

    model = TinyTransformerMinimal().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss() 

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); logit = model(xb)
            loss = crit(logit, yb); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        s_val = torch.sigmoid(model(X_va.to(DEVICE))).cpu().numpy()
    best_thr, (P_best, R_best, F1_best) = find_best_thr(y_va.numpy(), s_val)

    with torch.no_grad():
        s_te = torch.sigmoid(model(X_te.to(DEVICE))).cpu().numpy()
    roc_auc = roc_auc_score(y_te.numpy(), s_te)                 # <-- NEW
    pr_auc  = average_precision_score(y_te.numpy(), s_te)       # <-- NEW

    yhat = (s_te >= best_thr).astype(int)
    p,r,f1,_ = precision_recall_fscore_support(y_te.numpy(), yhat, average="binary", zero_division=0)
    rep = classification_report(y_te.numpy(), yhat, digits=4, zero_division=0)

    pos_rate = y.mean()
    summary = (f"[baseline=Transformer-min ratio={RATIO} gap={GAP}] "
               f"sparks={int(y.sum())}/{len(y)} pos_rate={pos_rate:.4f} "
               f"thr={best_thr:.2f} P={p:.3f} R={r:.3f} F1={f1:.3f} "
               f"| AUC-ROC={roc_auc:.3f} AUC-PR={pr_auc:.3f}")          # <-- NEW
    print(summary); print(rep)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/result_transformer_minimal.txt","w") as f:
        f.write(summary + "\n" + rep)

if __name__ == "__main__":
    main()
