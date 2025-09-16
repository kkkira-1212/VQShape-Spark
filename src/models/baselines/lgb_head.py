# LightGBM baseline for AD/AP (precursor).
# Uses ratio + suppression labeling and shifts to precursor on demand.
# Includes stratified splits, standardization (fit on train), early stopping, and F1-oriented threshold sweep.

import os, argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
import lightgbm as lgb


WINDOW, PATCH, STRIDE = 5, 30, 5      
RATIO, GAP, EPS = 0.005, 5, 1e-8        
SEED = 42

def load_data(path, drop_col="Time"):
    df = pd.read_csv(path)
    if drop_col in df.columns:
        df = df.drop(columns=[drop_col])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(" µA", "", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(axis=0, how="any").reset_index(drop=True)

def build_patches_and_labels(signal):
    Xp, yp, starts = [], [], []
    step_max = len(signal) - (PATCH + WINDOW) + 1
    if step_max <= 0:
        return Xp, yp, starts
    for start in range(0, step_max, STRIDE):
        seg = signal[start:start+WINDOW+PATCH]   # 长度=35 = window+patch
        region = seg[WINDOW:]                    # [30] 作为特征
        last_fire = -10**9                       # patch 内抑制
        spark_flags = []
        for t in range(WINDOW, len(seg)):
            mu = np.mean(seg[t-WINDOW:t])
            val = seg[t]
            is_spark = abs(val - mu) / (abs(mu) + EPS) > RATIO
            rel_t = t - WINDOW
            if is_spark and (rel_t - last_fire) >= GAP:
                spark_flags.append(1); last_fire = rel_t
            else:
                spark_flags.append(0)
        yp.append(int(any(spark_flags[-PATCH:])))
        Xp.append(region.copy())
        starts.append(start)
    return Xp, yp, starts

def to_precursor_labels(y, drop_last=True):
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return y, 0
    y_next = np.roll(y, -1)
    y_next[-1] = 0
    if drop_last:
        y_next = y_next[:-1]
    return y_next, len(y_next)

def find_best_thr(y, score, start=0.05, end=0.95, step=0.01):
    best_thr, best_f1, best_triplet = 0.5, -1.0, (0.0,0.0,0.0)
    thr = start
    while thr <= end + 1e-12:
        pred = (score >= thr).astype(int)
        p,r,f1,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_thr = float(thr); best_triplet = (float(p),float(r),float(f1))
        thr += step
    return best_thr, best_triplet

def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    df = load_data(args.data_csv)

    X_all, y_all = [], []
    for col in df.columns:
        sig = df[col].values.astype(float)
        Xp, yp, _ = build_patches_and_labels(sig)
        if len(Xp) == 0: 
            continue
        X_all.append(np.stack(Xp))            # [num_patches, 30]
        y_all.append(np.array(yp, dtype=int)) # [num_patches]
    if len(X_all) == 0:
        print("No samples. Check data length and slicing params."); return

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    if args.label_mode == "precursor":
        y, keep_len = to_precursor_labels(y, drop_last=args.drop_last_for_precursor)
        if args.drop_last_for_precursor:
            X = X[:keep_len]

    has_pos = y.sum() > 0 and (len(y) - y.sum()) > 0
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y if has_pos else None)
    has_pos_tmp = y_tmp.sum() > 0 and (len(y_tmp) - y_tmp.sum()) > 0
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp if has_pos_tmp else None)

    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - mean) / std
    X_va = (X_va - mean) / std
    X_te = (X_te - mean) / std

    clf = LGBMClassifier(
        n_estimators=2000, learning_rate=0.02,
        num_leaves=31, min_child_samples=200,
        subsample=0.8, colsample_bytree=0.5,
        lambda_l1=1.0, lambda_l2=1.0,
        random_state=SEED
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )

    # Validation set threshold sweep (maximize F1), test set evaluation
    s_val = clf.predict_proba(X_va)[:, 1]
    best_thr, (p_v, r_v, f_v) = find_best_thr(y_va, s_val)

    s_te = clf.predict_proba(X_te)[:, 1]
    yhat = (s_te >= best_thr).astype(int)

    p,r,f1,_ = precision_recall_fscore_support(y_te, yhat, average="binary", zero_division=0)
    roc_auc  = roc_auc_score(y_te, s_te)
    pr_auc   = average_precision_score(y_te, s_te)
    rep      = classification_report(y_te, yhat, digits=4, zero_division=0)

    pos_rate = y.mean()
    tag = f"[baseline=RawPatch+LGBM {args.label_mode} ratio={RATIO} gap={GAP}]"
    summary = (f"{tag} sparks={int(y.sum())}/{len(y)} pos_rate={pos_rate:.4f} "
               f"thr={best_thr:.2f} P={p:.3f} R={r:.3f} F1={f1:.3f} "
               f"| AUC-ROC={roc_auc:.3f} AUC-PR={pr_auc:.3f}")
    print(summary); print(rep)

    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/result_lgb_{args.label_mode}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(summary + "\n" + rep + "\n")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
