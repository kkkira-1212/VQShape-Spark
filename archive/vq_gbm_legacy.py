# Legacy multi-grid runner.
import os, sys, types
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

import argparse
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args()



def save_labels_to_csv(
    sensor_name,
    y_labels,
    start_indices,
    timestamps=None,
    out_path="outputs/labeled_data.csv",
):
    ts = None
    if timestamps is not None:
        try:
            if isinstance(timestamps, torch.Tensor):
                ts = timestamps.detach().cpu().numpy()
            elif hasattr(timestamps, "iloc"):
                ts = timestamps
            else:
                ts = np.asarray(timestamps)
        except Exception:
            ts = None

    rows = []
    for idx, (lbl, s_idx) in enumerate(zip(y_labels, start_indices)):
        try:
            s_idx_int = int(s_idx)
        except Exception:
            try:
                s_idx_int = int(s_idx.item())
            except Exception:
                s_idx_int = None

        row = {
            "sensor": sensor_name,
            "patch_idx": idx,
            "start_index": s_idx_int if s_idx_int is not None else -1,
            "label": int(lbl),
        }
        if ts is not None and s_idx_int is not None and s_idx_int >= 0:
            try:
                if hasattr(ts, "iloc"):
                    row["timestamp"] = str(ts.iloc[s_idx_int])
                else:
                    row["timestamp"] = str(ts[s_idx_int])
            except Exception:
                pass
        rows.append(row)

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(out_path):
        df_out.to_csv(out_path, index=False)
    else:
        df_out.to_csv(out_path, mode="a", header=False, index=False)


def find_best_threshold(y_true, y_score, start=0.05, end=0.95, step=0.01):
    best_thr, best_f1, best_triplet = 0.5, -1.0, (0.0, 0.0, 0.0)
    thr = start
    while thr <= end + 1e-12:
        y_pred = (y_score >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_triplet = (float(p), float(r), float(f1))
        thr += step
    return best_thr, best_triplet

def shift_to_precursor_labels(y_labels, drop_last=True):

    y = np.asarray(y_labels, dtype=int)
    if len(y) == 0:
        return y_labels, len(y)

    y_next = np.roll(y, -1)
    y_next[-1] = 0

    if drop_last:
        y_next = y_next[:-1]
        keep_len = len(y_next)
    else:
        keep_len = len(y_next)

    return y_next.tolist(), keep_len


PATCH_SIZE   = 30
WINDOW_SIZE  = 5
STRIDE       = 5

MODES        = ["ratio"]
RATIO_LIST   = [0.005]
GAP_LIST     = [5]
Q_LIST       = []   

CKPT_PATH = "finetuned_small_spark.ckpt"
DATA_CSV  = "GEM1h.csv"
DROP_COL  = "Time"

USE_PROB  = False  
USE_TFIDF = False 

LGBM_KW = dict(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.5,
    lambda_l1=1.0,
    lambda_l2=1.0,
    class_weight=None,
    random_state=42,
)


def load_data(path_csv: str, drop_col: str = "Time") -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    if drop_col in df.columns:
        df = df.drop(columns=[drop_col])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(" µA", "", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    return df


def build_one_sensor_patches(signal: np.ndarray, mode: str,
                             threshold_ratio: float | None,
                             q: float | None,
                             suppression_gap: int,
                             window_size: int = WINDOW_SIZE,
                             patch_size: int = PATCH_SIZE,
                             stride: int = STRIDE,
                             seq_len: int = 512):

    if mode == "quantile":
        diffs = np.abs(np.diff(signal, prepend=signal[0]))
        thr_q = np.quantile(diffs, q)

    X_patches, y_labels, start_indices = [], [], []
    for start in range(0, len(signal) - (patch_size + window_size) + 1, stride):
        end = start + window_size + patch_size
        full_patch = signal[start:end]
        region = full_patch[window_size:] 

        last_fire = -10**9
        spark_flags = []
        for t in range(window_size, len(full_patch)):
            mu = np.mean(full_patch[t - window_size: t])
            val = full_patch[t]
            if mode == "ratio":
                is_spark = abs(val - mu) / (abs(mu) + 1e-8) > float(threshold_ratio)
            else:
                is_spark = abs(val - full_patch[t-1]) > thr_q
            rel_t = t - window_size
            if is_spark and (rel_t - last_fire) >= suppression_gap:
                spark_flags.append(1)
                last_fire = rel_t
            else:
                spark_flags.append(0)

        label = int(any(spark_flags[-patch_size:]))
        y_labels.append(label)
        start_indices.append(start)

        x = torch.tensor(region, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,P]
        x = F.interpolate(x, size=seq_len, mode="linear")                        # [1,1,SEQ_LEN]
        X_patches.append(x)

    spark_count = int(np.sum(y_labels))
    return X_patches, y_labels, start_indices, spark_count


def run_one(df: pd.DataFrame, vq_model, device: str,
            mode: str, threshold_ratio: float | None, q: float | None,
            suppression_gap: int, NP: int, PP: int, SEQ_LEN: int):

    X_all, y_all = [], []
    total_sparks = total_patches = 0

    for sensor in df.columns:
        signal = df[sensor].values.astype(float)

        X_patches, y_labels, starts, spark_cnt = build_one_sensor_patches(
            signal=signal,
            mode=mode,
            threshold_ratio=threshold_ratio,
            q=q,
            suppression_gap=suppression_gap,
            window_size=WINDOW_SIZE,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            seq_len=SEQ_LEN,
        )

        if args.label_mode == "precursor":
            shifted_labels, keep_len = shift_to_precursor_labels(
                y_labels, drop_last=args.drop_last_for_precursor
            )
            if args.drop_last_for_precursor:
                X_patches = X_patches[:keep_len]
                starts     = starts[:keep_len]
            y_labels = shifted_labels
            spark_cnt = int(np.sum(y_labels))
        
        if mode == "ratio" and abs(threshold_ratio - 0.005) < 1e-12:
            save_labels_to_csv(sensor, y_labels, starts, timestamps=None,
                            out_path=f"outputs/labeled_data_{args.label_mode}.csv")

        total_sparks += spark_cnt
        total_patches += len(y_labels)
        if len(X_patches) == 0:
            continue

        # VQ tokenization -> histogram
        with torch.no_grad():
            X_tensor = torch.cat(X_patches, dim=0).to(device)            # [B,1,T]
            B, C, T = X_tensor.shape
            assert T == SEQ_LEN == NP * PP, f"{T} != {NP}*{PP}"
            rep, _ = vq_model(X_tensor, mode="tokenize")
            hist = rep["histogram"].detach().cpu().numpy()              # [B, K]

        X_all.append(hist)
        y_all.append(np.array(y_labels, dtype=int))

    if len(X_all) == 0:
        return {"ok": False, "msg": "no patches produced", "mode": mode, "ratio": threshold_ratio, "q": q, "gap": suppression_gap}

    X = np.concatenate(X_all, axis=0)   # [N, K] 直方图
    y = np.concatenate(y_all, axis=0)   # [N]

    
    def to_prob(A):
        s = A.sum(axis=1, keepdims=True) + 1e-12
        return A / s

    # if USE_PROB:
    #     X = to_prob(X)

    def apply_prob_tfidf(X_tr, X_val, X_te):
        if not USE_TFIDF:
            return X_tr, X_val, X_te
        df_vec = (X_tr > 0).sum(axis=0)        
        N = X_tr.shape[0]
        idf = np.log((N + 1) / (df_vec + 1)) + 1.0  
        return X_tr * idf, X_val * idf, X_te * idf

    has_both = (y.sum() > 0) and ((len(y) - y.sum()) > 0)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if has_both else None)
    has_both_tmp = (y_tmp.sum() > 0) and ((len(y_tmp) - y_tmp.sum()) > 0)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp if has_both_tmp else None)


    if USE_PROB:
        X_tr = to_prob(X_tr); X_val = to_prob(X_val); X_te = to_prob(X_te)
    X_tr, X_val, X_te = apply_prob_tfidf(X_tr, X_val, X_te)

    clf = LGBMClassifier(**LGBM_KW)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
    ],
)

    y_val_score = clf.predict_proba(X_val)[:, 1]
    best_thr, (p_v, r_v, f_v) = find_best_threshold(y_val, y_val_score)

    y_te_score = clf.predict_proba(X_te)[:, 1]
    y_pred = (y_te_score >= best_thr).astype(int)
    roc_auc = roc_auc_score(y_te, y_te_score)
    pr_auc = average_precision_score(y_te, y_te_score)
    print(f"AUC-ROC = {roc_auc:.3f}, AUC-PR = {pr_auc:.3f}")
    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    report = classification_report(y_te, y_pred, digits=4, zero_division=0)

    return {
        "ok": True,
        "mode": mode, "ratio": threshold_ratio, "q": q, "gap": suppression_gap,
        "sparks": int(total_sparks),
        "patches": int(total_patches),
        "pos_rate": float(total_sparks) / float(total_patches + 1e-9),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "best_thr": float(best_thr),
        "report": report,
    }


from finetune.pretrain import LitVQShape

def load_vqshape_with_finetuned(pretrained_ckpt: str, finetuned_ckpt: str | None, device: str):
    lit = LitVQShape.load_from_checkpoint(pretrained_ckpt, map_location=device)
    lit.eval().to(device)
    if not finetuned_ckpt:
        return lit.model.eval().to(device)

    ft = torch.load(finetuned_ckpt, map_location=device)
    sd_ft = ft.get("state_dict", ft)

    sd_cur = lit.state_dict()
    updated = {}
    for k, v in sd_ft.items():
        kk = k[6:] if k.startswith("model.") else k
        if kk in sd_cur:
            updated[kk] = v

    missing, unexpected = lit.load_state_dict({**sd_cur, **updated}, strict=False)
    if missing:
        print(f"[warn] missing keys after finetune merge: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys in finetuned ckpt: {len(unexpected)}")
    return lit.model.eval().to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    PRETRAINED_CKPT = "checkpoints/uea_dim256_codebook512/VQShape.ckpt"
    FINETUNED_CKPT  = CKPT_PATH
    vq_model = load_vqshape_with_finetuned(PRETRAINED_CKPT, FINETUNED_CKPT, device)

    NP = int(vq_model.num_patch)
    PP = int(vq_model.patch_size)
    SEQ_LEN = NP * PP
    print(f"VQShape config => num_patch: {NP} patch_size: {PP} seq_len: {SEQ_LEN}")

    df = load_data(DATA_CSV, DROP_COL)

    print("=== Single Run (A+C) ===")
    mode  = "ratio"
    ratio = 0.005
    gap   = 5
    mode_tag = "AP(precursor)" if args.label_mode == "precursor" else "AD"

    res = run_one(df, vq_model, device, mode, ratio, None, gap, NP, PP, SEQ_LEN)
    results = []
    if res["ok"]:
        print(f"[{mode_tag} mode={mode} ratio={ratio} gap={gap}] "
            f"sparks={res['sparks']}/{res['patches']} "
            f"pos_rate={res['pos_rate']:.4f} "
            f"thr={res['best_thr']:.2f} "
            f"P={res['precision']:.3f} R={res['recall']:.3f} F1={res['f1']:.3f} ")
        results.append(res)
        print(res["report"])
    else:
        print(f"[mode=ratio ratio={ratio} gap={gap}] FAILED: {res['msg']}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/result.txt", "w", encoding="utf-8") as f:
        for r in results:
            tag = (f"{mode_tag} "
            f"mode={r['mode']};ratio={r['ratio']};q={r['q']};gap={r['gap']}")
            line = (f"{tag} | sparks={r['sparks']}/{r['patches']} "
            f"| pos_rate={r['pos_rate']:.4f} "
            f"| thr={r['best_thr']:.2f} "
            f"| P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
            )

            f.write(line + "\n")
    print("Saved: outputs/result.txt")


if __name__ == "__main__":
    main()
