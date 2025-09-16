# only keep mode=ratio，threshold_ratio=0.005，gap=5
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier


PATCH_SIZE   = 30
WINDOW_SIZE  = 5
STRIDE       = 5
RATIO        = 0.005
GAP          = 5
EPS          = 1e-8
SEQ_LEN      = 512   


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


def build_one_sensor_patches(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    seq_len: int = SEQ_LEN,
):
    """
    Slice patches and assign AD labels (ratio threshold + suppression inside patch).
    Returns:
        patches: List[torch.FloatTensor] with shape [1,1,seq_len]
        labels:  List[int] (0/1), 1 if at least one non-suppressed spark in the last patch window
        starts:  List[int] start index of each patch in the original series
    """
    X_patches, y_labels, starts = [], [], []
    step_max = len(signal) - (patch_size + window_size) + 1
    if step_max <= 0:
        return X_patches, y_labels, starts

    for start in range(0, step_max, stride):
        end = start + window_size + patch_size
        full_patch = signal[start:end]
        region = full_patch[window_size:]  # length = patch_size

        # suppression inside the patch
        last_fire = -10**9
        sparks = []
        for t in range(window_size, len(full_patch)):
            mu = np.mean(full_patch[t - window_size: t])
            val = full_patch[t]
            is_spark = abs(val - mu) / (abs(mu) + EPS) > RATIO
            rel_t = t - window_size
            if is_spark and (rel_t - last_fire) >= GAP:
                sparks.append(1); last_fire = rel_t
            else:
                sparks.append(0)

        y_labels.append(int(any(sparks[-patch_size:])))
        starts.append(start)

        x = torch.tensor(region, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,P]
        x = F.interpolate(x, size=seq_len, mode="linear")  # [1,1,SEQ_LEN]
        X_patches.append(x)

    return X_patches, y_labels, starts

def shift_to_precursor_labels(y_labels, drop_last=True):
    """Convert AD labels y[i] to AP/precursor labels y'[i] = y[i+1]."""
    y = np.asarray(y_labels, dtype=int)
    if len(y) == 0:
        return y_labels, len(y_labels)
    y_next = np.roll(y, -1)
    y_next[-1] = 0
    if drop_last:
        y_next = y_next[:-1]
    return y_next.tolist(), len(y_next)


def load_vq_model(finetune_ckpt: str, pretrain_ckpt: str, device: str = "cuda"):
    """Load finetuned VQShape: build arch from pretrain, then load finetune weights."""
    from finetune.pretrain import LitVQShape
    lit = LitVQShape.load_from_checkpoint(pretrain_ckpt, map_location=device)
    state = torch.load(finetune_ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    lit.load_state_dict(state, strict=False)
    model = lit.model.to(device).eval()
    return model

@torch.no_grad()
def patches_to_histograms(model, patches, device="cuda", use_prob=False):
    """
    Encode each patch -> code indices -> histogram per patch.
    Args:
        patches: List[torch.FloatTensor [1,1,SEQ_LEN]]
    Returns:
        feats: np.ndarray [N, K] where K = codebook size
    """
    codebook = model.codebook.embedding
    K = codebook.num_embeddings
    z_dim = model.encoder.transformer.layers[0].self_attn.embed_dim
    c_dim = codebook.embedding_dim
    projector = torch.nn.Linear(z_dim, c_dim, device=device)

    hist_list = []
    for x in patches:
        x = x.to(device)  # [1,1,T]
        # simple per-sample normalization
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
        # encode -> project -> vector-quantize
        z = model.encoder.patch_and_embed(x)   # [1, L, z_dim]
        z = model.encoder.transformer(z)       # [1, L, z_dim]
        z = projector(z)                       # [1, L, c_dim]
        _, idx, _ = model.codebook(z)          # [1, L]
        idx = idx.view(-1)
        hist = torch.bincount(idx, minlength=K).float()
        if use_prob:
            s = hist.sum()
            hist = hist / s if s > 0 else hist
        hist_list.append(hist.cpu().numpy())
    return np.stack(hist_list, axis=0)  # [N,K]

def apply_tfidf(train_feats: np.ndarray, test_feats: np.ndarray):
    """
    Compute IDF on train set and apply to both train/test features.
    IDF = log((N+1)/(df+1)) + 1
    """
    N, K = train_feats.shape
    df = (train_feats > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1.0
    return train_feats * idf, test_feats * idf

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


def main():
    ap = argparse.ArgumentParser(description="Single-route VQ+GBM pipeline")
    ap.add_argument("--data_csv", type=str, default="GEM1h.csv")
    ap.add_argument("--finetune_ckpt", type=str, default="finetuned_small_spark.ckpt")
    ap.add_argument("--pretrain_ckpt", type=str, default="checkpoints/uea_dim256_codebook512/VQShape.ckpt")
    ap.add_argument("--label_mode", choices=["ad", "precursor"], default="ad",
                    help="ad=detect anomaly in current patch; precursor=predict anomaly in next patch")
    ap.add_argument("--drop_last_for_precursor", action="store_true",
                    help="drop last sample when converting AD->AP")
    ap.add_argument("--use-prob", action="store_true", help="use probability instead of raw counts for histograms")
    ap.add_argument("--use-tfidf", action="store_true", help="apply IDF weighting on histograms (after prob)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    df = load_data(args.data_csv)

    patches_all, labels_all, starts_all = [], [], []
    for col in df.columns:
        sig = df[col].values.astype(float)
        Xp, yp, sp = build_one_sensor_patches(sig)
        if len(yp) == 0: 
            continue
        patches_all.extend(Xp); labels_all.extend(yp); starts_all.extend([(col, s) for s in sp])

    if args.label_mode == "precursor":
        labels_all, keep_len = shift_to_precursor_labels(labels_all, drop_last=args.drop_last_for_precursor)
        patches_all = patches_all[:keep_len]
        starts_all  = starts_all[:keep_len]

    y = np.asarray(labels_all, dtype=int)
    N = len(y)
    if N == 0:
        raise RuntimeError("No samples constructed. Check data length vs (patch+window) and stride.")

    X_idx = np.arange(N)
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(X_idx, y, X_idx, test_size=0.2, random_state=args.seed, stratify=y)
    X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(X_tr, y_tr, idx_tr, test_size=0.25, random_state=args.seed, stratify=y_tr)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vq_model(args.finetune_ckpt, args.pretrain_ckpt, device=device)

    def stack_patches(id_list):
        return [patches_all[i] for i in id_list]

    feats_tr = patches_to_histograms(model, stack_patches(X_tr), device=device, use_prob=args.use_prob)
    feats_va = patches_to_histograms(model, stack_patches(X_va), device=device, use_prob=args.use_prob)
    feats_te = patches_to_histograms(model, stack_patches(X_te), device=device, use_prob=args.use_prob)

    if args.use_tfidf:
        feats_tr, feats_va = apply_tfidf(feats_tr, feats_va)
        _,        feats_te = apply_tfidf(feats_tr, feats_te)  # use IDF from train


    clf = LGBMClassifier(**LGBM_KW)
    clf.fit(
        feats_tr, y_tr,
        eval_set=[(feats_va, y_va)],
        eval_metric="auc",
        callbacks=[torch.cuda.empty_cache] if torch.cuda.is_available() else None,
        verbose=False
    )


    s_va = clf.predict_proba(feats_va)[:, 1]
    best_thr, (p_va, r_va, f1_va) = find_best_threshold(y_va, s_va)

    s_te = clf.predict_proba(feats_te)[:, 1]
    yhat_te = (s_te >= best_thr).astype(int)

    p_te, r_te, f1_te, _ = precision_recall_fscore_support(y_te, yhat_te, average="binary", zero_division=0)
    roc_te = roc_auc_score(y_te, s_te)
    pr_te  = average_precision_score(y_te, s_te)

    metrics = {
        "val_best_thr": round(best_thr, 4),
        "val_P": round(p_va, 4), "val_R": round(r_va, 4), "val_F1": round(f1_va, 4),
        "test_P": round(p_te, 4), "test_R": round(r_te, 4), "test_F1": round(f1_te, 4),
        "test_ROC_AUC": round(roc_te, 4), "test_PR_AUC": round(pr_te, 4),
        "N_train": int(len(y_tr)), "N_val": int(len(y_va)), "N_test": int(len(y_te)),
        "pos_rate_train": round(float(y_tr.mean()), 4),
        "pos_rate_val":   round(float(y_va.mean()), 4),
        "pos_rate_test":  round(float(y_te.mean()), 4),
        "use_prob": bool(args.use_prob), "use_tfidf": bool(args.use_tfidf),
        "label_mode": args.label_mode,
    }
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

    lab_path = os.path.join(args.out_dir, "labels.csv")
    rows = [{"sensor": s[0], "start_index": int(s[1]), "label": int(lbl)} for s, lbl in zip(starts_all, y)]
    pd.DataFrame(rows).to_csv(lab_path, index=False)

if __name__ == "__main__":
    main()
