# # A unified tool for data reading/labeling/patching/VQ tokenization
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def load_data(path_csv: str, drop_col: str | None = "Time") -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    if drop_col and drop_col in df.columns:
        df = df.drop(columns=[drop_col])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(" µA", "", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    return df

def shift_to_precursor_labels(y_labels, drop_last=True):
    y = np.asarray(y_labels, dtype=int)
    if len(y) == 0:
        return y_labels, len(y)
    y_next = np.roll(y, -1)
    y_next[-1] = 0
    if drop_last:
        y_next = y_next[:-1]
    return y_next.tolist(), len(y_next)

def build_one_sensor_patches(
    signal: np.ndarray, *,
    mode: str, threshold_ratio: float | None, q: float | None,
    suppression_gap: int, window_size: int, patch_size: int,
    stride: int, seq_len: int
):
    if mode == "quantile":
        diffs = np.abs(np.diff(signal, prepend=signal[0]))
        thr_q = np.quantile(diffs, q)

    X_patches, y_labels, starts = [], [], []
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
                spark_flags.append(1); last_fire = rel_t
            else:
                spark_flags.append(0)

        y_labels.append(int(any(spark_flags[-patch_size:])))
        starts.append(start)

        x = torch.tensor(region, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,P]
        x = F.interpolate(x, size=seq_len, mode="linear")                        # [1,1,SEQ]
        X_patches.append(x)

    spark_count = int(np.sum(y_labels))
    return X_patches, y_labels, starts, spark_count

@torch.no_grad()
def vq_histograms(X_patches, vq_model, device: str, seq_len: int):
    if len(X_patches) == 0:
        return np.zeros((0, 1), dtype=float)
    X_tensor = torch.cat(X_patches, dim=0).to(device)  # [B,1,T]
    assert X_tensor.shape[-1] == seq_len, "seq_len mismatch for VQ"
    rep, _ = vq_model(X_tensor, mode="tokenize")
    return rep["histogram"].detach().cpu().numpy()     # [B, K]

def to_prob(A):
    s = A.sum(axis=1, keepdims=True) + 1e-12
    return A / s

def apply_prob_tfidf(X_tr, X_val, X_te, use_tfidf: bool):
    if not use_tfidf:
        return X_tr, X_val, X_te
    df_vec = (X_tr > 0).sum(axis=0)
    N = X_tr.shape[0]
    idf = np.log((N + 1) / (df_vec + 1)) + 1.0
    return X_tr * idf, X_val * idf, X_te * idf

def build_dataset(
    df: pd.DataFrame, vq_model, device: str, *,
    patch_size=30, window_size=5, stride=5,
    mode="ratio", threshold_ratio=0.005, q=None, suppression_gap=5,
    seq_len=512,
    label_mode="ad", drop_last_for_precursor=True,
    return_concat=True
):
    X_all, y_all, starts_all = [], [], []
    for sensor in df.columns:
        sig = df[sensor].values.astype(float)
        Xp, yl, st, _ = build_one_sensor_patches(
            sig, mode=mode, threshold_ratio=threshold_ratio, q=q,
            suppression_gap=suppression_gap, window_size=window_size,
            patch_size=patch_size, stride=stride, seq_len=seq_len
        )
        if label_mode == "precursor":
            yl, keep_len = shift_to_precursor_labels(yl, drop_last_for_precursor)
            if drop_last_for_precursor:
                Xp = Xp[:keep_len]; st = st[:keep_len]
        H = vq_histograms(Xp, vq_model, device, seq_len)
        X_all.append(H); y_all.append(np.array(yl, dtype=int)); starts_all.append(np.array(st, dtype=int))

    if return_concat:
        X = np.concatenate(X_all, axis=0) if len(X_all) else np.zeros((0,1))
        y = np.concatenate(y_all, axis=0) if len(y_all) else np.zeros((0,), dtype=int)
        starts = np.concatenate(starts_all, axis=0) if len(starts_all) else np.zeros((0,), dtype=int)
        return X, y, starts
    else:
        return X_all, y_all, starts_all
