# src/features/vqshape_tokens.py

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SparkDataset(Dataset):

    def __init__(self, dataframe, patch_size=30, window_size=5, stride=5, label_file="ae_labels.csv"):
        self.X, self.Y, self.labels = [], [], []

        old_df = pd.read_csv(label_file)
        label_lookup = {(row["sensor"], row["index"]): int(row["new_label"]) for _, row in old_df.iterrows()}

        for sensor_col in dataframe.columns:
            signal = dataframe[sensor_col].values
            # iterate over sliding patches
            for i in range(0, len(signal) - patch_size - window_size, stride):
                x_patch = signal[i : i + patch_size]
                y_patch = signal[i + patch_size : i + patch_size + window_size]
                self.X.append(x_patch)
                self.Y.append(y_patch)
                # label is aligned to the first prediction time inside the window-aligned index
                pos = (sensor_col, i + window_size)
                self.labels.append(label_lookup.get(pos, 0))

        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(1)
        self.Y = torch.tensor(self.Y, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.labels[idx]


# ----------------------------
# Model loaders (finetune + decoder)
# ----------------------------
def load_model(finetune_ckpt="finetuned_small_spark.ckpt",
               pretrain_ckpt="checkpoints/uea_dim256_codebook512/VQShape.ckpt",
               device="cuda"):
    """
    Load finetuned VQShape model:
      1) build architecture from the pretrain checkpoint (ensures matching modules)
      2) load weights from your finetuned checkpoint

    Returns:
        model (torch.nn.Module): lit.model in eval mode on `device`
    """
    from finetune.pretrain import LitVQShape
    lit = LitVQShape.load_from_checkpoint(pretrain_ckpt, map_location=device)
    state = torch.load(finetune_ckpt, map_location=device)
    # Accept both pure state_dict or Lightning 'state_dict' wrapper
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    lit.load_state_dict(state, strict=False)
    model = lit.model.to(device).eval()
    return model


def load_shape_decoder_from_ckpt(ckpt_path, device="cuda"):
    
    from models.vqshape.networks import ShapeDecoder
    shape_decoder = ShapeDecoder(dim_embedding=8, dim_output=128, out_kernel_size=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    prefix = "model.shape_decoder."
    dec_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    shape_decoder.load_state_dict(dec_state, strict=False)
    shape_decoder.eval()
    return shape_decoder


# ----------------------------
# Token activation counting
# ----------------------------
@torch.no_grad()
def count_token_activations(model, dataloader, device="cuda", positive_only=True):
    """
    Count how many times each codebook token is selected by the VQ module.

    Pipeline:
        X --(normalize)--> encoder.patch_and_embed --> transformer --> Linear(z_dim->code_dim) --> codebook
        indices from codebook are counted into a histogram over token IDs.

    Args:
        model: VQShape model with .encoder, .codebook
        dataloader: provides (X, Y, label)
        positive_only (bool): if True, count activations only on positive (label==1) samples.

    Returns:
        token_counts (np.ndarray): shape [K], where K is codebook size.
    """
    model.eval()
    codebook_size = model.codebook.embedding.num_embeddings
    token_counts = torch.zeros(codebook_size, device=device)

    # Derive dims from model
    z_dim = model.encoder.transformer.layers[0].self_attn.embed_dim
    code_dim = model.codebook.embedding.embedding_dim
    projector = torch.nn.Linear(z_dim, code_dim, device=device)

    for x, _, label in dataloader:
        if positive_only:
            mask = (label == 1)
            if mask.sum() == 0:
                continue
            x = x[mask]
        x = x.to(device)

        # simple per-sample normalization
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)

        # encode -> project -> vector-quantize
        z = model.encoder.patch_and_embed(x)     # [B, T, z_dim]
        z = model.encoder.transformer(z)         # [B, T, z_dim]
        z = projector(z)                         # [B, T, code_dim]
        _, indices, _ = model.codebook(z)        # [B, T]
        token_counts += torch.bincount(indices.view(-1), minlength=codebook_size)

    return token_counts.detach().cpu().numpy()


# ----------------------------
# CLI entry: build -> load -> count -> save
# ----------------------------
def _read_numeric_csv(path_csv: str) -> pd.DataFrame:
    """Read CSV and coerce non-numeric columns (e.g., '12.3 µA') into float."""
    df = pd.read_csv(path_csv)
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    for c in df.columns:
        # strip unit suffix if present
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.extract(r"([-+]?[0-9]*\.?[0-9]+)").astype(float)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="VQShape-Spark core: token activation counting")
    parser.add_argument("--data_csv", type=str, default="GEM1h.csv", help="Input multivariate time series CSV")
    parser.add_argument("--label_csv", type=str, default="ae_labels.csv", help="AE labels CSV (sensor,index,new_label)")
    parser.add_argument("--finetune_ckpt", type=str, default="finetuned_small_spark.ckpt", help="Finetuned checkpoint")
    parser.add_argument("--pretrain_ckpt", type=str, default="checkpoints/uea_dim256_codebook512/VQShape.ckpt", help="Pretrain checkpoint to build architecture")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patch_size", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="analysis_out")
    parser.add_argument("--topk", type=int, default=10, help="How many top tokens to display in console")
    parser.add_argument("--positive_only", action="store_true", help="Count activations only on label==1 samples")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load data
    df = _read_numeric_csv(args.data_csv)

    # 2) build dataset/loader
    ds = SparkDataset(
        df,
        patch_size=args.patch_size,
        window_size=args.window_size,
        stride=args.stride,
        label_file=args.label_csv,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # 3) load model
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.finetune_ckpt, args.pretrain_ckpt, device=device)

    # 4) count activations
    counts = count_token_activations(model, loader, device=device, positive_only=args.positive_only)

    # 5) save artifacts
    npy_path = os.path.join(args.out_dir, "token_counts.npy")
    csv_path = os.path.join(args.out_dir, "token_counts.csv")
    np.save(npy_path, counts)
    pd.DataFrame({"token_id": np.arange(len(counts)), "count": counts}).to_csv(csv_path, index=False)

    # 6) print top-k summary
    order = np.argsort(counts)[::-1]
    top_ids = order[: args.topk].tolist()
    top_vals = counts[order[: args.topk]].tolist()
    print(f"Top-{args.topk} tokens:", top_ids)
    print("Counts:", [int(v) for v in top_vals])
    print("Saved:", npy_path, "|", csv_path)


if __name__ == "__main__":
    main()
