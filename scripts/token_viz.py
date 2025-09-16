# Visualization utilities for VQShape-Spark tokens.
# Each plot is a dedicated function. Use the CLI in main() to run end-to-end.
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
import matplotlib.ticker as ticker

from scripts.vqshape_tokens import (
    load_model, load_shape_decoder_from_ckpt
)


def plot_activation_topk_bar(token_counts, k=10, save_path="figs/token_topk_bar.png"):
    """Bar chart for top-k token activations."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    idx = np.argsort(token_counts)[::-1][:k]
    vals = token_counts[idx]

    plt.figure(figsize=(8, 4))
    plt.bar([str(i) for i in idx], vals)
    plt.title(f"Top-{k} token activations")
    plt.xlabel("Token ID")
    plt.ylabel("Activations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_codebook_embeddings(model, token_ids, save_path="figs/token_embeddings.png"):
    """Plot selected codebook embeddings across embedding dimensions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    emb = model.codebook.embedding.weight.detach().cpu().numpy()  # [K, D]
    D = emb.shape[1]

    plt.figure(figsize=(8, 5))
    for tid in token_ids:
        plt.plot(range(D), emb[tid], marker="o", label=f"T{tid}")
    plt.title("Selected token embeddings (codebook)")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Value")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_decoded_token_waveforms(model, shape_decoder, token_ids, target_len=15,
                                 save_path="figs/decoded_tokens.png", device="cuda"):
    """Decode tokens to waveforms with ShapeDecoder and plot in a single figure."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = np.linspace(0, target_len, target_len)

    plt.figure(figsize=(7, 6))
    for tid in token_ids:
        emb = model.codebook.embedding.weight[tid].unsqueeze(0).unsqueeze(0).to(device)  # [1,1,8]
        recon_patch, _, _ = shape_decoder(emb)  # [1,1,T_orig]
        y = recon_patch.squeeze().detach().cpu().numpy()
        y = zoom(y, target_len / y.shape[0])
        plt.plot(x, y, label=f"T{tid}")

    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    plt.title("Decoded token waveforms")
    plt.xlabel("Time step"); plt.ylabel("Signal value")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_all_topk_decoded(model, shape_decoder, token_counts, k=10, target_len=15,
                          save_path="figs/decoded_topk.png", device="cuda"):
    """Convenience wrapper: take top-k tokens by count and plot decoded waveforms."""
    idx = np.argsort(token_counts)[::-1][:k]
    plot_decoded_token_waveforms(model, shape_decoder, idx, target_len, save_path, device)


def main():
    parser = argparse.ArgumentParser(description="Token visualization (read-only).")
    parser.add_argument("--counts_npy", type=str, required=True,
                        help="Path to precomputed token_counts.npy from vqshape_tokens.py")
    parser.add_argument("--out_dir", type=str, default="figs", help="Output directory for figures")
    parser.add_argument("--topk", type=int, default=10, help="Top-k tokens for plots")

    # Optional: decode waveforms for top-k tokens
    parser.add_argument("--decode-topk", action="store_true",
                        help="If set, will load models and plot decoded waveforms of top-k tokens")
    parser.add_argument("--finetune_ckpt", type=str, default="finetuned_small_spark.ckpt",
                        help="Finetuned checkpoint (only used when --decode-topk)")
    parser.add_argument("--pretrain_ckpt", type=str, default="checkpoints/uea_dim256_codebook512/VQShape.ckpt",
                        help="Pretrain checkpoint to build architecture (only when --decode-topk)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for optional decoding (cuda/cpu)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load counts produced by vqshape_tokens.py
    counts = np.load(args.counts_npy)

    # 2) basic plots that do not require model
    plot_activation_topk_bar(counts, k=args.topk,
                             save_path=os.path.join(args.out_dir, "token_topk_bar.png"))

    # 3) optional: decode waveforms for top-k tokens (requires model + shape decoder)
    if args.decode_topk:
        top_ids = np.argsort(counts)[::-1][:args.topk]
        model = load_model(args.finetune_ckpt, args.pretrain_ckpt, device=args.device)
        shape_decoder = load_shape_decoder_from_ckpt(args.finetune_ckpt, device=args.device)

        # A) codebook embeddings (top-3 for readability)
        sel = top_ids[:3].tolist()
        plot_codebook_embeddings(model, sel,
                                 save_path=os.path.join(args.out_dir, "token_embeddings_top3.png"))
        # B) decoded waveforms (top-k)
        plot_all_topk_decoded(model, shape_decoder, counts, k=args.topk, target_len=15,
                              save_path=os.path.join(args.out_dir, "decoded_topk.png"), device=args.device)

    print(" Finished. Figures saved to:", args.out_dir)


if __name__ == "__main__":
    main()
