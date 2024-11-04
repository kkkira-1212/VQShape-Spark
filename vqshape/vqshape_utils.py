import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange


def plot_code_heatmap(code_indices, num_codes, title=''):
    """
    Plots a heatmap for visualizing the use of codes in a vector-quantization model.

    Parameters:
    - code_indices: torch.Tensor, a 2D tensor where each element is a code index.
    - num_codes: int, the total number of different codes.

    The function creates a heatmap where each row represents a different code and
    each column represents a position in the input tensor, showing the frequency of each code.
    """
    code_indices = code_indices.cpu()
    # Initialize a frequency matrix
    codes, counts = torch.unique(code_indices, return_counts=True)
    heatmap = torch.zeros(num_codes).scatter_(-1, codes, counts.float())
    if num_codes <= 64:
        heatmap = heatmap.view(8, -1)
    elif num_codes <= 256:
        heatmap = heatmap.view(16, -1)
    elif num_codes <= 1024:
        heatmap = heatmap.view(32, -1)
    else:
        heatmap = heatmap.view(64, -1)
    # heatmap = heatmap.view(int(np.sqrt(num_codes)), -1)

    heatmap = heatmap.numpy()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Frequency")
    ax.set_title(f'Code Usage Heatmap - step {title}')
    plt.tight_layout()

    return fig


def visualize(x, x_hat, s, s_hat, t, l, mu, sigma, num_sample=10, num_s_sample=25, title=''):
    sample_idx = np.random.randint(0, x.shape[0], num_sample)
    x, x_hat, s, s_hat, t, l, mu, sigma = x.float().cpu().numpy(), x_hat.float().cpu().numpy(), s.float().cpu().numpy(), s_hat.float().cpu().numpy(), t.float().cpu().numpy(), l.float().cpu().numpy(), mu.float().cpu().numpy(), sigma.float().cpu().numpy()
    fig = plt.figure(figsize=(30, 4))
    for i, idx in enumerate(sample_idx):
        ax = fig.add_subplot(num_sample//5, 5, i+1)
        ax.plot(np.linspace(0, 1, x.shape[-1]), x[idx], color='tab:grey', linewidth=5, alpha=0.3)
        ax.plot(np.linspace(0, 1, x.shape[-1]), x_hat[idx], color='tab:blue', linewidth=5, alpha=0.3)
        for j in range(t.shape[1]):
            ts = np.linspace(t[idx, j], min(t[idx, j]+l[idx, j], 1), s_hat[idx, j].shape[-1])
            ax.plot(ts, s_hat[idx, j])
    plt.tight_layout()

    s = rearrange(s, 'B N L -> (B N) L')
    s_hat = rearrange(s_hat, 'B N L -> (B N) L')
    t = rearrange(t, 'B N L -> (B N) L')
    l = rearrange(l, 'B N L -> (B N) L')
    s_samples_idx = np.random.randint(0, s.shape[0], num_s_sample)
    s_fig = plt.figure(figsize=(15, 8))
    for i, idx in enumerate(s_samples_idx):
        ax = s_fig.add_subplot(5, num_s_sample//5, i+1)
        ax.plot(np.linspace(t[idx], t[idx] + l[idx], s.shape[-1]), s[idx], alpha=0.5)
        ax.plot(np.linspace(t[idx], t[idx] + l[idx], s_hat.shape[-1]), s_hat[idx], alpha=0.5)
    plt.tight_layout()

    return fig, s_fig


class Timer:
    def __init__(self):
        self.t = time.time_ns()

    def __call__(self):
        ret = f"Interval: {(time.time_ns() - self.t)/1e6:.1f} ms"
        self.t = time.time_ns()
        return ret


def compute_accuracy(logits, labels):
    """
    Compute the accuracy for multi-class classification.

    Args:
    logits (torch.Tensor): The logits output by the model. Shape: [n_samples, n_classes].
    labels (torch.Tensor): The true labels for the data. Shape: [n_samples].

    Returns:
    float: The accuracy of the predictions as a percentage.
    """
    # Get the indices of the maximum logit values along the second dimension (class dimension)
    # These indices correspond to the predicted classes.
    _, predicted_classes = torch.max(logits, dim=1)

    # Compare the predicted classes to the true labels
    correct_predictions = (predicted_classes == labels).float()  # Convert boolean to float

    # Calculate accuracy
    accuracy = correct_predictions.sum() / len(labels)

    return accuracy.item()  # Convert to Python scalar


def compute_binary_accuracy(logits, labels):
    """
    Compute the accuracy of binary classification predictions.

    Args:
    logits (torch.Tensor): The logits output by the model. Logits are raw, unnormalized scores.
    labels (torch.Tensor): The true labels for the data.

    Returns:
    float: The accuracy of the predictions as a percentage.
    """
    # Convert logits to predictions
    predictions = nn.functional.sigmoid(logits) >= 0.5  # Apply sigmoid and threshold
    labels = labels >= 0.5

    # Compare predictions with true labels
    correct_predictions = (predictions == labels).float()  # Convert boolean to float for summing

    # Calculate accuracy
    accuracy = correct_predictions.sum() / len(labels)

    return accuracy.item()  # Convert to Python scalar


def smooth_labels(labels: torch.Tensor, smoothing: float = 0.05):
    """
    Apply label smoothing to a tensor of binary labels.

    Args:
    labels (torch.Tensor): Tensor of binary labels (0 or 1).
    smoothing (float): Smoothing factor to apply to the labels.

    Returns:
    torch.Tensor: Tensor with smoothed labels.
    """
    # Ensure labels are in float format for the smoothing operation
    labels = labels.float()
    
    # Apply label smoothing
    smoothed_labels = labels * (1 - smoothing) + (1 - labels) * smoothing

    return smoothed_labels


def get_gpu_usage():
    gpu_mem = {}
    for i in range(torch.cuda.device_count()):
        gpu_mem[f'GPU {i}'] = torch.cuda.max_memory_allocated(i)/1e9
        # torch.cuda.reset_peak_memory_stats(i)
    return gpu_mem