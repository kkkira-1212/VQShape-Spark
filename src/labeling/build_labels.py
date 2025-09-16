#old label vs new label
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

patch_size = 30
window_size = 5
stride = 5
threshold_ratio = 0.015  
ae_threshold_percentile = 95  
suppression_gap = 5

df = pd.read_csv("GEM1h.csv")
df = df.drop(columns=["Time"]).apply(lambda col: col.str.replace(" µA", "").astype(float))

class AE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


scaler = StandardScaler()
X_all = []
old_labels = []
patch_positions = []
sensor_names = []
global_step = 0
last_spark_global_idx = -np.inf

for sensor_col in df.columns:
    signal = df[sensor_col].values

    signal_reshaped = signal.reshape(-1, 1)
    signal_scaled = scaler.fit_transform(signal_reshaped).flatten()

    num_patches = (len(signal_scaled) - patch_size - window_size) // patch_size

    for i in range(num_patches):
        start = i * patch_size
        end = start + patch_size + window_size
        full_patch = signal_scaled[start:end]

        patch_segment = full_patch[window_size:]

        spark_flags = []
        for t in range(window_size, len(full_patch)):
            local_window = full_patch[t - window_size:t]
            local_mean = np.mean(local_window)
            value = full_patch[t]

            is_spark = abs(value - local_mean) / (abs(local_mean) + 1e-8) > threshold_ratio
            is_suppressed = False

            if is_spark:
                if global_step - last_spark_global_idx >= suppression_gap:
                    spark_flags.append(1)
                    last_spark_global_idx = global_step
                else:
                    spark_flags.append(0)
                    is_suppressed = True
            else:
                spark_flags.append(0)

            global_step += 1

        old_label = int(any(spark_flags[-patch_size:]))
        old_labels.append(old_label)

        x = torch.tensor(patch_segment, dtype=torch.float32)
        X_all.append(x)

        patch_positions.append((sensor_col, i * patch_size + window_size))
        sensor_names.append(sensor_col)

        
X = np.concatenate(X_all, axis=0)        
X_all_tensor = torch.stack(X_all).cuda()

model = AE(input_dim=patch_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(20):
    model.train()
    perm = torch.randperm(len(X_all_tensor))
    for i in range(0, len(X_all_tensor), 64):
        idx = perm[i:i+64]
        batch = X_all_tensor[idx]
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
with torch.no_grad():
    recon_all = model(X_all_tensor)
    ae_errors = torch.mean((X_all_tensor - recon_all) ** 2, dim=1).cpu().numpy()

ae_threshold = np.percentile(ae_errors, ae_threshold_percentile)
new_labels = (ae_errors > ae_threshold).astype(int)

df_compare = pd.DataFrame(patch_positions, columns=["sensor", "index"])
# df_compare["old_label"] = old_labels
df_compare["new_label"] = new_labels[:len(df_compare)]  
df_compare.to_csv("ae_labels.csv", index=False)
print("success")

from sklearn.metrics import classification_report
# print(classification_report(old_labels, new_labels[:len(old_labels)], digits=4))

old_count = sum(old_labels)
new_count = sum(new_labels)
both_spark = sum([1 for o, n in zip(old_labels, new_labels) if o == 1 and n == 1])
print(f"\n Old sparks: {old_count},  New sparks: {new_count}, Overlap: {both_spark}")

deviation_results = []
for i, (is_new, pos) in enumerate(zip(new_labels, patch_positions)):
    if is_new:
        sensor, idx = pos
        original_patch = df[sensor].values[idx : idx + patch_size]
        # last_point = original_patch[-1]
        patch_mean = np.mean(original_patch)
        deviation = np.max(np.abs(original_patch - patch_mean))
        deviation_results.append((i, deviation))

deviation_df = pd.DataFrame(deviation_results, columns=["Patch Index", "Current Deviation (µA)"])
print("\n Deviation of Detected Sparks (last point vs. patch mean):")
print(deviation_df.head(50))

plt.hist(ae_errors, bins=50)
plt.axvline(ae_threshold, color='r', linestyle='--', label=f'{ae_threshold_percentile}th Threshold')
plt.title("AE Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Patches")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

fire_patch_indices = np.where(new_labels == 1)[0]

num_to_plot = 50
print(f"Total new sparks: {len(fire_patch_indices)} | Showing first {num_to_plot} patches")

for i, idx in enumerate(fire_patch_indices[:num_to_plot]):
    sensor, patch_start = patch_positions[idx]
    patch_signal = df[sensor].values[patch_start:patch_start + patch_size]
    patch_mean = np.mean(patch_signal)

    plt.figure(figsize=(10, 3))
    plt.plot(range(patch_size), patch_signal, label="Current", linewidth=1.5)
    plt.axhline(patch_mean, color='gray', linestyle='--', label="Patch Mean")
    plt.title(f"Patch #{idx} | Sensor: {sensor} | Start Index: {patch_start}")
    plt.xlabel("Time Step in Patch")
    plt.ylabel("Current (µA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

