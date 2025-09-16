#refer to label
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

patch_size = 30
window_size = 5
stride = 30
threshold_ratio = 0.002

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

model = AE(input_dim=patch_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
scaler = StandardScaler()

X_all, patch_positions, errors_all = [], [], []

for sensor_col in df.columns:
    signal = df[sensor_col].values
    for i in range(0, len(signal) - patch_size - window_size, stride):
        start = i + window_size
        patch = signal[start:start + patch_size]
        local_window = signal[i: i + window_size]
        local_mean = np.mean(local_window)

        patch_reshaped = patch.reshape(-1, 1)
        norm_patch = scaler.fit_transform(patch_reshaped).flatten()

        is_spark = abs(patch[-1] - local_mean) / abs(local_mean) > threshold_ratio

        x = torch.tensor(norm_patch, dtype=torch.float32)
        X_all.append(x)
        patch_positions.append((sensor_col, i + window_size))
        errors_all.append(int(is_spark))

X_all_tensor = torch.stack(X_all).cuda()

for epoch in range(20):
    model.train()
    perm = torch.randperm(len(X_all_tensor))
    total_loss = 0
    for i in range(0, len(X_all_tensor), 64):
        idx = perm[i:i+64]
        batch = X_all_tensor[idx]
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: AE Loss = {total_loss:.6f}")

model.eval()
with torch.no_grad():
    recon_all = model(X_all_tensor)
    errors = torch.mean((X_all_tensor - recon_all) ** 2, dim=1).cpu().numpy()
np.save("ae_reconstruction_errors.npy", errors)

plt.hist(errors, bins=50)
plt.title("AE Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Patches")
plt.grid(True)
plt.show()

threshold_95 = np.percentile(errors, 95)
print(f"95th Percentile Threshold = {threshold_95:.6f}")
pred_labels = (errors > threshold_95).astype(int)
detected_count = np.sum(pred_labels)
print(f"Detected sparks (98th threshold): {detected_count} / {len(errors)} ({100 * detected_count / len(errors):.2f}%)")


N = 20
spark_indices = np.where(pred_labels == 1)[0][:N]

for i, idx in enumerate(spark_indices):
    signal = X_all_tensor[idx].cpu().numpy()
    sensor_name, start_idx = patch_positions[idx]

    plt.figure(figsize=(8, 3))
    plt.plot(signal, label="Normalized Patch", color='black')
    plt.title(f"[AE Spark] Sensor: {sensor_name}, Start Index: {start_idx}")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Current")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"spark_patch_{i+1}_{sensor_name}_{start_idx}.png")
    # plt.show()
    plt.close()


from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


errors_all_np = np.array(errors_all)
pred_labels_np = np.array(pred_labels)

report = classification_report(errors_all_np, pred_labels_np, output_dict=True)
precision = report["1"]["precision"]
recall = report["1"]["recall"]
f1 = report["1"]["f1-score"]

print(f"\n🔥 Classification Report (New vs Old Spark Labels):")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


diff_list = []
patch_index_list = []

for i in range(len(pred_labels_np)):
    if pred_labels_np[i] == 1:
        norm_patch = X_all_tensor[i].cpu().numpy().reshape(-1, 1)
        original_patch = scaler.inverse_transform(norm_patch).flatten()
        last_point = original_patch[-1]
        patch_mean = original_patch.mean()
        diff = last_point - patch_mean

        patch_index_list.append(i)
        diff_list.append(diff)


df_diff = pd.DataFrame({
    "Patch Index": patch_index_list,
    "Current Deviation (µA)": diff_list
})

print("\n⚡ Deviation of Detected Sparks (last point vs. patch mean):")
print(df_diff.head(10))  
