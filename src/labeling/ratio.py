# fine-tune ratio of spark
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from finetune.pretrain import LitVQShape
import matplotlib.pyplot as plt


patch_size = 30
window_size = 3
ckpt_path = "checkpoints/uea_dim256_codebook512/VQShape.ckpt"
threshold_ratios = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02]

df = pd.read_csv("GEM1h.csv")
df = df.drop(columns=["Time"]).apply(lambda col: col.str.replace(" µA", "").astype(float))

results = []

for threshold_ratio in threshold_ratios:
    X_all, y_all = [], []
    for col in df.columns:
        signal = df[col].values
        num_patches = (len(signal) - patch_size - window_size) // patch_size
        X_patches, y_labels = [], []

        for i in range(num_patches):
            start = i * patch_size
            end = start + patch_size + window_size
            full_patch = signal[start:end]

            spark_flags = []
            for t in range(window_size, len(full_patch)):
                window = full_patch[t - window_size:t]
                mean_val = np.mean(window)
                value = full_patch[t]
                if abs(value - mean_val) / (abs(mean_val) + 1e-8) > threshold_ratio:
                    spark_flags.append(1)
                else:
                    spark_flags.append(0)

            label = int(any(spark_flags[-patch_size:]))
            y_labels.append(label)

            patch_segment = full_patch[window_size:]
            x = torch.tensor(patch_segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
            x = F.interpolate(x, size=512, mode='linear')
            x = rearrange(x, 'b c t -> (b c) t')
            X_patches.append(x)

        if sum(y_labels) > 0:
            X_tensor = torch.cat(X_patches, dim=0).cuda()
            if 'vq_model' not in locals():
                lit_model = LitVQShape.load_from_checkpoint(ckpt_path, map_location='cuda')
                vq_model = lit_model.model.eval().cuda()
            with torch.no_grad():
                rep, _ = vq_model(X_tensor, mode="tokenize")
                hist = rep["histogram"].cpu().numpy()
            X_all.append(hist)
            y_all.append(np.array(y_labels))

    if len(X_all) == 0:
        continue

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "threshold_ratio": threshold_ratio,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1-score": report["1"]["f1-score"],
        "accuracy": report["accuracy"]
    })


results_df = pd.DataFrame(results)
print(results_df.sort_values(by="f1-score", ascending=False))


plt.figure(figsize=(8, 4))
plt.plot(results_df["threshold_ratio"], results_df["f1-score"], marker='o', color='green', label='F1-score')
plt.title("F1-score vs Threshold Ratio")
plt.xlabel("Threshold Ratio")
plt.ylabel("F1-score")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
