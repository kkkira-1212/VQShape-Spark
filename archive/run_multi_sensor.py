# Historical script using linear one-step prediction error as labels.
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from finetune.pretrain import LitVQShape
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

patch_size = 20
window_size = 10
threshold = 0.2  
hist_size = 512


plot_targets = []
df = pd.read_csv("GEM1h.csv")
df = df.drop(columns=["Time"]).apply(lambda col: col.str.replace(" µA", "").astype(float))

X_all, y_all = [], []

for col in df.columns:
    signal = df[col].values
    num_patches = (len(signal) - patch_size) // patch_size
    X_patches, y_labels = [], []

    for i in range(num_patches):
        start = i * patch_size
        end = start + patch_size

        if start - window_size < 0:
            continue
        patch_with_context = signal[start - window_size:end]

        window = list(patch_with_context[:window_size])
        spark_flags = []
        pred_list, true_list = [], []

        for t in range(window_size, len(patch_with_context)):
            x_train = np.arange(window_size).reshape(-1, 1)
            y_train = np.array(window).reshape(-1)
            reg = LinearRegression()
            reg.fit(x_train, y_train)

            pred = reg.predict(np.array([[window_size]]))[0]
            y_true = patch_with_context[t]

            pred_list.append(pred)
            true_list.append(y_true)

            if abs(pred - y_true) > threshold:
                spark_flags.append(1)
            else:
                spark_flags.append(0)

            window.pop(0)
            window.append(y_true)

        label = int(any(spark_flags))
        y_labels.append(label)

        if i == 0 and col == df.columns[0]:
            plt.figure(figsize=(10, 4))
            plt.plot(range(window_size, len(patch_with_context)), true_list, label="True")
            plt.plot(range(window_size, len(patch_with_context)), pred_list, label="Predicted")
            plt.title(f"{col} patch 0 prediction vs actual")
            plt.xlabel("Time step in patch")
            plt.ylabel("Current")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        patch = signal[start:end]
        x = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        x = F.interpolate(x, size=512, mode='linear')
        x = rearrange(x, 'b c t -> (b c) t')
        X_patches.append(x)

    print(f"{col} 标签分布: {np.unique(y_labels, return_counts=True)}")

    if len(y_labels) > 0 and all(l == 1 for l in y_labels):
        plot_targets.append(col)

    if sum(y_labels) > 0:
        X_tensor = torch.cat(X_patches, dim=0).cuda()

        if 'vq_model' not in locals():
            ckpt_path = "checkpoints/uea_dim256_codebook512/VQShape.ckpt"
            lit_model = LitVQShape.load_from_checkpoint(ckpt_path, map_location='cuda')
            vq_model = lit_model.model.eval().cuda()

        with torch.no_grad():
            rep, _ = vq_model(X_tensor, mode="tokenize")
            hist = rep['histogram'].cpu().numpy()

        X_all.append(hist)
        y_all.append(np.array(y_labels))


for sensor in plot_targets:
    sig = df[sensor].values[:patch_size]
    plt.figure(figsize=(10, 3))
    plt.plot(sig)
    plt.title(f"Sensor {sensor} - 全为 spark 的 patch")
    plt.xlabel("Time")
    plt.ylabel("Current")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if len(X_all) == 0:
    print("没有任何 spark 样本被触发，请调整 threshold 或 patch_size。")
    exit()

X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)
print("最终样本标签分布：", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LGBMClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, digits=4))
