# Visualize spark based on sliding window mean deviation (non-predictive definition)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

patch_size = 30
window_size = 10
threshold_ratio = 0.002
eps = 1e-8  # 防止除以 0

df = pd.read_csv("GEM1h.csv")
df = df.drop(columns=["Time"]).apply(lambda col: col.str.replace(" µA", "").astype(float))

os.makedirs("spark_patches", exist_ok=True)

for sensor_col in df.columns:
    signal = df[sensor_col].values
    num_patches = (len(signal) - patch_size - window_size) // patch_size

    for i in range(num_patches):
        start = i * patch_size
        end = start + patch_size + window_size
        full_patch = signal[start:end]

        spark_flags = []
        true_list = []
        mean_list = []

        for t in range(window_size, len(full_patch)):
            window = full_patch[t - window_size:t]
            mean_val = np.mean(window)
            value = full_patch[t]

            true_list.append(value)
            mean_list.append(mean_val)

            if abs(value - mean_val) / (abs(mean_val) + eps) > threshold_ratio:
                spark_flags.append(1)
            else:
                spark_flags.append(0)

        if any(spark_flags):
            spark_indices = [j for j, flag in enumerate(spark_flags) if flag == 1]
            spark_values = [true_list[j] for j in spark_indices]

            plt.figure(figsize=(8, 3))
            plt.plot(range(len(true_list)), true_list, label='Signal')
            plt.plot(range(len(mean_list)), mean_list, linestyle='--', label='Mean (Window)', color='gray')
            plt.scatter(spark_indices, spark_values, color='red', label='Spark', zorder=5)
            plt.title(f"{sensor_col} - Patch #{i} (Spark Detected)")
            plt.xlabel("Time Step in Patch")
            plt.ylabel("Current")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
