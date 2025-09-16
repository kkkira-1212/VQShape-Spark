# generate_old_labels.py
import pandas as pd
import numpy as np

patch_size = 30
window_size = 5
stride = 5
threshold_ratio = 0.0015  
suppression_gap = 5

df = pd.read_csv("GEM1h.csv")
df = df.drop(columns=["Time"]).apply(lambda col: col.str.replace(" µA", "").astype(float))

spark_labels = []
positions = []

for sensor_col in df.columns:
    signal = df[sensor_col].values
    last_spark_idx = -np.inf

    for i in range(0, len(signal) - patch_size - window_size, stride):
        patch = signal[i : i + patch_size + window_size]
        spark_found = 0
        for t in range(window_size, len(patch)):
            local_mean = np.mean(patch[t - window_size : t])
            if abs(patch[t] - local_mean) / (abs(local_mean) + 1e-8) > threshold_ratio:
                if i + t - window_size - last_spark_idx >= suppression_gap:
                    spark_found = 1
                    last_spark_idx = i + t - window_size
                    break

        spark_labels.append(spark_found)
        positions.append((sensor_col, i + window_size))


label_df = pd.DataFrame(positions, columns=["sensor", "index"])
label_df["new_label"] = spark_labels
label_df.to_csv("ae_labels.csv", index=False)
print("✅ Saved AE labels with old logic to ae_labels.csv")
