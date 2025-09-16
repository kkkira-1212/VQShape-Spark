import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

patch_size = 30
batch_size = 64
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_df = pd.read_csv("ae_labels.csv")  # 包含 sensor, index, new_label
label_dict = {(row['sensor'], row['index']): row['new_label'] for _, row in label_df.iterrows()}

df = pd.read_csv("GEM1h.csv")
df = df.drop(columns=["Time"]).apply(lambda col: col.str.replace(" µA", "").astype(float))

X_patches, y_labels = [], []

for sensor_col in df.columns:
    signal = df[sensor_col].values
    num_patches = (len(signal) - patch_size) // patch_size

    for i in range(num_patches):
        start = i * patch_size
        end = start + patch_size
        patch = signal[start:end]


        spark_in_patch = 0
        for idx in range(start, end):
            if label_dict.get((sensor_col, idx), 0) == 1:
                spark_in_patch = 1
                break

        X_patches.append(patch)
        y_labels.append(spark_in_patch)

X = np.array(X_patches)
y = np.array(y_labels)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [N, 30, 1]
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

class SparkLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

model = SparkLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
pos_weight = torch.tensor([5.0]).to(device)  # 针对不平衡类别
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model.train()
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    print(f"[Epoch {epoch+1}] Loss = {total_loss / len(train_loader.dataset):.4f}")

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = torch.sigmoid(model(xb)).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend((preds > 0.5).astype(int))

print(classification_report(y_true, y_pred, digits=4))
