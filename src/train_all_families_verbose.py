import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

print("=================================================================")
print("  MULTI-FAMILY TRAINING BENCHMARK (WITH EPOCH-BY-EPOCH LOGGING)  ")
print("=================================================================")

# 1. Load Data
df = pd.read_csv('datasets/uci_ctg/CTG_features_engineered.csv')
X = df.drop(columns=['NSP']).values
y = df['NSP'].values - 1  # 0: Normal, 1: Suspect, 2: Pathologic

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Data Split: {len(y_train)} Train Samples | {len(y_test)} Held-Out Test Samples")
print(f"Test Class Breakdown: Normal={np.sum(y_test==0)}, Suspect={np.sum(y_test==1)}, Pathologic={np.sum(y_test==2)}\n")

# -------------------------------------------------------------
# FAMILY 1: GRADIENT BOOSTED TREES (LightGBM & XGBoost)
# -------------------------------------------------------------
print("-------------------------------------------------------------")
print("[1/5] Training Family 1: Gradient Boosted Trees (250 Boosting Rounds)")
print("-------------------------------------------------------------")
lgb_clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=31, class_weight='balanced', random_state=42, verbose=-1)
lgb_clf.fit(X_train, y_train)
lgb_preds = lgb_clf.predict(X_test)
print(f"  -> LightGBM  | Macro F1: {f1_score(y_test, lgb_preds, average='macro'):.4f} | Pathologic Recall: {recall_score(y_test, lgb_preds, average=None)[2]*100:.2f}%")

xgb_clf = xgb.XGBClassifier(n_estimators=250, learning_rate=0.04, max_depth=6, eval_metric='mlogloss', random_state=42)
xgb_clf.fit(X_train, y_train)
xgb_preds = xgb_clf.predict(X_test)
print(f"  -> XGBoost   | Macro F1: {f1_score(y_test, xgb_preds, average='macro'):.4f} | Pathologic Recall: {recall_score(y_test, xgb_preds, average=None)[2]*100:.2f}%")

# -------------------------------------------------------------
# FAMILY 2: BAGGED DECISION TREES (Random Forest)
# -------------------------------------------------------------
print("\n-------------------------------------------------------------")
print("[2/5] Training Family 2: Bagged Tree Ensembles (300 Trees)")
print("-------------------------------------------------------------")
rf_clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4, class_weight='balanced_subsample', random_state=42)
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)
print(f"  -> Random Forest | Macro F1: {f1_score(y_test, rf_preds, average='macro'):.4f} | Pathologic Recall: {recall_score(y_test, rf_preds, average=None)[2]*100:.2f}%")

# -------------------------------------------------------------
# FAMILY 3: KERNEL MARGIN MODELS (Support Vector Machine)
# -------------------------------------------------------------
print("\n-------------------------------------------------------------")
print("[3/5] Training Family 3: Kernel Margin Models (SVM RBF Kernel)")
print("-------------------------------------------------------------")
svm_clf = SVC(C=2.5, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=42)
svm_clf.fit(X_train_scaled, y_train)
svm_preds = svm_clf.predict(X_test_scaled)
print(f"  -> SVM (RBF) | Macro F1: {f1_score(y_test, svm_preds, average='macro'):.4f} | Pathologic Recall: {recall_score(y_test, svm_preds, average=None)[2]*100:.2f}%")

# -------------------------------------------------------------
# FAMILY 4: LINEAR PROBABILISTIC MODELS (Logistic Regression)
# -------------------------------------------------------------
print("\n-------------------------------------------------------------")
print("[4/5] Training Family 4: Linear Regularized Models (Cost-Sensitive LR)")
print("-------------------------------------------------------------")
lr_clf = LogisticRegression(C=1.5, class_weight='balanced', max_iter=1000, random_state=42)
lr_clf.fit(X_train_scaled, y_train)
lr_preds = lr_clf.predict(X_test_scaled)
print(f"  -> Logistic Regression | Macro F1: {f1_score(y_test, lr_preds, average='macro'):.4f} | Pathologic Recall: {recall_score(y_test, lr_preds, average=None)[2]*100:.2f}%")

# -------------------------------------------------------------
# FAMILY 5: DEEP NEURAL NETWORKS (PyTorch Tabular MLP - Epochs)
# -------------------------------------------------------------
print("\n-------------------------------------------------------------")
print("[5/5] Training Family 5: PyTorch Deep Neural Network (80 Epochs)")
print("-------------------------------------------------------------")
class CTGDeepClassifier(nn.Module):
    def __init__(self, in_features, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class_counts = np.bincount(y_train)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CTGDeepClassifier(in_features=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

epochs = 80
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 20 == 0 or epoch == epochs:
        print(f"    Epoch [{epoch:>2}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

model.eval()
all_preds = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        logits = model(batch_X.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

print(f"  -> PyTorch MLP | Macro F1: {f1_score(y_test, all_preds, average='macro'):.4f} | Pathologic Recall: {recall_score(y_test, all_preds, average=None)[2]*100:.2f}%")

print("\n=================================================================")
print("  ALL 5 FAMILIES TRAINED AND EVALUATED ON HELD-OUT TEST SPLIT!   ")
print("=================================================================")
