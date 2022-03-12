"""
Train a simple logistic regression model to predict if a mushroom 
is poisonous based on qualitative features. The mushroom dataset [1]
is from UCI Machine Learning Repository.

Train/Val/Test splits: 3248/812/4062

Final performance:
Train loss: 0.0006	Train accuracy: 0.9972
  Val loss: 0.0006	  Val accuracy: 0.9957
 Test loss: 0.0006	 Test accuracy: 0.9975

Wall time: 16.9 s

[1] https://archive.ics.uci.edu/ml/datasets/mushroom
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

seed = 42
torch.manual_seed(seed)
val_size = 0.5
test_size = 0.2
epochs = 50
batch_size = 32
lr = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
dataset = pd.read_csv("agaricus-lepiota.data")
features, labels = dataset.iloc[:, 1:], dataset.p
features_encoded = np.float32(OneHotEncoder().fit_transform(features).todense())
labels_encoded = np.float32(labels.apply(lambda x: 0 if x == "e" else 1))
x_train_val, x_test, y_train_val, y_test = train_test_split(
    features_encoded,
    labels_encoded,
    test_size=test_size,
    random_state=seed,
    shuffle=True,
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=val_size, random_state=seed, shuffle=True
)
x_train, y_train, x_val, y_val, x_test, y_test = map(
    torch.tensor, [x_train, y_train, x_val, y_val, x_test, y_test]
)
train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size, shuffle=True)
val_dataloader = DataLoader(TensorDataset(x_val, y_val), batch_size)
test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size)

# Initialize model
model = nn.Linear(features_encoded.shape[1], 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

train_loss_hist, train_acc_hist = [], []
val_loss_hist, val_acc_hist = [], []

for epoch in range(epochs):
    # Training stage
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x).squeeze()
        loss = F.binary_cross_entropy_with_logits(output, y)
        train_loss += loss.item()
        train_correct += torch.sigmoid(output).round().eq(y).sum().item()
        loss.backward()
        optimizer.step()
    train_loss_hist.append(train_loss / len(train_dataloader.dataset))
    train_acc_hist.append(train_correct / len(train_dataloader.dataset))

    # Validation stage
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x).squeeze()
            val_loss += F.binary_cross_entropy_with_logits(output, y).item()
            val_correct += torch.sigmoid(output).round().eq(y).sum().item()
    val_loss_hist.append(val_loss / len(val_dataloader.dataset))
    val_acc_hist.append(val_correct / len(val_dataloader.dataset))

print(
    "Train loss: {:.4f}\tTrain accuracy: {:.4f}".format(
        train_loss_hist[-1], train_acc_hist[-1]
    )
)
print(
    "  Val loss: {:.4f}\t  Val accuracy: {:.4f}".format(
        val_loss_hist[-1], val_acc_hist[-1]
    )
)

# Final evaluation on test set
model.eval()
test_loss = 0
test_correct = 0
with torch.no_grad():
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        output = model(x).squeeze()
        test_loss += F.binary_cross_entropy_with_logits(output, y).item()
        test_correct += torch.sigmoid(output).round().eq(y).sum().item()

print(
    " Test loss: {:.4f}\t Test accuracy: {:.4f}".format(
        test_loss / len(test_dataloader.dataset),
        test_correct / len(test_dataloader.dataset),
    )
)

torch.save(model.state_dict(), "model.ckpt")
