"""
Train a simple logistic regression model to predict if a mushroom 
is poisonous based on qualitative features. The mushroom dataset [1]
is from UCI Machine Learning Repository.

Train/Val/Test sizes: 3248/812/4062

Final performance:
Train loss: 0.0006	Train accuracy: 0.9972
  Val loss: 0.0006	  Val accuracy: 0.9957
 Test loss: 0.0006	 Test accuracy: 0.9975

Wall time: 23.6 s

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
def prepare_data():
    global input_shape
    dataset = pd.read_csv("agaricus-lepiota.data")
    features, labels = dataset.iloc[:, 1:], dataset.p
    features_encoded = np.float32(OneHotEncoder().fit_transform(features).todense())
    labels_encoded = np.float32(labels.apply(lambda x: 0 if x == "e" else 1))
    input_shape = features_encoded[0].shape[1]
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
    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train), batch_size, shuffle=True
    )
    val_dataloader = DataLoader(TensorDataset(x_val, y_val), batch_size)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size)
    return (train_dataloader, val_dataloader, test_dataloader)


def init_model():
    model = nn.Linear(input_shape, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return (model, optimizer)


def train(model, optimizer, dataloader):
    model.train()
    train_loss = 0
    train_correct = 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x).squeeze()
        loss = F.binary_cross_entropy_with_logits(output, y)
        train_loss += loss.item()
        train_correct += torch.sigmoid(output).round().eq(y).sum().item()
        loss.backward()
        optimizer.step()
    return (train_loss, train_correct)


def evaluate(model, dataloader):
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x).squeeze()
            val_loss += F.binary_cross_entropy_with_logits(output, y).item()
            val_correct += torch.sigmoid(output).round().eq(y).sum().item()
    return (val_loss, val_correct)


def main():
    train_dataloader, val_dataloader, test_dataloader = prepare_data()
    model, optimizer = init_model()
    train_loss_hist, train_acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # Training stage
        train_loss, train_correct = train(model, optimizer, train_dataloader)
        train_loss_hist.append(train_loss / len(train_dataloader.dataset))
        train_acc_hist.append(train_correct / len(train_dataloader.dataset))
        # Validation stage
        val_loss, val_correct = evaluate(model, val_dataloader)
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
    test_loss, test_correct = evaluate(model, test_dataloader)
    print(
        " Test loss: {:.4f}\t Test accuracy: {:.4f}".format(
            test_loss / len(test_dataloader.dataset),
            test_correct / len(test_dataloader.dataset),
        )
    )
    # Save model
    torch.save(model.state_dict(), "model.ckpt")


if __name__ == "__main__":
    main()
