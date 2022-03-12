import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

seed = 42
torch.manual_seed(seed)
val_size = 0.5
test_size = 0.2
epochs = 100
batch_size = 32
lr = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data():
    global input_shape
    dataset = pd.read_csv(
        "agaricus-lepiota.data",
        header=None,
        names=["label"] + [f"feature{i}" for i in range(22)],
    )
    features, labels = dataset.iloc[:, 1:], dataset.label
    features_encoded = np.float32(pd.get_dummies(features))
    labels_encoded = np.float32(labels.apply(lambda x: 0 if x == "e" else 1))
    input_shape = len(features_encoded[0])
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
    loss_ = 0
    correct = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x).squeeze()
        loss = F.binary_cross_entropy_with_logits(output, y)
        loss_ += loss.item()
        correct += torch.sigmoid(output).round().eq(y).sum().item()
        loss.backward()
        optimizer.step()
    return (loss.item(), correct)


def evaluate(model, dataloader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x).squeeze()
            loss += F.binary_cross_entropy_with_logits(output, y).item()
            correct += torch.sigmoid(output).round().eq(y).sum().item()
    return (loss, correct)


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

    # Plot loss and accuracy curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(np.arange(epochs), train_loss_hist, label="train loss")
    ax[0].plot(np.arange(epochs), val_loss_hist, label="val loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Train/validation loss")
    ax[0].legend()

    ax[1].plot(np.arange(epochs), train_acc_hist, label="train acc")
    ax[1].plot(np.arange(epochs), val_acc_hist, label="val acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Train/validation acc")
    ax[1].legend()

    plt.show()


if __name__ == "__main__":
    main()
