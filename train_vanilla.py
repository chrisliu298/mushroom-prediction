"""
Train a simple logistic regression model to predict if a mushroom 
is poisonous based on qualitative features. The mushroom dataset [1]
is from UCI Machine Learning Repository.

Train/Val/Test sizes: 3248/812/4062

Final performance:
Train loss: 0.0085	Train accuracy: 0.9991
  Val loss: 0.0111	  Val accuracy: 0.9988
 Test loss: 0.0082	 Test accuracy: 0.9990

Wall time: 6.06 s

[1] https://archive.ics.uci.edu/ml/datasets/mushroom
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

seed = 42
np.random.seed(seed)
val_size = 0.2
test_size = 0.5
epochs = 500
lr = 1


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
    return (x_train, y_train, x_val, y_val, x_test, y_test)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy_loss(pred, target):
    return -np.mean(target * np.log(pred))


def accuracy(pred, target):
    return np.count_nonzero(pred == target) / len(pred)


def train(beta, x, y):
    beta -= lr * (x.T @ (sigmoid(x @ beta) - y)) / x.shape[0]
    loss = cross_entropy_loss(sigmoid(x @ beta), y)
    acc = accuracy(np.round(sigmoid(x @ beta)), y)
    return (loss, acc)


def evaluate(beta, x, y):
    loss = cross_entropy_loss(sigmoid(x @ beta), y)
    acc = accuracy(np.round(sigmoid(x @ beta)), y)
    return (loss, acc)


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data()
    x_train = np.array(np.hstack((np.ones((x_train.shape[0], 1)), x_train)))
    x_val = np.array(np.hstack((np.ones((x_val.shape[0], 1)), x_val)))
    x_test = np.array(np.hstack((np.ones((x_test.shape[0], 1)), x_test)))

    beta = np.random.standard_normal(input_shape + 1)
    train_loss_hist, train_acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # Training stage
        train_loss, train_acc = train(beta, x_train, y_train)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        # Validation stage
        val_loss, val_acc = evaluate(beta, x_val, y_val)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

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
    test_loss, test_acc = evaluate(beta, x_test, y_test)
    print(" Test loss: {:.4f}\t Test accuracy: {:.4f}".format(test_loss, test_acc))


if __name__ == "__main__":
    main()
