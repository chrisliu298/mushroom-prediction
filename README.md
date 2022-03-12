# mushroom-prediction

Train a simple logistic regression model to predict if a mushroom 
is poisonous based on qualitative features using PyTorch. 
The mushroom dataset [1] is from UCI Machine Learning Repository.

The final performance is shown below. 

|                  |  Loss  | Accuracy |
| :--------------: | :----: | :------: |
|   Train (3248)   | 0.0006 |  0.9972  |
| Validation (812) | 0.0006 |  0.9957  |
|   Test (4062)    | 0.0006 |  0.9975  |



# How to run

```shell
wget -nv https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
python3 train.py
# Wall time: 16.9 s
```

# References 

1. [Mushroom Data Set](https://archive.ics.uci.edu/ml/datasets/mushroom)