# mushroom-prediction

Train a simple logistic regression model to predict if a mushroom 
is poisonous based on qualitative features using PyTorch. 
The mushroom dataset [1] is from UCI Machine Learning Repository.

Train/Val/Test splits: 3248/812/4062

Final performance:
Train loss: 0.0006	Train accuracy: 0.9972
  Val loss: 0.0006	  Val accuracy: 0.9957
 Test loss: 0.0006	 Test accuracy: 0.9975

Wall time: 16.9 s

# How to run

```shell
python3 train.py
```

# References 

1. [Mushroom Data Set](https://archive.ics.uci.edu/ml/datasets/mushroom)