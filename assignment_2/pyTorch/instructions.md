# Assignment 2 Instructions

This assignment is geared towards letting you guys be familiar with building models in PyTorch using `nn.Module`, and performing hyperparameter grid search manually using Python's `itertools` module. As such, this homework is very similar to the tutorial.

Like in the tutorial, we will be using a manual grid search as a means to find the best hyperparameters for the model *without needing any prior knowledge as to which hyperparameters are best to use*. This is an easy way to get a good performing model given that you have enough time to test hyperparameter permutations. As you progress through the course and in your experience in creating machine learning models, you'll get a better feel for which hyperparameters are best to use for what data.

For this homework, you will be using the CIFAR-100 dataset that is provided within this folder (they're binary files named "train" and "test"). The CIFAR-100 dataset is a collection of small images that all get classified to 100 categories.

![cifar_samples](md_res/cifar.png)

More information on this dataset can be found [here](http://www.cs.utoronto.ca/~kriz/cifar.html)

## 1.
Use the CIFAR-100 dataset found within the assignment folder, then move this data (`train` and `test`) to your project directory.

Below, you are provided with code on how to load the data (which is similar to the instructions provided in the CIFAR website). As such, you will not be needing to make a train-test split yourself, etc.

Here is the code:
```python
with open('./train', 'rb') as f:
    train_dict = pickle.load(f, encoding='bytes')
with open('./test', 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')

X_train = train_dict[b'data'].astype('float32') / 255.0
y_train = np.array(train_dict[b'coarse_labels'])

X_test = test_dict[b'data'].astype('float32') / 255.0
y_test = np.array(test_dict[b'coarse_labels'])
```
With this, you already have your train test split. As you can see, we have 100 categories as answers. Note that we normalize pixel values to the range [0, 1] by dividing by 255 — this is important for activation functions to work correctly.

## 2.
I will be providing you with your parameter dictionary for the grid search:
```python
p = {
    'units': [120, 240],
    'hidden_activations': ['relu', 'sigmoid'],
    'activation': ['softmax', 'sigmoid'],
    'loss': ['mse', 'categorical_crossentropy'],
    'optimizer': ['adam', 'adagrad'],
    'batch_size': [1000, 2000]
}
```

## 3.
The model you will build for the project is very much like what is found in the tutorial, but with some differences as instructed in steps 3 - 7.

Unlike in the tutorial, you do not need to flatten the input, as the CIFAR-100 image data is already provided as a 1-dimensional array (3072 features = 32×32×3). As such, your first `nn.Linear` layer should use `params['units']` as the output size and `3072` as the input size.

## 4.
Add at least 5 `nn.Linear` layers as hidden layers to your model. Remember that more and more hidden layers doesn't necessarily mean higher accuracy, and that more hidden layers increases computational and memory cost.

## 5.
As you see in the previous step 2, unlike the tutorial there is a new key within your parameter dictionary named `'hidden_activations'`. Within your hidden layers, I would like you to set each hidden layer's activation function using `params['hidden_activations']`. In PyTorch, activations are separate modules:
* `'relu'` → `nn.ReLU()`
* `'sigmoid'` → `nn.Sigmoid()`

## 6.
Note that we are using one-hot encoding for our labels. Like what is mentioned in the tutorial, **pay attention to your output layer's number of features**. Please refer to the tutorial if you do not know what I mean. Moreover, depending on which loss function your current hyperparameter combination uses, you may need to think carefully about what format your labels should be in when passed to the loss function. Different loss functions in PyTorch have different expectations — passing the wrong format will either cause an error or silently produce incorrect results.

> **Important note on `categorical_crossentropy`:** In PyTorch, `nn.CrossEntropyLoss()` internally applies `log_softmax`, so when using this loss you should **not** apply a softmax activation to your output layer. Pass raw logits instead. When using `nn.MSELoss()`, apply your output activation normally.

## 7.
Set your number of `epochs` for training to `200`.

## 8 (optional).
Look at the `.csv` output found within `grid_search_output`! You can view it within a spreadsheet application (e.g. Excel, Google Sheets, etc.) to sort the columns if you would like. Additionally, if you know how to use Jupyter notebooks, you can use code like this to see your data in an easy-to-look-at table:

```python
import pandas as pd

df = pd.read_csv('grid_search_output/results.csv')
print(df.sort_values('val_accuracy', ascending=False).to_string(index=False))
```

**Note: do not expect high accuracy. As previously mentioned in class, MLP networks do not do very well on images.**


# Grading rubric
**Out of 100 points**
- 10 points: Load the dataset properly
- 20 points: Have the neural network train
- 20 points: Have the correct `param['dict_keys']` in the right layers/places in the model, the correct # of epochs, etc.
- 40 points: Include a `.csv` of your grid search output with *all 64 permutations* included.
- 10 points: On your best performing hyperparameter permutation, for every accuracy (*not validation accuracy*) point below 20%, lose 1 point. *This is a very low bar just so that you know you're not doing something wrong*.
  - For example, if your best performing hyperparameter permutation gets 12% accuracy, you will get an 8 point deduction.
