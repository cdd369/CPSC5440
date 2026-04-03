# Assignment 2

## Table of Contents
- [Part 1: Pre-processing](#part-1-pre-processing)
- [Part 2: Hyperparameters](#part-2-hyperparameters)
- [Part 3: Homework Tutorial](#part-3-homework-tutorial)

## Part 1: Pre-processing
I'd like to emphasize a fundamental concept in data science: pre-processing.

Machine learning is not magic, as most of you already know. Your neural network will perform not as well as it should if you mindlessly throw your training data into it (it will probably perform quite poorly). Many of you may have heard of a phrase similar to "feed a neural network garbage, and you will get a garbage neural network." A lot of what makes a good neural network is not the neural network architecture itself -- it's the data that you use to train it. In simpler terms, we need to make our data as "intuitive" as possible for the machine to learn by removing things that are unnecessary to a prediction. This entails determining what part of the data you would consider important, and what part of the data is not.

Let's take a very simple example: I would like to create a machine learning algorithm to detect the leafy part of a strawberry. Let's assume that the images I will test/validate on *only* has strawberries.

![strawberry leaf](https://i.imgur.com/WnsH1fm.jpg)

Now I would be able to just throw this exact image above into a model, telling the model "this is the leafy part of a strawberry." However, I can do a simple threshold of the image to only keep certain colors within the image, and get something like this:

![thresholded strawberry leaf](https://i.imgur.com/rb9n4fM.png)

As you can see, I've thresholded colors in a way that everything except the greens of the image are simply turned black. Let's assume that through the feature extraction, what the neural network ends up "seeing" is the edges of the picture:

Unprocessed | Processed
------------ | -------------
![unprocessed edge](https://i.imgur.com/O16cN9k.png) | ![processed edge](https://i.imgur.com/0zOprgZ.png)

Unprocessed, there seems to be a lot of edges that are not even part of the leafy part of the strawberry. But on the right, we can see that most of that noise is gone. Just by thresholding certain colors, we were able to get an image without unnecessary data (such as edges that aren't part of the leaves of a strawberry).

Removing unnecessary data can be very beneficial for how your model performs. This example is very simple, but I'd like to emphasize this so you may keep this in mind whenever you're training your models.

## Part 2: Hyperparameters
You may have noticed word "parameters" occasionally been thrown left and right in class (don't quote me on that). While internal parameters are parameters that are set during training (such that their values change as the neural network trains), hyperparameters are parameters that are set before training.

Some common hyperparameters you may encounter during coding:
* Learning rate
* Number of neurons per layer
* Filter size (convolutions)
* Activation function per layer
* Number of convolutions
* Dimensionality of data

Why is this important? Depending on the hyperparameters you choose before even training your neural network may drastically improve (or degrade) the performance of your neural network. That being said, you should pick hyperparameters that make sense for the goal that you're trying to achieve. This should go without saying, but sometimes this is not something completely intuitive to think about.

For example, a look at the sigmoid activation function below. It is apparent that the steepness of the curve between -2 and 2 are much steeper relative to the steepness from -6 to -4, and 4 to 6.

![sigmoid](https://i.imgur.com/lN4ZskZ.png)

When using this in binary classification, where you're trying to output to -1 or 1, this is great! This looks like a smooth step-function, and when data is passed through this activation function, data in the middle will end up having steep differences in value compared to the extremities, in this case closer to 1 or 0. This makes the data less ambiguous when categorizing, for example.

You can kind of visualize what a sigmoid does to data through these images**:

![sig-images](https://i.imgur.com/IKoJG8I.png)

<sub> ** Note: Not really accurate, but is used as a visual analogy. Image from http://ccis2k.org/iajit/PDF/vol.1,no.2/10-nagla.pdf </sub>

You can think of it as the pixels that are already very dark *don't* change value a lot, but the pixels within the image that are roughly halfway in the middle of black and white in the original image get pushed closer to the extremities (in this case, they turn whiter).

Using sigmoids is good in this scenario, but you do not always want to do this. If you're not predicting binary classifications but rather regressions, using sigmoids doesn't make a lot of sense.

Take my one of my research projects that focuses on human motion data. Let's assume that in the GIFs you see below, the pixels from the left to right represent some x-values. We are then taking the x-values of my foot and putting it through a "sigmoid":

"Linear" Activation | "Sigmoid" Activation
------------ | -------------
![norm](https://i.imgur.com/HTLcaSJ.gif) | ![sig](https://i.imgur.com/6TWh0QF.gif)

<sub> Note: Do not @ me about my legs or leg day </sub>

Again, this is not really accurate to what would actually happen to the data, but hopefully this is a good enough example as to why you wouldn't want to use something like a sigmoid for this type of data. In the "sigmoid" GIF, you may notice how my foot motion ends up on either the very left or the very right most of the time, but motion in the middle is almost non-existent, making the movement unnatural for walking. If your goal is to generate natural walking data, then applying this kind of activation function to your data is like giving your neural network bad information to learn with, and you'll find that the performance will be quite horrible. Think of it as trying to write Chinese when all you've been given to learn from is the alphabetical system.

## Part 3: Homework Tutorial

Okay, let's do some coding! We're going to be making an MLP neural network in PyTorch, and doing hyperparameter grid search manually using Python's built-in `itertools` module. Additionally, we are going to be doing some very rudimentary pre-processing to our data today. 

Note: This tutorial uses MNIST as a simple example. In the actual assignment, you will be loading CIFAR-100 from the provided pickle files instead.

### Required libraries
Before we start, make sure you have PyTorch and torchvision installed. These are the libraries we will be using to build and train our neural network, and to download the MNIST dataset. More information can be found here: [PyTorch](https://pytorch.org/)

Using Anaconda Prompt (Windows), or terminal (macOS or Linux), activate your python environment and type this to install these libraries. If you're lost, you may refer to the machine setup guide where you do roughly the same thing.

```
pip install torch torchvision
```

### Defining the neural network

In PyTorch, models are defined as Python classes that inherit from `torch.nn.Module`. You define your layers inside `__init__`, and you define how data flows through them inside `forward`. This structure makes it explicit and easy to see exactly what your network is doing at every step.

#### Imports

Let's import what we need:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import itertools
import csv
import os
```

#### Dataset

For this tutorial, we'll be using MNIST for our dataset. It is a dataset that contains 28 x 28 px images of the digits 0 to 9.

![mnist](md_res/mnist.png)

Let's load the data using `torchvision`. Note how all of the values within images are being divided by 255. As many of you would know, the most common image formats store pixel values from a range of 0 to 255. Dividing this by 255 will allow the image ranges to be 0 - 1, which can be make-or-break for some activation functions. `transforms.ToTensor()` handles this normalization automatically.

```python
train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=transforms.ToTensor())
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Extract as NumPy arrays and normalize to [0, 1]
x_train = train_data.data.numpy().astype('float32') / 255.0
y_train = train_data.targets.numpy()
x_test  = test_data.data.numpy().astype('float32') / 255.0
y_test  = test_data.targets.numpy()

# Each MNIST image is 28x28 pixels. We flatten it into a single vector of
# 784 values so that our fully connected (Linear) layers can process it.
x_train = x_train.reshape(x_train.shape[0], -1)
x_test  = x_test.reshape(x_test.shape[0], -1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
```

To create the label data, we build a one-hot (binary classification) matrix. NumPy's identity matrix `np.eye` gives us a clean way to do this — passing in an array of class indices picks out the corresponding rows, giving each sample a vector of zeros with a single `1` at its class position:

```python
y_train_oh = np.eye(10, dtype='float32')[y_train]
y_test_oh  = np.eye(10, dtype='float32')[y_test]
```

What does the above do? Essentially, this changes how the label data is formatted. Why is this important? Lets say that the digit in question is a 6:

![mnist_6](md_res/6.png)

The one-hot label stored in your training targets for this digit would be:

```[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]```

where at index 6 is a `1`, while all other indices are a `0`.

Note that, when using one-hot encoding, **the amount of nodes/dimensionality of your output layer must equal the length of the label array**. As such, the above label is of length 10, and as such we need 10 output nodes (for this homework).

Why would we use this? This is actually really helpful rather than using scalar variables like what we did in the last assignment. Not only does it increase accuracy, but given certain activation functions for your output layer, can allow for *probabilities* for the output.

Lets say that a trained model is predicting a 6. The output may look something like this:
`[0.15, 0, 0, 0.10, 0, 0.10, 0.55, 0, 0.10, 0]`

The prediction would still be 6, as it has a probability of 55%, but we can also see the probability of predictions that it is making. This is something not possible with scalar representations for labels.

#### Building the neural network

##### Defining a model class

In PyTorch, we define our neural network as a class that inherits from `nn.Module`. There are two things you must always do inside this class:

1. Define all layers inside `__init__`
2. Define how data flows through those layers inside `forward`

A fully connected layer is defined with `nn.Linear(in_features, out_features)`. It holds a weight matrix and a bias vector, and when called, computes `output = input @ weight.T + bias`.

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 24)   # input layer  (784 → 24)
        self.fc2 = nn.Linear(24, 10)    # output layer (24 → 10)

    def forward(self, x):
        x = torch.softmax(self.fc1(x), dim=1)
        x = torch.softmax(self.fc2(x), dim=1)
        return x
```

Where:
* `nn.Linear(in, out)` is a fully connected layer — it connects every input feature to every output neuron with learned weights
* `forward(self, x)` describes how data `x` passes through the network layer by layer
* `torch.softmax(x, dim=1)` applies the softmax activation function along the class dimension

**Keep in mind that the last layer in `forward` is your output layer. Therefore, it should have the same number of neurons as the number of output classes.** For example, we are training on digits from 0-9, giving us 10 different categories. This is the reason why the last `nn.Linear` has 10 output features.

##### Training your model

After defining the model class, three things are needed to train it:
1. A **loss function** — measures how wrong the predictions are
2. An **optimizer** — adjusts the weights to reduce the loss
3. A **training loop** — repeatedly feeds batches of data through the model, computes the loss, and updates the weights

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = MLP().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# DataLoader handles splitting data into mini-batches and shuffling each epoch
dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train_oh))
loader  = DataLoader(dataset, batch_size=2000, shuffle=True)

for epoch in range(100):
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()         # clear gradients from previous step
        output = model(X_batch)       # forward pass: compute predictions
        loss   = criterion(output, y_batch)  # compute loss
        loss.backward()               # backward pass: compute gradients
        optimizer.step()              # update weights using the gradients
```

Where:
* `nn.MSELoss()` computes the mean squared error between predictions and targets
* `optim.Adam(model.parameters())` is the Adam optimizer — it adapts the learning rate for each parameter automatically
* `DataLoader` wraps the dataset and yields shuffled mini-batches during training
* `optimizer.zero_grad()` must be called before each backward pass to clear gradients accumulated from the previous step
* `loss.backward()` computes the gradient of the loss with respect to every learnable parameter
* `optimizer.step()` uses those gradients to update each parameter

We've built and trained the neural network! Your complete code should look something like this:

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 24)
        self.fc2 = nn.Linear(24, 10)

    def forward(self, x):
        x = torch.softmax(self.fc1(x), dim=1)
        x = torch.softmax(self.fc2(x), dim=1)
        return x

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = MLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
dataset   = TensorDataset(torch.tensor(x_train), torch.tensor(y_train_oh))
loader    = DataLoader(dataset, batch_size=2000, shuffle=True)

for epoch in range(100):
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
```

Running this code should entail your neural network training on the MNIST dataset.

### Doing hyperparameter grid search
We've discussed a few hyperparameters in the previous section, namely:
* Number of units
* Activation function
* Loss
* Optimizer
* Batch Size
* Number of epochs

And for each, there are many choices. Let's say you don't have any intuition for what activation function between softmax, sigmoid, and ReLU will perform best on your dataset. What you can do is a hyperparameter grid search. This is an automated way of trying every combination of loss, optimizer, # units, etc. that you specify.

Let's instantiate a dictionary *param_dict* that contains the hyperparameters that we want to be testing:

```python
param_dict = {
    'units':      [12, 24],
    'activation': ['softmax', 'sigmoid'],
    'loss':       ['mse', 'binary_crossentropy'],
    'optimizer':  ['adam', 'adagrad'],
    'batch_size': [1000, 2000]
}
```

As you can see, we set units number, activation, loss, and optimizer params. This gives us 2 x 2 x 2 x 2 x 2 = 32 different combinations of these hyperparameters.

Now, let's wrap our model in a function. We use Python's built-in `itertools.product` to generate every combination of hyperparameters and loop through them. For each combination, we build a fresh model, train it, evaluate it, and log the results — all in plain Python with no external libraries needed.

```python
def my_model(x_train, y_train_oh, x_val, y_val_oh, y_val_idx, params):
```

Inside this function, we build the model dynamically using the values in `params`. Here is what your model inside a function should look like:

```python
def my_model(x_train, y_train_oh, x_val, y_val_oh, y_val_idx, params):
    use_ce = (params['loss'] == 'binary_crossentropy')

    # Choose activation function
    act_fn = nn.Sigmoid() if params['activation'] == 'sigmoid' else nn.Softmax(dim=1)

    model = nn.Sequential(
        nn.Linear(784, params['units']),
        act_fn,
        nn.Linear(params['units'], 10),
        act_fn
    ).to(device)

    opt = (optim.Adam(model.parameters())
           if params['optimizer'] == 'adam'
           else optim.Adagrad(model.parameters()))

    criterion = nn.BCELoss() if use_ce else nn.MSELoss()

    dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train_oh))
    loader  = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model.train()
    for epoch in range(20):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            opt.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            opt.step()

    # Evaluate accuracy on both training and validation sets
    model.eval()
    with torch.no_grad():
        train_out = model(torch.tensor(x_train).to(device))
        val_out   = model(torch.tensor(x_val).to(device))
        train_acc = (train_out.argmax(dim=1) == torch.tensor(y_train_oh.argmax(axis=1)).to(device)).float().mean().item()
        val_acc   = (val_out.argmax(dim=1)   == torch.tensor(y_val_idx).to(device)).float().mean().item()

    return {'accuracy': round(train_acc, 4), 'val_accuracy': round(val_acc, 4)}
```

As you can see, we are changing these hyperparameters in the model into ones that call from param_dict dictionary:
* Units of all layers but the last
* Activation
* Loss
* Optimizer
* Batch size

Now that we have the model wrapped in a function, we can use `itertools.product` to run the hyperparameter grid search. `itertools.product` generates every possible combination of the values across all hyperparameter lists — the same idea as a nested for-loop over every option, but much cleaner. The results are saved to a `.csv` file so you can inspect and compare every run.

```python
keys         = list(param_dict.keys())
combinations = list(itertools.product(*param_dict.values()))
results      = []

for combo in combinations:
    params  = dict(zip(keys, combo))
    metrics = my_model(x_train, y_train_oh, x_test, y_test_oh, y_test, params)
    results.append({**params, **metrics})
    print(f"{params}  =>  acc={metrics['accuracy']:.4f}  val_acc={metrics['val_accuracy']:.4f}")

os.makedirs('grid_search_output', exist_ok=True)
with open('grid_search_output/results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
```

Finally, your complete hyperparameter grid search code should look like this:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import itertools
import csv
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

param_dict = {
    'units':      [12, 24],
    'activation': ['softmax', 'sigmoid'],
    'loss':       ['mse', 'binary_crossentropy'],
    'optimizer':  ['adam', 'adagrad'],
    'batch_size': [1000, 2000]
}

train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=transforms.ToTensor())
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

x_train = train_data.data.numpy().astype('float32') / 255.0
y_train = train_data.targets.numpy()
x_test  = test_data.data.numpy().astype('float32') / 255.0
y_test  = test_data.targets.numpy()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test  = x_test.reshape(x_test.shape[0], -1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train_oh = np.eye(10, dtype='float32')[y_train]
y_test_oh  = np.eye(10, dtype='float32')[y_test]


def my_model(x_train, y_train_oh, x_val, y_val_oh, y_val_idx, params):
    use_ce = (params['loss'] == 'binary_crossentropy')
    act_fn = nn.Sigmoid() if params['activation'] == 'sigmoid' else nn.Softmax(dim=1)

    model = nn.Sequential(
        nn.Linear(784, params['units']),
        act_fn,
        nn.Linear(params['units'], 10),
        act_fn
    ).to(device)

    opt = (optim.Adam(model.parameters())
           if params['optimizer'] == 'adam'
           else optim.Adagrad(model.parameters()))

    criterion = nn.BCELoss() if use_ce else nn.MSELoss()

    dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train_oh))
    loader  = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model.train()
    for epoch in range(20):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            opt.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_out = model(torch.tensor(x_train).to(device))
        val_out   = model(torch.tensor(x_val).to(device))
        train_acc = (train_out.argmax(dim=1) == torch.tensor(y_train_oh.argmax(axis=1)).to(device)).float().mean().item()
        val_acc   = (val_out.argmax(dim=1)   == torch.tensor(y_val_idx).to(device)).float().mean().item()

    return {'accuracy': round(train_acc, 4), 'val_accuracy': round(val_acc, 4)}


keys         = list(param_dict.keys())
combinations = list(itertools.product(*param_dict.values()))
results      = []

for combo in combinations:
    params  = dict(zip(keys, combo))
    metrics = my_model(x_train, y_train_oh, x_test, y_test_oh, y_test, params)
    results.append({**params, **metrics})
    print(f"{params}  =>  acc={metrics['accuracy']:.4f}  val_acc={metrics['val_accuracy']:.4f}")

os.makedirs('grid_search_output', exist_ok=True)
with open('grid_search_output/results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
```

Go ahead and run that code to train it! **Note: if you want to suppress per-combination print output, simply remove or comment out the `print(...)` line inside the loop.**
