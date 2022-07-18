progress: P2 - M5 - L4 - C3

progress: P7 - M1 - L5 - C8

# Neural Networks

## Introduction

**weights change**

```mathematica
Δwi = -(∂Error / ∂wi)
```

**output layer**
η = learning rate
```mathematica
e = (y - ŷ)
δ = e * ƒ’(h)
Δwi = η * δ * xi
```

**hidden layer**
j = hidden layer, k = output layer
```mathematica
ej = (∑k δk * wjk)
δj = ej * ƒ’(hj)
Δwi = η * δ * xi
```

### Functions

**Sigmoid**
```mathematica
f(x) = 1 / (1 + e^-x)
```

**Soft-max**

```mathematica
f(i) = e^xi / (e^x1 + e^x2 + ... + e^xn)
```

### Error Functions

**cross-entropy loss**
```mathematica
Error = ∑ (yi * log(ŷi) + (1-yi) * log(1-ŷi))
∂Error / ∂ŷ = -(y - ŷ)
```

**errorest**
M = mean square error
```mathematica
Error = 1/M ∑M ½ * (yi - ŷi)²
∂Error / ∂ŷ = - 1/M ∑M (yi - ŷi)
```

**soft-max**
```mathematica
f(h) = e^h / ∑ e^hi
∂ ƒ(hi) / ∂ ha = 
if i = a => ∂ ƒ(hi) / ∂ ha = f(ha)(1 - f(ha))
if i != a => ∂ ƒ(hi) / ∂ ha = -f(hi)f(ha)
```

https://www.python-course.eu/softmax.php

https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/

https://cs231n.github.io/neural-networks-case-study/

### Training Neural Networks

The reason why neural network doesn't train as planned:

- Architecture can be poorly chosen
- Data can be noisy
- The model could be taking years to run and we need it to run faster

#### Model Training Optimization

- **Training, Validation, Test** <small>(P2-M1-L1-C7, P2-M4-L2-C20)</small>

  - Separate training data, validation data and test data
  - Never use Testing set for training model <small>(P2-M1-L1)</small>
  - Complexity (Find the best neurons' architect) (Under-fitting, Over-fitting) <small>(P7-M1-L3-C5)</small>
  - Early Termination (train until increasing errors of test set) <small>(P7-M1-L3-C5)</small>

- **K Fold Cross Validation** <small>(P2-M1-L1-C8)</small>

  > to break our data in to K buckets. then we just train our model K times, each times using a different bucket as our testing set and the remaining points as out training set.
  >
  > Then we average the results the get final model.

- **Regularization** <small>(P7-M1-L3-C7, P2-M5-L1-C9)</small>

  > *"The whole problem with A.I. is that bad models are so certain of themselves, and good models so full of doubts" Bertrand Russell*
  >
  > Keeping the weights smaller to prevent model from being certain.
  >
  > change error function. add L1 or L2.
  > 
  > L1: good for feature selection.
  > L2: normally better for training models.

- **Dropout** <small>(P7-M1-L3, P2-M5-L1-C11,C12)</small>

  > turn off some nodes in each epoch

- **Local Minimum** <small>(P7-M1-L3)</small>

  > **Random Restart**: To run from local minimum and find total minimum
  >
  > **Momentum**: Use average of previous gradients to push from local minimum <small>(P2-M4-L2-C23, P7-M1-L3-C15)</small>

- **Other Activation Function** <small>(P7-M1-L3)</small>

  > Rectified Linear Unit

- **Under shooting** <small>(P2-M2-L1)</small>

  > The network trains very very slowly but it does tend to make progress.

- **Noise Versus Signal** <small>(P2-M2-L1)</small>

  > To eliminate the noise

- **Numerical Stability** <small>(P2-M4-L2-C19,C23)</small>

  > **Normalized Inputs And Initial Weights**:
  >   1. Inputs with zero mean and small equal variance
  >   2. Initial weights randomly with zero mean and small equal variance

- **Learning rate** in SGD <small>(P2-M4-L2)</small>

  > **Learning rate decay**: Making learning rate smaller over the steps. <small>(P2-M4-L2-C23)</small>
  >
  > **Learning rate tunning**: Use low learning rate. <small>(P2-M4-L2-C24)</small>

#### Some Optimizers

- **Stochastic Gradient Descent (SGD)** <small>(P2-M4-L2)</small>

  > SGD: taking different small subsets of data, run them through neural network, calculate the gradient of error function, then move one step in that direction in each epoch. <small>(P2-M4-L2-C22)</small>
  > ADAGRAD: it uses momentum and learning rate decay <small>(P2-M4-L2-C24)</small>
  > Mini-batch <small>(P2-M4-L2-C25)</small>
  > This is Stochastic Gradient Descent. It uses the following parameters: <small>(P7-M1-L5-C5)</small>
  > - Learning rate.
  > - Momentum (This takes the weighted average of the previous steps, in order to get a bit of momentum and go over bumps, as a way to not get stuck in local minima).
  > - Nesterov Momentum (This slows down the gradient when it's close to the solution).
  >
  > BGD: taking all of data to train our model in each epoch.

- **Adam** <small>(P7-M1-L5-C5)</small>

  > Adam (Adaptive Moment Estimation) uses a more complicated exponential decay that consists of not just considering the average (first moment), but also the variance (second moment) of the previous steps.

- **RMSProp** <small>(P7-M1-L5-C5)</small>

  > RMSProp (RMS stands for Root Mean Squared Error) decreases the learning rate by dividing it by an exponentially decaying average of squared gradients.

## Convolutional Neural Networks

- **Convolution** <small>(P2-M5-L2-C4,C11)</small>

  > Sharing the parameters and weights to all input data
  >
  > **Stride**: the count of moving conv-net in input layer
  >
  > **Padding**: the padding of conv-net from the border of input layer.
  >
  > - Same padding: the conv-net goes on input leayer without any extrusion
  >   > `out_height = ceil(in_height / strides[1])` <br>
  >   > `out_width = ceil(in_width / strides[2])`
  > - Valid padding: the conv-net goes on input leayer with extrusion
  >   > `out_height = ceil((in_height - filter_height + 1) / strides[1])` <br>
  >   > `out_width = ceil((in_width - filter_width + 1) / strides[2])`

- **Pooling** <small>(P2-M5-L2-C18-C28)</small>

  > Combining all conv-net result neighborhood togather. (consider pooling size and pooling strides)
  > - Maximum
  > - Average
  >
  > For a pooling layer the output depth is the same as the input depth. Additionally, the pooling operation is applied individually for each depth slice.
  >
  > Recently, pooling layers have fallen out of favor. Some reasons are: <small>(P2-M5-L2-C21)</small>
  > - Recent datasets are so big and complex we're more concerned about underfitting.
  > - Dropout is a much better regularizer.
  > - Pooling results in a loss of information. Think about the max pooling operation as an example. We only keep the largest of n numbers, thereby disregarding n-1 numbers completely.

- **1x1 Convolutions** <small>(P2-M5-L2-C28)</small>

  > Is a very inexpensive way to make your models deeper and have more parameters without completely changeing their structure.

- **Inception** <small>(P2-M5-L2-C29)</small>

  > Using composition of multi-way convolution instead of a single convolution.

These are the resources we recommend in particular:

  - Andrej Karpathy's [CS231n Stanford course](http://cs231n.github.io/) on Convolutional Neural Networks.
  - Michael Nielsen's [free book](http://neuralnetworksanddeeplearning.com/) on Deep Learning.
  - Goodfellow, Bengio, and Courville's more advanced [free book](http://deeplearningbook.org/) on Deep Learning.

## Recurrent Neural Networks