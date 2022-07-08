progress: P02 - M04 - L02 - C27

progress: P07 - M01 - L04 - C23

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
  > change error function. add L1 and L2

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

- **Stochastic Gradient Descent (SGD)** <small>(P2-M4-L2)</small>

  > SGD: taking different small subsets of data, run them through neural network, calculate the gradient of error function, then move one step in that direction in each epoch. <small>(P2-M4-L2-C22)</small>
  > ADAGRAD: it uses momentum and learning rate decay <small>(P2-M4-L2-C24)</small>
  > Mini-batch <small>(P2-M4-L2-C25)</small>
  >
  > BGD: taking all of data to train our model in each epoch.

- **Learning rate** in SGD <small>(P2-M4-L2)</small>

  > **Learning rate decay**: Making learning rate smaller over the steps. <small>(P2-M4-L2-C23)</small>
  >
  > **Learning rate tunning**: Use low learning rate. <small>(P2-M4-L2-C24)</small>

## Convolutional Neural Networks