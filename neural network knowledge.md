progress: P2 - M5 - L4 - C3

progress: P8 - M1 - L4 - C3 <<<

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

## Training Neural Networks

The reason why neural network doesn't train as planned:

- Architecture can be poorly chosen
- Data can be noisy
- The model could be taking years to run and we need it to run faster

### Model Training Optimization

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

- **Numerical Stability** <small>(P2-M4-L2-C19,C23, P8-M1-L3)</small>

  > **Normalized Inputs And Initial Weights**:
  >   1. Inputs with zero mean and small equal variance
  >   2. Initial weights randomly with zero mean and small equal variance
  >
  > **Uniform Distribution**: <small>(P8-M1-L3-C4)</small>
  >   1. Use random weights from -0.1 to +0.1.
  >   2. Use general rule weights -1/√n to +1/√n (n is number of units).
  >   *the result is same.*
  >
  > **Normal Distribution**: <small>(P8-M1-L3-C5)</small>
  >
  >   This is a little bit better than Uniform distribution.
  >   Use truncated normal distibution.
  >
  > Additional Material
  > New techniques for dealing with weights are discovered every few years. We've provided the most popular papers in this field over the years.
  >   - [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
  >   - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852v1.pdf)
  >   - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v2.pdf)

- **Learning rate** in SGD <small>(P2-M4-L2)</small>

  > **Learning rate decay**: Making learning rate smaller over the steps. <small>(P2-M4-L2-C23)</small>
  >
  > **Learning rate tunning**: Use low learning rate. <small>(P2-M4-L2-C24)</small>

### Some Optimizers

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

Part 1: <small>P2-M5-L2-C35</small>

  - Andrej Karpathy's [CS231n Stanford course](http://cs231n.github.io/) on Convolutional Neural Networks.
  - Michael Nielsen's [free book](http://neuralnetworksanddeeplearning.com/) on Deep Learning.
  - Goodfellow, Bengio, and Courville's more advanced [free book](http://deeplearningbook.org/) on Deep Learning.

Part 2: <small>P8-M1-L4-C2</small>

  - Read about the [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) model.
    - Why train an A.I. to talk, when you can train it to sing ;)? In April 2017, researchers used a variant of the WaveNet model to generate songs. The original paper and demo can be found [here](http://www.creativeai.net/posts/W2C3baXvf2yJSLbY6/a-neural-parametric-singing-synthesizer).
  - Learn about CNNs [for text classification](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).
    - You might like to sign up for the author's [Deep Learning Newsletter](https://www.getrevue.co/profile/wildml)!
  - Read about Facebook's [novel CNN approach](https://code.facebook.com/posts/1978007565818999/a-novel-approach-to-neural-machine-translation/) for language translation that achieves state-of-the-art accuracy at nine times the speed of RNN models.
  - Play [Atari games](https://deepmind.com/research/dqn/) with a CNN and reinforcement learning. You can [download](https://sites.google.com/a/deepmind.com/dqn/) the code that comes with this paper.
    - If you would like to play around with some beginner code (for deep reinforcement learning), you're encouraged to check out Andrej Karpathy's [post](http://karpathy.github.io/2016/05/31/rl/).
  - Play [pictionary](https://quickdraw.withgoogle.com/#) with a CNN!
    - Also check out all of the other cool implementations on the [A.I. Experiments](https://aiexperiments.withgoogle.com/) website. Be sure not to miss [AutoDraw](https://www.autodraw.com/)!
  - Read more about [AlphaGo](https://deepmind.com/research/alphago/).
    - Check out [this article](https://www.technologyreview.com/s/604273/finding-solace-in-defeat-by-artificial-intelligence/?set=604287), which asks the question: *If mastering Go “requires human intuition,” what is it like to have a piece of one’s humanity challenged?*
  - Check out these *really cool* videos with drones that are powered by CNNs.
    - Here's an interview with a startup - [Intelligent Flying Machines (IFM)](https://www.youtube.com/watch?v=AMDiR61f86Y).
    - Outdoor autonomous navigation is typically accomplished through the use of the [global positioning system (GPS)](http://www.droneomega.com/gps-drone-navigation-works/), but here's a demo with a CNN-powered [autonomous drone](https://www.youtube.com/watch?v=wSFYOw4VIYY).
  - If you're excited about using CNNs in self-driving cars, you're encouraged to check out:
    - our [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013), where we classify signs in the [German Traffic Sign](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset in [this project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project).
    - our [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009), where we classify house numbers from the [Street View House Numbers](http://ufldl.stanford.edu/housenumbers/) dataset in [this project](https://github.com/udacity/machine-learning/tree/master/projects/digit_recognition).
    - this [series of blog posts](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/) that details how to train a CNN in Python to produce a self-driving A.I. to play Grand Theft Auto V.
  - Check out some additional applications not mentioned in the video.
    - Some of the world's most famous paintings have been [turned into 3D](http://www.businessinsider.com/3d-printed-works-of-art-for-the-blind-2016-1) for the visually impaired. Although the article does not mention *how* this was done, we note that it is possible to use a CNN to [predict depth](https://www.cs.nyu.edu/~deigen/depth/) from a single image.
    - Check out [this research](https://research.googleblog.com/2017/03/assisting-pathologists-in-detecting.html) that uses CNNs to localize breast cancer.
    - CNNs are used to [save endangered species](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)!
    - An app called [FaceApp](http://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/) uses a CNN to make you smile in a picture or change genders.

## Recurrent Neural Networks