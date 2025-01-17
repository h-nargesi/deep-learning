# Neural Networks

Do the main projects in P1/P2

progress: P3 - M2 - L1 - C1 **(TO-READ)**

progress: P8 - M1 - L4

passed: P8 - M1 - L5,6,7
passed: P8 - M2

## Introduction

- **weights change**

```mathematica
Δwi = -(∂Error / ∂wi)
```

- **output layer**: η = learning rate

```mathematica
e = (y - ŷ)
δ = e * ƒ’(h)
Δwi = η * δ * xi
```

- **hidden layer**: j = hidden layer, k = output layer

```mathematica
ej = (∑k δk * wjk)
δj = ej * ƒ’(hj)
Δwi = η * δ * xi
```

### Activation Functions

- **Sigmoid**

```mathematica
f(x) = 1 / (1 + e^-x)
```

- **Soft-max**

```mathematica
f(i) = e^xi / (e^x1 + e^x2 + ... + e^xn)
```

if you'd like more information on activation functions, check out this [website](http://cs231n.github.io/neural-networks-1/#actfun). <small>(P8-M1-L4-C4)</small>

### Error Functions

- **cross-entropy loss**

```mathematica
Error = ∑ (yi * log(ŷi) + (1-yi) * log(1-ŷi))
∂Error / ∂ŷ = -(y - ŷ)
```

- **errorest**: M = mean square error

```mathematica
Error = 1/M ∑M ½ * (yi - ŷi)²
∂Error / ∂ŷ = - 1/M ∑M (yi - ŷi)
```

- **soft-max**

```mathematica
f(h) = e^h / ∑ e^hi
∂ ƒ(hi) / ∂ ha = 
if i = a => ∂ ƒ(hi) / ∂ ha = f(ha)(1 - f(ha))
if i != a => ∂ ƒ(hi) / ∂ ha = -f(hi)f(ha)
```

more information:

- [python-course.eu](https://www.python-course.eu/softmax.php)
- [adeveloperdiary.com](https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/)
- [cs231n.github.io](https://cs231n.github.io/neural-networks-case-study/)
- There are many different [loss functions](https://keras.io/losses/) in Keras. <small>(P8-M1-L4-C5)</small>

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

	to break our data in to K buckets. then we just train our model K times, each times using a different bucket as our testing set and the remaining points as out training set.

	Then we average the results the get final model.

- **Regularization** <small>(P7-M1-L3-C7, P2-M5-L1-C9)</small>

	*"The whole problem with A.I. is that bad models are so certain of themselves, and good models so full of doubts" Bertrand Russell*

	Keeping the weights smaller to prevent model from being certain.

	change error function. add L1 or L2.

	- L1: good for feature selection.
	- L2: normally better for training models.

- **Dropout** <small>(P7-M1-L3, P2-M5-L1-C11,C12)</small>

	turn off some nodes in each epoch

	check out the [first research paper](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) to propose dropout as a technique for overfitting. <small>(P8-M1-L4-C4)</small>

- **Local Minimum** <small>(P7-M1-L3)</small>

	**Random Restart**: To run from local minimum and find total minimum

	**Momentum**: Use average of previous gradients to push from local minimum <small>(P2-M4-L2-C23, P7-M1-L3-C15)</small>

- **Other Activation Function** <small>(P7-M1-L3)</small>

	Rectified Linear Unit

- **Under shooting** <small>(P2-M2-L1)</small>

	The network trains very very slowly but it does tend to make progress.

- **Noise Versus Signal** <small>(P2-M2-L1)</small>

	To eliminate the noise

- **Numerical Stability** <small>(P2-M4-L2-C19,C23, P8-M1-L3)</small>

	**Normalized Inputs And Initial Weights**:

		1. Inputs with zero mean and small equal variance
		2. Initial weights randomly with zero mean and small equal variance

	**Uniform Distribution**: <small>(P8-M1-L3-C4)</small>

		1. Use random weights from -0.1 to +0.1.
		2. Use general rule weights -1/√n to +1/√n (n is number of units).
		*the result is same.*

	**Normal Distribution**: <small>(P8-M1-L3-C5)</small>

		- This is a little bit better than Uniform distribution.
		- Use truncated normal distibution.

	Additional Material
	New techniques for dealing with weights are discovered every few years. We've provided the most popular papers in this field over the years.

	- [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
	- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852v1.pdf)
	- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v2.pdf)

- **Learning rate** in SGD <small>(P2-M4-L2, P3-M1-L3)</small>

	**Learning rate decay**: Making learning rate smaller over the steps. <small>(P2-M4-L2-C23)</small>

	**Learning rate tunning**: Use low learning rate. <small>(P2-M4-L2-C24)</small>

### Some Optimizers

- **BGD**: taking all of data to train our model in each epoch.

- **Stochastic Gradient Descent (SGD)** <small>(P2-M4-L2)</small>

	SGD: taking different small subsets of data, run them through neural network, calculate the gradient of error function, then move one step in that direction in each epoch. <small>(P2-M4-L2-C22)</small>

	ADAGRAD: it uses momentum and learning rate decay <small>(P2-M4-L2-C24)</small>
	[AdagradOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)

	Mini-batch <small>(P2-M4-L2-C25,)</small> and it's relation with Learning Rate. <small>(P3-M1-L3-C5)</small>

	This is Stochastic Gradient Descent. It uses the following parameters: <small>(P7-M1-L5-C5)</small>

	- Learning rate.
	- Momentum (This takes the weighted average of the previous steps, in order to get a bit of momentum and go over bumps, as a way to not get stuck in local minima).
	- Nesterov Momentum (This slows down the gradient when it's close to the solution).

- **Adam** <small>(P7-M1-L5-C5)</small>

	Adam (Adaptive Moment Estimation) uses a more complicated exponential decay that consists of not just considering the average (first moment), but also the variance (second moment) of the previous steps.

	[AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)

- **RMSProp** <small>(P7-M1-L5-C5)</small>

	RMSProp (RMS stands for Root Mean Squared Error) decreases the learning rate by dividing it by an exponentially decaying average of squared gradients.

Check out the [list of available optimizers](https://keras.io/optimizers/) in Keras. The optimizer is specified when you compile the model (in Step 7 of the notebook). <small>(P8-M1-L4-C5)</small> **(TOREAD)**

## Convolutional Neural Networks

<small>(P2-M5-L2, P8-M1-L4)</small>

- **Convolution** <small>(P2-M5-L2-C4,C11, P8-M1-L4-C9..C12)</small>

	Sharing the parameters and weights to all input data

	**Stride**: the count of moving conv-net in input layer

	**Padding**: the padding of conv-net from the border of input layer.

	- Same padding: the conv-net goes on input leayer without any extrusion <br>
			`out_height = ceil(in_height / strides[1])` <br>
			`out_width = ceil(in_width / strides[2])`
	- Valid padding: the conv-net goes on input leayer with extrusion <br>
			`out_height = ceil((in_height - filter_height + 1) / strides[1])` <br>
			`out_width = ceil((in_width - filter_width + 1) / strides[2])`

- **Pooling** <small>(P2-M5-L2-C18-C28, P8-M1-L4-C15)</small>

	Combining all conv-net result neighborhood togather. (consider pooling size and pooling strides)

	- Maximum
	- Average

	For a pooling layer the output depth is the same as the input depth. Additionally, the pooling operation is applied individually for each depth slice.

	Recently, pooling layers have fallen out of favor. Some reasons are: <small>(P2-M5-L2-C21)</small>

	- Recent datasets are so big and complex we're more concerned about underfitting.
	- Dropout is a much better regularizer.
	- Pooling results in a loss of information. Think about the max pooling operation as an example. We only keep the largest of n numbers, thereby disregarding n-1 numbers completely.

	Check out the Keras [documentation](https://keras.io/layers/pooling/) on different types of pooling layers! <small>(P8-M1-L4-C15)</small>

- **1x1 Convolutions** <small>(P2-M5-L2-C28)</small>

	Is a very inexpensive way to make your models deeper and have more parameters without completely changeing their structure.

- **Inception** <small>(P2-M5-L2-C29)</small>

	Using composition of multi-way convolution instead of a single convolution **(TOREAD)**

- **Practical Tips**

	- Resize all of our images to input size. <small>(P8-M1-L4-C17)</small>
	- Kernel size: between 2 and 5. <small>(P8-M1-L4-C17)</small>
	- Make your images augmented for training your model, and then use `fit_generator` method to train them.<small>(P8-M1-L4-C20)</small>

### Transfer Learning

<small>(P8-M1-L4-C25)</small>

Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set.
includes removeing the final layers of the networrk that are very specific to the training data set while keeping the ealier layers. Then we can add one or two more and train only the final layers.

That depending on both:

- the size of the new data set, and
- the similarity of the new data set to the original data set

the approach for using transfer learning will be different. There are four main cases:

1. new data set is small, new data is similar to original training data **(End of ConvNet)**
2. new data set is small, new data is different from original training data **(Start of ConvNet)**
3. new data set is large, new data is similar to original training data **(Fine-tune)**
4. new data set is large, new data is different from original training data **(Fine-tune or Retrain)**

### Kebras (Tensorflow)

Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.

There are many callbacks (such as ModelCheckpoint) that you can use to monitor your model during the training process. If you'd like, check out the [details](https://keras.io/callbacks/#modelcheckpoint) here. You're encouraged to begin with learning more about the EarlyStopping callback. If you'd like to see another code example of ModelCheckpoint, check out [this blog](http://machinelearningmastery.com/check-point-deep-learning-models-keras/).

## Recurrent Neural Networks

- Cell Types

	- **LSTM (Long Short Term Memory)** <small>(P3-M1-L1-C2)</small>
	- **GRU (Gated Recurrent Unit)**
	- **Vanilla RNN**

- **Sequence Batching** <small>(P3-M1-L1-C4,6)</small>

## Hyperparameters

<small>(P3-M1-L3)</small>

1. Optimizer Hyperparameters
	- Learning Rate
	- Minibatch
	- Epochs (Validation Monitor <small>(P3-M1-L3-C6)</small>)
2. Model Hyperparameters
	- Number of Hidden Units Layers
	- Model Specific Hyperparameters

### Monitoring Validation Loss vs. Training Loss

The most important quantity to keep track of is the difference between your training loss (printed during training) and the validation loss (printed once in a while when the RNN is run on the validation data (by default every 1000 iterations)). In particular:

- If your training loss is much lower than validation loss then this means the network might be **overfitting**. Solutions to this are to decrease your network size, or to increase dropout. For example you could try dropout of 0.5 and so on.
- If your training/validation loss are about equal then your model is **underfitting**. Increase the size of your model (either number of layers or the raw number of neurons per layer)

### Approximate number of parameters

The two most important parameters that control the model are `lstm_size` and `num_layers`. I would advise that you always use `num_layers` of either 2/3. The `lstm_size` can be adjusted based on how much data you have. The two important quantities to keep track of here are:

- The number of parameters in your model. This is printed when you start training.
- The size of your dataset. 1MB file is approximately 1 million characters.

These two should be about the same order of magnitude. It's a little tricky to tell. Here are some examples:

- I have a 100MB dataset and I'm using the default parameter settings (which currently print 150K parameters). My data size is significantly larger (100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can comfortably afford to make `lstm_size` larger.
- I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm carefully monitoring my validation loss. If it's larger than my training loss then I may want to try to increase dropout a bit and see if that helps the validation loss.

### Best models strategy

The winning strategy to obtaining very good models (if you have the compute time) is to always err on making the network larger (as large as you're willing to wait for it to compute) and then try different dropout values (between 0,1). Whatever model has the best validation performance (the loss, written in the checkpoint filename, low is good) is the one you should use in the end.

It is very common in deep learning to run many different models with many different hyperparameter settings, and in the end take whatever checkpoint gave the best validation performance.

By the way, the size of your training and validation splits are also parameters. Make sure you have a decent amount of data in your validation set or otherwise the validation performance will be noisy and not very informative.

## Genetic Algorithm

- leayers
	- leayers counts
	- leayers types
	- ...

- initial weights and bias

- training process
	- learning rate
	- batch size
	- epoch
	- ...