# Practical 3: Convolutional Neural Networks
## 1. Introduction
In Practical 1 and Practical 2 you studied fully-connected neural networks and how to implement them using *numpy* routines and Google's open-source Deep Learning framework called [TensorFlow](https://www.TensorFlow.org/). In this practical, you will go further and we will work with [Convolutional Neural Networks (CNNs)](http://neuralnetworksanddeeplearning.com/chap6.html) which are considered to be a more natural choice for images than fully-connected neural networks.

In Practical 3, the main goal is to get you familiar with CNNs and how you can use them. You will develop basic CNN and extend it to a siamese variant. We will also use [VGG-16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) model for transfer learning task on CIFAR10 dataset.

As in Practical 2, you will need to write a report about your experiments and results. Sections **Prerequisites**, **Tutorials** and **Troubleshooting** are almost the same as in Practical 2 and kept here for convenience purposes.

## 2. Prerequisites
In this practical, we recommend to use GPU to train CNNs. You can use your own machine or SURF Sara to do it. The instruction of how to install TensorFlow is below.

There are two main methods how you can install TensorFlow.

1. **Pre-built Docker container with TensorFlow**:

  To install TensorFlow using [Docker](https://www.docker.com/) follow these [instructions](https://www.TensorFlow.org/versions/r0.11/get_started/os_setup.html#docker-installation).

	***Note:*** If you are on a Windows machine, this method is your only option due to lack of native TensorFlow support.

2. **Install TensorFlow on your computer (Linux or Mac OS X only)**:

	 Follow the instructions to [download and setup TensorFlow](https://www.TensorFlow.org/versions/master/get_started/os_setup.html#download-and-setup). Choose one of the four ways to install:

    - ***Pip*** : Install TensorFlow directly on your computer. You need to have Python 2.7 and pip installed; and this may impact other Python packages that you may have.
    - ***Virtualenv***: Install TensorFlow in an isolated (virtual) Python environment. You need to have Python 2.7 and virtualenv installed; this will not affect Python packages in any other environment.
    - ***Anaconda*** : Install TensorFlow using Anaconda's package manager called 'conda' that has its own environment system similar to Virtualenv.
    - ***Docker***: Run TensorFlow in an isolated Docker container (virtual machine) on your computer. You need to have Vagrant, Docker and virtualization software like VirtualBox installed; this will keep TensorFlow completely isolated from the rest of your computer, but may require more memory to run.

As debugging code on the server like SURF Sara is not an easy task so it is highly recommended to have TensorFlow on your own machine. At least CPU version.

## 3. Tutorials

- [Stanford's Deep Learning in NLP (CS224d) presentation about TensorFlow](https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf).

- These three tutorials are very important to help you to get through the basics of TensorFlow: [MNIST for ML Beginners](https://www.TensorFlow.org/versions/master/tutorials/index.html), [Deep MNIST for Experts](https://www.TensorFlow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts) and [TensorFlow Mechanics 101](https://www.TensorFlow.org/versions/master/tutorials/mnist/tf/index.html#TensorFlow-mechanics-101).  

- Other useful links: [Tutorials](https://www.TensorFlow.org/versions/r0.11/tutorials/index.html), [How-Tos](https://www.TensorFlow.org/versions/master/how_tos/index.html), [Resources](https://www.TensorFlow.org/versions/master/resources/index.html), [Source code](https://github.com/TensorFlow/TensorFlow), [Stack Overflow](https://stackoverflow.com/questions/tagged/TensorFlow).

You can easily find other TensorFlow tutorials and examples by yourself.

## 4. Troubleshooting
If you have any questions about TensorFlow, first, try to find the answer by yourself using the following resources: [TensorFlow FAQ](https://www.TensorFlow.org/versions/r0.11/resources/faq.html), [TensorFlow FAQ on StackOverflow](http://stackoverflow.com/questions/tagged/TensorFlow?sort=frequent), [TensorFlow Google Groups](https://groups.google.com/a/TensorFlow.org/forum/#!forum/discuss). If you still have questions ask them during practical sessions or in [Piazza](https://piazza.com/class/iuxuidh437j3ed) so other students can also see the question.

If you are using SURF Sara check the [SURF Sara tutorial]("http://uvadlc.github.io/lectures/surfsara-slides.pdf") and [Practical-0](https://github.com/uvadlc/uvadlc_practicals_2016/tree/master/practical_0). Another useful resources are [SURF Sara Interactive Usage](https://userinfo.surfsara.nl/systems/cartesius/usage/interactive-usage) and [SURF Sara Program Development](https://userinfo.surfsara.nl/systems/cartesius/usage/Program-development).

## 5. Assignment

### Overview


We have provided to you several files:
- ***cifar10_utils.py*** : This file contains utility functions that you can use to read CIFAR10 data. This file is a combination of the [similar file from Practical 1](https://github.com/uvadlc/uvadlc_practicals_2016/blob/master/practical_1/uva_code/cifar10_utils.py) and [TensorFlow wrapper over MNIST dataset](https://github.com/TensorFlow/TensorFlow/blob/master/TensorFlow/contrib/learn/python/learn/datasets/mnist.py). Read through this file to get familiar with the interface of the **Dataset** class. The main goal of this class is to sample new batches, so you don't need to worry about it anymore. Unlike Practical 1, in this assignment you will use [one-hot encoding of labels ](https://en.wikipedia.org/wiki/One-hot).  

  **You don't need to change or implement anything in this file.**

  Usage examples:
    - Prepare CIFAR10 data:

    ```python
    import cifar10_utils
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar10-10-batches-py')
    ```
    - Get a new batch with the size of batch_size from the train set:

    ```python
    x, y = cifar10.train.next_batch(batch_size)
    ```
    - Get test images and labels:

    ```python
    x, y = cifar10.test.images, cifar10.test.labels
    ```
    *Note*: `x`, in this practical, is in matrix form rather than vector as compared to previous practicals.

- ***cifar10_siamese_utils.py*** : This file contains utility functions that you can use to read CIFAR10 data. Read through this file to get familiar with the interface of the `Dataset` class. The main goal of this class is to sample new batches for the siamese architecture. The class `Dataset` is very similar to that of *cifar10_utils* except the `next_batch()` method. However, this time you are going to implement two auxiliary methods which will be utilized during training siamese networks.

  **You need to implement the following methods in this file.**
  1. Data sampling
      ```python
      def next_batch(self, batch_size, fraction_same=0.1):
        """
        Returns the next `batch_size` examples from this data set. A batch consist of image pairs and a label.

        Args:
          batch_size: Batch size.
          fraction_same: float in range [0,1], defines the fraction
                            of genuine pairs in the batch

        Returns:
          x1: 4D numpy array of shape [batch_size, 32, 32, 3]
          x2: 4D numpy array of shape [batch_size, 32, 32, 3]
          labels: numpy array of shape [batch_size]                
        """  
        ```
    This is a data generation utility as in ***cifar10_utils.py***. It's goal is to fetch you a minibatch from the training set to be fed into your model. We provide you some explanation how you can to do sampling of minibatch in  ***cifar10_siamese_utils.py*** file, however, you will do a small research to come up with alternative (possibly better) methods. The output of this methods is three *numpy* arrays with 4, 4 and 1 dimensions respectively. More details can be found in the file documentation.
  2. Data creation
    ```python
    def create_dataset(source='Train', num_tuples=100, batch_size = 32, fraction_same=0.1):
        """
        Creates a list of validation tuples. A tuple consist of image pairs and a label.
        A tuple is basically a minibatch to be used in validation.

        Args:
          source: Where to sample from train or test set.
          num_tuples: Number of tuples to be used in the validation
          batch_size: Batch size.
          fraction_same: float in range [0,1], defines the fraction
                            of genuine pairs in the batch

        Returns:
          dset: A list of tuples of length num_tuples.
                Each tuple (minibatch) is of shape [batch_size, 32, 32, 3]                  
        """
    ```
    Since abovementioned sampling leads to dataset expansion due to a number of possible outcomes of pairings `(x1, x2)`, we need a fixed evaluation (i.e. test/val) set. This set is a subset of all possible pairings - with the weak assumption that our training and evaluation sets (somehow) do not overlap in terms of samples. You can hence utilize the `next_batch` method you just implemented to realize this method.

  Usage examples:
    - Prepare CIFAR10 data:

    ```python

    from cifar10_siamese_utils import get_cifar10 as get_cifar_10_siamese
    cifar10 = get_cifar_10_siamese('cifar10/cifar10-10-batches-py')
    ```
    - Get a new batch with the size of batch_size from the train set:

    ```python
    x1, x2, y = cifar10.train.next_batch(batch_size)
    ```
    - Get test images and labels:
    For the siamese network, the size of validation/test set is quite large considering all possible combinations of the image pairs. Hence, we need to limit ourselves to a certain number of tuples, `<x1, x2, y>`. The interface is as follows:
    
    ```python
    from cifar10_siamese_utils import create_dataset
    val_set = create_dataset(source='Test', num_tuples, batch_size, fraction_same)  
    ```

- ***convnet.py*** : This file contains an interface for the `ConvNet` class. There are 4 methods of this class: `__init__(*args)`, `inference(*args)`, `loss(*args)` and `accuracy(*args)`. Implement these methods by strictly following the interfaces given. You are free to add other methods to the class but, please, follow the conventions and append an *underscore* , for example, ***\_private_method***. It is considered as a standard approach to specify a non-public part of the class interface. Check this [link](https://shahriar.svbtle.com/underscores-in-python#single-underscore-before-a-name-eg-code-class_1) for more information.

- ***siamese.py*** : This file contains an interface for the `Siamese` class. There are 3 methods of this class: `__init__(*args)`, `inference(*args)`, `loss(*args)`. Implement these methods by strictly following the interfaces provided. You are again free to add other methods to the class considering the naming conventions.

- ***train_model.py*** : This file contains two main functions. The function `train()` is a function where you need to implement training and testing procedure of the network on CIFAR10 dataset using `ConvNet` class. Similarly, you have `train_siamese()` method which, as it speaks for itself, defines the operations to train/test your siamese model using `Siamese` class. You will need also a few auxiliary functions such as `train_step()` and `feature_extraction()`. Carefully examine the provided code documentation and implement your code where they are asked to be. Finally, carefully go through and get familiar with all possible command line parameters and their possible values for running ***train_model.py***. You are going to implement each of these into your code.

- ***vgg.py*** : This file contains a graph constructor for the convolutional part of VGG-16. Call `load_pretrained_VGG16_pool5` in order to get activations from the 5th pooling layer of the VGG model. ***vgg.py*** will be used in the Task 3 of this assignment.

- ***retrain_vgg.py*** A wrapper for performing task 3 using VGG. This file is mostly similar to ***train_model.py*** and to ***train_mlp.py*** from Practical 2.

- ***report/nips_2016.tex***, ***report/nips_2016.sty*** : The original LaTex  template and style file for writing NIPS paper. You should check them before writing your report. If you are not familiar with LaTex  you can check, for example, [this](https://www.latex-tutorial.com/) or [this](https://www.latex-tutorial.com/tutorials/) tutorials.

- ***report/report_lastname.tex*** : The template for your report in this assignment with the predefined structure of the sections. More information can be found in the Report section of this file. When you are submitting the assignment, please, replace ***lastname*** with your last name.

- ***cifar10/get_cifar10.sh*** : Shell script to download CIFAR10 data as in the Practical 1.

### Task 1: CIFAR10  [40 points]
We are going to build a CNN model using TensorFlow. Unlike previous practicals, you are given a fixed network to work with. This task involves the implementation of a lightweight class, namely `ConvNet() `. In the class `ConvNet()` you need to complete the methods:

- `inference(*args)`
- `loss(*args)`
- `accuracy(*args)`

In addition to these class methods you are going to write your own training function in ***train_model.py*** as stated earlier.  

The architecture of CNN is below:

|Block name   | Elements 		| Kernel Size         | Filter depth  |  Output depth  | Stride  | Misc. Info |
|-----------  |-------   		|:-------------:      | :--------:    | :-------:      | :-----  |------------|
|             | `Convolution`           |   `[5, 5]`          |   `3`         |   `64`         | `[1, 1]`|            |
|`conv1`      | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             | `Max-pool`   		|  `[3, 3]`           | `None`        | `None`         |`[2, 2]` |            |  
|             |                         |                     |               |                |         |            |
|             | `Convolution`           |  `[5, 5]`           |   `64`        |   `64`         | `[1, 1]`|            |
| `conv2`     | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             | `Max-pool`   		|  `[3, 3]`           | `None`        | `None`         |`[2, 2]` |            |
|             |                         |                     |               |                |         |            |
| `flatten`   |   `Flatten`             |                     |               |                |         |            |
|             |                         |                     |               |                |         |            |
|   `fc1`     | `Multiplication`        |  `[dim(conv2), 384]`|   `None`      |    `None`      | `None`  |            |
|             | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             |                         |                     |               |                |         |            |
|   `fc2`     | `Multiplication`        |  `[384, 192]`       |   `None`      |   `None`       |  `None` |            |
|             | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             |                         |                     |               |                |         |            |
|   `fc3`     | `Multiplication`        |  `[192, 10]`        |   `None`      |   `None`       |  `None` |            |
|             | `Softmax`      		|   `None`            | `None`        |  `None`        | `None`  |            |

Given model architecture and hyperparameters, you need to describe it in terms of TensorFlow operations in `ConvNet.inference(*args)`.

`ConvNet.loss(*args)` returns operations to compute cross-entropy loss given softmax activations. You need to define those ops as well.

As for the training function `train()` in ***train_model.py***, you are expected to use the `ConvNet` class to define your graph and write the training loop where you optimize your model parameters using gradient descent variants such as `Adam`. This is pretty much the same way as in Practical 2.

#### Classification: CIFAR10
Following the spirit passed on from previous practicals, you are going to train the model to convergence with a classification objective. Report the performance in accuracy and hyperparameters used to reach it. You should use of TensorBoard for monitoring training and visualizations to be used in the report.

***Hint:*** You should expect a classification accuracy on the test set at least 70% with this plain CNN.

#### Feature extraction
- Use `feature_extraction()` in order to compute features for test samples at layer `fc2`. Use these features to visualize learned space by the help of `t-SNE`. Include the visualization in your report.

  ***Note:*** You can download and use the implementation for `t-SNE` from [here](https://lvdmaaten.github.io/tsne/). You can also use [sklearn implementation of t-SNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

- Train 10 linear one-vs-rest classifiers for each class and report the performances. Can you draw similar conclusions by just looking at the visualization and thinking about separability and how good classes are represented by the model?

  ***Note:*** You can use any linear classifier implementation available as long as you preserve consistency in experiments. See `scikitlearn`, `libsvm`, `liblinear`, `VLFeat` and etc.

- Repeat the above experiments for `flatten` and `fc1`. Discuss the results. What is more suited for classification purposes: `flatten`, `fc1` or `fc2`?

- Can you improve test performance by using regularization techniques? Try L2 weight regularization on the fully-connected layers. You can also try others regularization techniques like dropout and batch normalization. Report your conclusions in your report with experimental support.         

### Task 2: CIFAR10 Siamese [30 points]
Now that you have learned how to implement CNN model in TensorFlow you can take it a bit further and start getting familiar with `weight/parameter sharing` idea. It is an important concept which you might encounter quite often in practice. In this exercise we are going to focus on a particular architecture, referred to as *Siamese*, which is based on parameter sharing idea. This task involves the implementation of another lightweight class, namely `Siamese() `. In class `Siamese()` you need to complete the methods as you did for `ConvNet()`:

- `inference(*args)`
- `loss(*args)`

In addition to these class methods you are going to implement another training function in ***train_model.py***.  

The inference model consists of two channels of convolutional networks whose parameters are tied (=shared) and the following table describes the layers for one these channels. It is more or less the same as the one described in the first task.

|Block name   | Elements 		| Kernel Size         | Filter depth  |  Output depth  | Stride  | Misc. Info |
|-----------  |-------   		|:-------------:      | :--------:    | :-------:      | :-----  |------------|
|             | `Convolution`           |   `[5, 5]`          |   `3`         |   `64`         | `[1, 1]`|            |
|`conv1`      | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             | `Max-pool`   		|  `[3, 3]`           | `None`        | `None`         |`[2, 2]` |            |  
|             |                         |                     |               |                |         |            |
|             | `Convolution`           |  `[5, 5]`           |   `64`        |   `64`         | `[1, 1]`|            |
| `conv2`     | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             | `Max-pool`   		|  `[3, 3]`           | `None`        | `None`         |`[2, 2]` |            |
|             |                         |                     |               |                |         |            |
| `flatten`   |   `Flatten`             |                     |               |                |         |            |
|             |                         |                     |               |                |         |            |
|   `fc1`     | `Multiplication`        |  `[dim(conv2), 384]`|   `None`      |    `None`      | `None`  |            |
|             | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             |                         |                     |               |                |         |            |
|   `fc2`     | `Multiplication`        |  `[384, 192]`       |   `None`      |   `None`       |  `None` |            |
|             | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             |                         |                     |               |                |         |            |
|   `L2-norm` | `L2-normalization`      | `None`              |   `None`      |   `None`       |  `None` |            |


Given model architecture and hyperparameters, all you need to do is to describe it in terms of TensorFlow operations in `Siamese.inference(*args)`. In this method, you should implement single channel as you will build the complete model (a.k.a siamese) in `train_siamese()`.

`Siamese.loss(*args)` returns operations to compute contrastive loss. You are going to define the ops for this particular loss.

In the training function `train_siamese()` in ***train_model.py***, you are expected to use `Siamese` class to define your graph and write the training loop where you optimize your model parameters using gradient descent variants such as `Adam`.

#### Metric Learning
As clear from the definition of contrastive loss, we can see that this objective is not explicitly designed for classification purpose although it is yet a discriminative one. Identify one or two applications of such loss models. Train the model till convergence. Don't forget to monitor validation set loss.

#### Feature extraction
- Use `feature_extraction()` in order to compute features for test samples at layer `L2-norm`. Use the features to visualize learnt space by the help of `t-SNE`. Include the visualization into your report.

- Does the visualisation tell anything regarding the quality of training or about the similarities/dissimilarities between two loss functions?

- Train 10 linear one-vs-rest classifiers for each class and report the performances. Can you draw similar conclusions by just looking at the visualisation and thinking about separability and how good classes are represented by the model? Which loss is superior in the context of these experiments? Provide discussion in your report.

### Task 3: Transfer Learning [30 points]

In this task you will perform transfer learning on the CIFAR10 classification task. To do that you will use the lower layers of a pretrained model as feature representations for your classifier. Transfer learning is the idea that learned low- level representations have commonalities across different data sets. Due to the limited size of the CIFAR10 data set the hope is that feature representation which was learned from a large data set can boost the performance of this task.

Here, you will perform transfer learning using [VGG Net](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) which won the ImageNet competition in 2014. You can download the weights of VGG-16 for Tensorflow from [David Fossards Blog](https://www.cs.toronto.edu/~frossard/post/vgg16/). For your convenience, we have supplied you with a function `vgg.py` which takes an input image and outputs the respective feature activations from the last pooling layer (pool5) of VGG-16.

You will use these features as an input to a fully-connected network of the form

|Block name   | Elements 		| Kernel Size         | Filter depth  |  Output depth  | Stride  | Misc. Info |
|-----------  |-------   		|:-------------:      | :--------:    | :-------:      | :-----  |------------|
|             |                         |                     |               |                |         |            |
| `flatten`   |   `Flatten`             |                     |               |                |         |            |
|             |                         |                     |               |                |         |            |
|   `fc1`     | `Multiplication`        |  `[dim(conv2), 384]`|   `None`      |    `None`      | `None`  |            |
|             | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             |                         |                     |               |                |         |            |
|   `fc2`     | `Multiplication`        |  `[384, 192]`       |   `None`      |   `None`       |  `None` |            |
|             | `ReLU`      		|   `None`            | `None`        |  `None`        | `None`  |            |
|             |                         |                     |               |                |         |            |
|   `fc3`     | `Multiplication`        |  `[192, 10]`        |   `None`      |   `None`       |  `None` |            |
|             | `Softmax`      		|   `None`            | `None`        |  `None`        | `None`  |            |

You will notice that these layers have the exact same form as the fully connected layers in Task 1. Essentially, in this task we are going to replace all convolutional layers from Task 1 with a set of pre-trained layers from VGG-16. Make sure to implement your training procedure for this transfer learning task in ***retrain_vgg.py***.

In all experiments, make sure to report training and test accuracy. Make a screenshot of your graph in TensorBoard to ensure that gradient information is passed on in the way as expected. Always compare your results to the results from Task 1. Put new results in context of previous results in this task.

Unless specified otherwise, use the following parameter in all experiments:

```python
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 10000
OPTIMIZER_DEFAULT = 'ADAM'
```

***Hint:*** Make sure to call `assign_ops` from `vgg.py` after initializing variables.

#### Feature extraction
Use pool5 of VGG-16 as a feature extractor. Only train the fully-connected layers of your network, do not update the VGG parameters. Did the VGG-16 features improve performance over your results in task 1? What reasons do you think lead to your results?

***Hint***: To not update the VGG-16 parameters you could use `tf.stop_gradient()`.

#### Retraining
After your first experiment with transfer learning, we ask you to jointly update the parameters from all layers in your model.
How do your results compare to your previous experiments? Track the progression of parameters using `tf.histogram_summary`. You might need to edit ***vgg.py*** in order to do that.

#### Refining
In this task you will first update only the fully connected layers for a number of iterations while keeping the VGG parameters fixed (as in feature extraction). After those iterations you will continue to jointly update all parameters (as in retraining).

Use `FLAGS.refine_after_k` to define the number of iterations for the feature extraction step, and before the training procedure performs retraining of the VGG parameters.

Run experiments with `refine_after_k = [100, 1000, 2500]`. Is performance dependent on the choice of `refine_after_k`? How? consider also that for the task **Retraining** `refine_after_k = MAX_STEPS_DEFAULT`, and for the task **Refining** `refine_after_k = 0`.

***Hint:*** In order to distinguish between the 2 phases you can make use of `tf.cond()`.

#### Further Improvements
What other factors in this transfer learning task could influence the performance? Name at least two. Test your hypothesis in a short experiment and report your results.

### Report
You should write a small report about your study of CNN models on CIFAR10 dataset using the provided template for NIPS papers. Please, make your report to be self-contained without this README.md file.

The report should contain the following sections:

- **Abstract** : Should contain information about the current task and the summary of the study of the CNN models on CIFAR10 dataset.
- **Task 1** : Should contain all needed information about Task 1 and report of all your experiments for that task.  
- **Task 2** : Should contain all needed information about Task 2 and report of all your experiments for that task.
- **Task 3** : Should contain all needed information about Task 3 and report of all your experiments for that task.
- **Conclusion** : Should contain conclusion of this study.
- **References** : Reference section if needed.

### Submission
Create zip archive with the following structure:

```
lastname_assignment_3.zip
│   report_lastname.pdf
│   convnet.py
|   siamese.py
│   train_model.py
|   retrain_vgg.py
|   vgg.py
```

Replace `lastname` with your last name. After you create the zip file, please send it to **uva.deeplearning@gmail.com ONLY**. Because of the large number of students if you send it to any other email, we cannot guarantee that it will be corrected in time. As the subject of the email use: "Assignment 2: Your Full Name".

The deadline for the assignment is the **7th of December, 23:59**.
