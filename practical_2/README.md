# Practical 2: Introduction to TensorFlow

## 1. Introduction
In Practical 1  we studied the backpropagation algorithm and we implemented a multinomial logistic regression classifier with an extension to a multi-layer neural network using *numpy* routines. In Practical 2, one of our goals is to get you familiar with Google's open-source Deep Learning framework called [TensorFlow](https://www.TensorFlow.org/). In general, TensorFlow is a library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (i.e. tensors) communicated between them. As a TensorFlow user, you define the computational architecture of your predictive model, combine that with your objective function, and just add data - TensorFlow handles computing the derivatives for you.

The second goal of this practical is to investigate the influence of different parameters of model and training procedure. You will build your first neural network using TensorFlow and study the impacts of weight initialization, regularization, activation functions, optimizers and neural network architecture. You will need to write a small report about your findings during this study. As consequence, by the end of this practical, you will know the basic TensorFlow interfaces and routines which should be enough to start using your own models for either research purposes or for creating applications which can be deployed to many platforms like [mobile](https://www.TensorFlow.org/mobile.html).

## 2. Prerequisites
As this practical doesn't need heavy GPU computations you can install TensorFlow on your own machine. We will need just CPU version of TensorFlow.

There are two main methods how you can install TensorFlow.

1. **Pre-built Docker container with TensorFlow**:

  To install TensorFlow using [Docker](https://www.docker.com/) follow these [instructions](https://www.TensorFlow.org/versions/r0.11/get_started/os_setup.html#docker-installation).

	***Note***: If you are on a Windows machine, this method is your only option due to lack of native TensorFlow support.

2. **Install TensorFlow on your computer (Linux or Mac OS X only)**:

	 Follow the instructions to [download and setup TensorFlow](https://www.TensorFlow.org/versions/master/get_started/os_setup.html#download-and-setup). Choose one of the four ways to install:

    - ***Pip*** : Install TensorFlow directly on your computer. You need to have Python 2.7 and pip installed; and this may impact other Python packages that you may have.
    - ***Virtualenv***: Install TensorFlow in an isolated (virtual) Python environment. You need to have Python 2.7 and virtualenv installed; this will not affect Python packages in any other environment.
    - ***Anaconda*** : Install TensorFlow using Anaconda's package manager called 'conda' that has its own environment system similar to Virtualenv.
    - ***Docker***: Run TensorFlow in an isolated Docker container (virtual machine) on your computer. You need to have Vagrant, Docker and virtualization software like VirtualBox installed; this will keep TensorFlow completely isolated from the rest of your computer, but may require more memory to run.

As debugging code on the server like SURF Sara is not an easy task so it is highly recommended to have TensorFlow on your own machine.

## 3. Tutorials

During a practical session on the 10th of November, we will go through the basic tutorial introducing TensorFlow. In general, it is [Stanford's Deep Learning in NLP (CS224d) presentation about TensorFlow](https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf).

These three tutorials are also very important to help you to get through the basics of TensorFlow: [MNIST for ML Beginners](https://www.TensorFlow.org/versions/master/tutorials/index.html), [Deep MNIST for Experts](https://www.TensorFlow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts) and [TensorFlow Mechanics 101](https://www.TensorFlow.org/versions/master/tutorials/mnist/tf/index.html#TensorFlow-mechanics-101).  

Other useful links: [Tutorials](https://www.TensorFlow.org/versions/r0.11/tutorials/index.html), [How-Tos](https://www.TensorFlow.org/versions/master/how_tos/index.html), [Resources](https://www.TensorFlow.org/versions/master/resources/index.html), [Source code](https://github.com/TensorFlow/TensorFlow), [Stack Overflow](https://stackoverflow.com/questions/tagged/TensorFlow).

You can easily find other TensorFlow tutorials and examples by yourself.

## 4. Troubleshooting
If you have any questions about TensorFlow, first, try to find the answer by yourself using the following resources: [TensorFlow FAQ](https://www.TensorFlow.org/versions/r0.11/resources/faq.html), [TensorFlow FAQ on StackOverflow](http://stackoverflow.com/questions/tagged/TensorFlow?sort=frequent), [TensorFlow Google Groups](https://groups.google.com/a/TensorFlow.org/forum/#!forum/discuss). If you still have questions ask them during practical sessions or in [Piazza](https://piazza.com/class/iuxuidh437j3ed) so other students can also see the question.

If you are using SURF Sara check the [SURF Sara tutorial]("http://uvadlc.github.io/lectures/surfsara-slides.pdf") and [Practical-0](https://github.com/uvadlc/uvadlc_practicals_2016/tree/master/practical_0). Another useful resources are [SURF Sara Interactive Usage](https://userinfo.surfsara.nl/systems/cartesius/usage/interactive-usage) and [SURF Sara Program Development](https://userinfo.surfsara.nl/systems/cartesius/usage/Program-development).

## 5. Assignment

### Overview
The assignment consists of three tasks. The first task is to answer questions which aim to make sure that you have a good understanding of TensorFlow framework basics. The second task of this assignment is to implement a multilayer perceptron (MLP) in TensorFlow. In the third task you will need to study how the MLP's performance is influenced by different settings: weight initialization, regularization, activation functions, optimizers and architecture of the network. We encourage you to make your implementation flexible enough so that you can easily run experiments for the third task of this assignment and that we can automatically test your implementation. As a final step, we ask you to write a small report using [template](https://nips.cc/Conferences/2016/PaperInformation/StyleFiles) of [NIPS (Neural Information Processing Systems) conference](https://nips.cc/). NIPS is one of the top conferences in machine learning where you can monitor the cutting edge advances in deep learning and machine learning.

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
- ***mlp.py*** : This file contains interface of the `MLP` class. There are four methods of this class: `__init__(*args)`, `inference(*args)`, `loss(*args)`, `accuracy(*args)`. Implement these methods by strictly following the interfaces of these methods, otherwise we will not be able to check your implementation. You are free to add other methods to the class but, please, add an underscore in the beginning of the method, for example, ***\_private_method***. It is considered as a standard approach to specify a non-public part of the class interface. Check this [link](https://shahriar.svbtle.com/underscores-in-python#single-underscore-before-a-name-eg-code-class_1) for more information.

- ***train_mlp.py*** : This file contains two main functions. The function `train()` is a function where you need to implement training and testing procedure of the network on CIFAR10 dataset. You also need to save summaries which can be used for getting insights of your model on TensorBoard. Check the [MNIST with summaries tutorial](https://github.com/TensorFlow/TensorFlow/blob/master/TensorFlow/examples/tutorials/mnist/mnist_with_summaries.py) and [TensorFlow summary operations](https://www.TensorFlow.org/versions/r0.11/api_docs/python/train.html#summary-operations). TensorBoard is a very useful tool to understand, debug and optimize TensorFlow programs. Check [this tutorial](https://www.TensorFlow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html) to learn how to use it. You can also download the data from TensorBoard in CSV format in case you want the data (e.g. learning curves etc) accessible via other tools for visualization purposes such as `matplotlib`.

  Another function is the `main()` function which gets rid of already existing log directory and reinitializes one. Then it calls your `train()` function.

  Carefully go through all possible command line parameters and their possible values for running ***train_mlp.py***. You are going to implement each of these into your code. Otherwise we can not test your code.

- ***report/nips_2016.tex***, ***report/nips_2016.sty*** : The original LaTex  template and style file for writing NIPS paper. You should check them before writing your report. If you are not familiar with LaTex  you can check, for example, [this](https://www.latex-tutorial.com/) or [this](https://www.latex-tutorial.com/tutorials/) tutorials.

- ***report/report_lastname.tex*** : The template for your report in this assignment with the predefined structure of the sections. More information can be found in the Report section of this file. When you are submitting the assignment, please, replace ***lastname*** with your last name.

- ***cifar10/get_cifar10.sh*** : Shell script to download CIFAR10 data as in the Practical 1.

### Task 1 [20 points total]
After you have familiarized yourself with TensorFlow you should have a good understanding of the inner workings of the framework.

Please answer the following questions in your own words:

1. Describe TensorFlow constants, placeholders and variables. Point out the differences and provide explanation of how to use them in the context of convolutional neural networks. `[2 points]`
2. Give two examples of how to initialize variables in TensorFlow. Provide an example code (snippet). `[2 points]`
3. What is the difference between `tf.shape(x)` and `x.get_shape()` for a `Tensor` x? `[2 points]`
4. What do `tf.constant(True) == True` and `tf.constant(True) == tf.constant(True)` evaluate to? What consequence does that have on the use of conditionals in TensorFlow? `[2 points]`
5. What is the TensorFlow equivalent of `if ... else ...` when using `Bool` Tensors? Write down a short example code for such an `if ... else ...` statement in TensorFlow and report the results. `[2 points]`
6. Name 3 TensorFlow components that you need in order to run computations in TensorFlow. `[2 points]`
7. What are variable scopes used for? Is there a difference between a variable scope and a name scope? `[2 points]`
8. Can you *freeze* a given variable tensor such that it will maintain its value during, for instance, optimization? How? `[2 points]`
9. Does TensorFlow perform automatic differentiation? Name two occasions in which TensorFlow mechanism for differentiation can make your life more difficult. What are the advantages? `[2 points]`
10. Describe two ways to feed your own data into a TensorFlow graph. Shortly explain the pipelines. `[2 points]`

Answer the questions in the specified section of your report.

### Task 2 [50 points]
1. Implement an MLP in the ***mlp.py*** file by following the instructions in the Overview section and inside the file.
2. Implement training and testing procedures for your model in ***train_mlp.py*** by following instructions in the Overview section and inside the file.

To confirm that your code works, run your network with the default configuration. You should have similar performance as in Practical 1.

Include into the specified section of your report the following:
- Plots of accuracy and loss curves for the train and test data;
- Graph of your model;
- Histograms of the weights and biases of each layer;
- Histograms of logits.

### Task 3 [30 points total]
Using the implemented MLP model to study different settings and values of parameters to explore their influence on the model behavior. Perform the following experiments and report your results by plotting accuracy and loss curves (on train and test data separately). For each experiment, explain your findings. Put into your report any other information that supports your findings and explorations.

#### Experiment-1: Weight initialization [5 points]
In this experiment we are studying the influence of weight initialization on performance. The network architecture is the same as in the Practical 1. It consists of one hidden layer with 100 hidden units followed by a softmax layer.

Use these command line parameters for this experiment:
```bash
python train_mlp.py --dnn_hidden_units 100 --learning_rate 1e-3 --weight_reg none --max_steps 1500 --batch_size 200 --optimizer sgd --dropout_rate 0. --activation relu
```

Perform the following experiments.

1. Initialize weights using a normal distribution (as you did in Practical 1) with mean 0 and varying standard deviation parameter. Use `1e-5`, `1e-4`, `1e-3`, `1e-2` as the standard deviation. Do you notice any significant changes in classification performance and convergence rates?

2. Initialize weights using uniform distribution on intervals , `[-1e-5, 1e-5]`, `[-1e-4, 1e-4]`,`[-1e-3, 1e-3]`,`[-1e-2, 1e-2]`. Do you notice any significant changes in classification performance and convergence rates (i.e., how quickly the learning curves flat out)? Explain the factors that might be the cause of the differences that you observe.

3. Initialize weights using 'Xavier' initializer. Is it better than the previous two methods? Why?

#### Experiment-2: Interaction between initialization and activation [5 points]

Now, consider the following scenarios:
+ `--activation tanh, --weight_init normal --weight_init_scale 0.001`
+ `--activation tanh --weight_init xavier`
+ `--activation relu --weight_init normal --weight_init_scale 0.001`
+ `--activation relu --weight_init xavier`

Keep other parameters the same as in the Experiment-1.

Report the results and discuss relationships about activation functions and weight initialization methods. What other factors are important to consider for this type of experiment to make stronger conclusions? Why do you think we asked you to perform the experiment with `--activation tanh`. You can check the [original paper](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) that proposed this initialization method to have better understading.

#### Experiment-3: Architecture [5 points]
In this experiment we will go deeper and wider with the architecture. Now network will contain two hidden layers with 300 hidden units in each.

Consider the following scenarios:
+ `--dnn_hidden_units 300,300 --activation tanh, --weight_init normal --weight_init_scale 0.001`
+ `--dnn_hidden_units 300,300 --activation tanh --weight_init xavier`
+ `--dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 0.001`
+ `--dnn_hidden_units 300,300 --activation relu --weight_init xavier`

Keep other parameters the same as in Experiment-1.

What are your observations? Do you have the same accuracy performance as for shallower model as in Experiment-1 and Experiment-2? What do you think are the main reasons for such behavior?

#### Experiment-4: Optimizers [5 points]
In this experiment we will try to improve our results with deep model using other optimizers.  

Consider the following scenarios:

+ `--learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer adam`
+ `--learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer adagrad`
+ `--learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer adadelta`
+ `--learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer rmsprop`
+ `--learning_rate 1e-3 --dnn_hidden_units 300,300 --activation relu --weight_init normal --weight_init_scale 1e-3 --optimizer sgd`

Keep other parameters the same as in Experiment-1. Compare optimizers, what is the best one in terms of test accuracy and convergence rate? How can you explain the results? You can read more about different optimizers [here](http://sebastianruder.com/optimizing-gradient-descent/).

#### Experiment-5: Be creative [10 points]

In this experiment you are free to set any parameter configurations for the MLP model that you have implemented. For example you may want to try different regularization types, run your network for more iterations, add more layers, change the learning rate and other parameters as you like. Your goal is to beat the best test accuracy you have achieved so far.

*Hint*: You should be able to get at least 0.53 accuracy on the test set. But higher is better.

Explain in the report how you are choosing new parameters to test. Do you have any particular strategy?

Study your best model by plotting accuracy and loss curves. Also, plot a confusion matrix for this model. You can also visualize 4-5 examples per class for which your model makes confident (wrong) decisions.

### Report
You should write a small report about your study of MLP model on CIFAR10 dataset using the provided template for NIPS papers. Please, make your report to be self-contained without this README.md file.

The report should contain the following sections:

- **Abstract** : Should contain information about the current task and the summary of the study of the MLP model on CIFAR10 dataset.
- **Task 1** : Should contain information about the current task and some description of the MLP model and TensorFlow framework. Put the answers for the questions from the Task 1 into this section.  
- **Task 2** : Should contain your study of the default model from the Task 2.
- **Task 3** : Should contain results of your experiments. Please, describe all experiments settings to make the report self-contained without this file. Put each experiment in separate subsection.     
- **Conclusion** : Should contain conclusion of this study. For example, you can try to answer the following questions. What was done during this assignment? What features of TensorBoard were positive and what were negative for implementing MLP model and performing the experiments? What are the main insights you got from the study of the MLP model on CIFAR10 dataset?
- **References** : Reference section if needed.

### Submission
Create zip archive with the following structure:

```
lastname_assignment_2.zip
│   report_lastname.pdf
│   mlp.py
│   train_mlp.py
```

Replace `lastname` with your last name. After you create the zip file, please send it to **uva.deeplearning@gmail.com ONLY**. Because of the large number of students if you send it to any other email, we cannot guarantee that it will be corrected in time. As the subject of the email use: "Assignment 2: Your Full Name".

The deadline for the assignment is the **18th of November, 23:59**.
