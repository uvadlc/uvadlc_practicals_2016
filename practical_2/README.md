# Practical-2: Introduction to TensorFlow
## 1. Introduction
In Practical-1 we studied the backpropagation algorithm and we implemented a multinomial logistic regression classifier with an extension to a multi-layer neural network using *numpy* routines. In Practical-2, one of the goals is to get you familiar with Google's open-source Deep Learning framework called [TensorFlow](https://www.tensorflow.org/). In general, TensorFlow is a library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. As a TensorFlow user, you define the computational architecture of your predictive model, combine that with your objective function, and just add data - TensorFlow handles computing the derivatives for you.

The second goal of this practical is to investigate the influence of different parameters of model and training procedure. You will build your first neural network using TensorFlow and study the impacts of weight initialization, regularization, activation functions, optimizers and neural network architecture. You will need to write a small report about your findings during this study. As consequence, by the end of this practical, you will know the basic TensorFlow interfaces and routines which should be enough to start using your own models for either research purposes or for creating applications which can be deployed to many platforms like [mobile](https://www.tensorflow.org/mobile.html).

## 2. Prerequisites
As this practical doesn't need heavy GPU computations you can install TensorFlow on your own machine. We will need just CPU version of TensorFlow.

There are two main methods how you can install TensorFLow.

1. **Pre-built Docker container with TensorFlow**:

  To install TensorFlow using [Docker](https://www.docker.com/) follow these [instructions](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#docker-installation).

	***Note***: If you are on a Windows machine, this method is your only option due to lack of native TensorFlow support.

2. **Install TensorFlow on your computer (Linux or Mac OS X only)**:

	 Follow the instructions to [download and setup TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup). Choose one of the three ways to install:

    - ***Pip***: Install TensorFlow directly on your computer. You need to have Python 2.7 and pip installed; and this may impact other Python packages that you may have.
    - ***Virtualenv***: Install TensorFlow in an isolated (virtual) Python environment. You need to have Python 2.7 and virtualenv installed; this will not affect Python packages in any other environment.
    - ***Docker***: Run TensorFlow in an isolated Docker container (virtual machine) on your computer. You need to have Vagrant, Docker and virtualization software like VirtualBox installed; this will keep TensorFlow completely isolated from the rest of your computer, but may require more memory to run.

As debugging code on the server like SURF Sara is not an easy task so it is highly recommended to have TensorFlow on your own machine.

## 3. Tutorials

During a practical session on the 10th of November, we will go through the basic tutorial introducing TensorFlow. In general, it is [Stanford's Deep Learning in NLP (CS224d) presentation about TensorFlow](https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf).

These three tutorials are also very important to help you to get through the basics of TensorFlow: [MNIST for ML Beginners](https://www.tensorflow.org/versions/master/tutorials/index.html), [Deep MNIST for Experts](https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts) and [TensorFlow Mechanics 101](https://www.tensorflow.org/versions/master/tutorials/mnist/tf/index.html#tensorflow-mechanics-101).  

Other useful links: [Tutorials](https://www.tensorflow.org/versions/r0.11/tutorials/index.html), [How-Tos](https://www.tensorflow.org/versions/master/how_tos/index.html), [Resources](https://www.tensorflow.org/versions/master/resources/index.html), [Source code](https://github.com/tensorflow/tensorflow), [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).

## 4. Troubleshooting
If you have any questions about TensorFlow, first, try to find the answer by yourself using the following resources: [TensorFlow FAQ](https://www.tensorflow.org/versions/r0.11/resources/faq.html), [TensorFlow FAQ on StackOverflow](http://stackoverflow.com/questions/tagged/tensorflow?sort=frequent), [TensorFlow Google Groups](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss). If you still have questions ask them during practical sessions or in [Piazza](https://piazza.com/class/iuxuidh437j3ed) so other students can also see the question.

If you are using SURF Sara check the [SURF Sara tutorial]("http://uvadlc.github.io/lectures/surfsara-slides.pdf") and [Practical-0](https://github.com/uvadlc/uvadlc_practicals_2016/tree/master/practical_0). Another useful resources are [SURF Sara Interactive Usage](https://userinfo.surfsara.nl/systems/cartesius/usage/interactive-usage) and [SURF Sara Program Development](https://userinfo.surfsara.nl/systems/cartesius/usage/Program-development).

## 5. Assignment
Will be available soon. 
