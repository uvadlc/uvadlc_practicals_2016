# Assignment 1

## Introduction
In this assignment you will get familiar with standard multi-layer perceptrons. You will start with a multinomial logistic regression classifier and then convert it to an one-layer neural network. Then you will add more layers to the network including different types of activation function.   

For those who are not familiar with python it is highly recommended to check https://docs.scipy.org/doc/numpy-dev/user/quickstart.html.

Also, check an ipython tutorial http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb.

## Instructions
The assignment consists of two parts: an ipython notebook **python_assignment_1.ipynb** and a pdf file **paper_assignment_1.pdf** with exercises that you will have to solve with pen and paper. You should follow the instructions in these files. You should work on both parts of the assignment.

For easier submission procedure we provide a simple script **collect_assignment.py** that will collect your submission files into a zip file that you should send to us. You can run the command as follows:

```bash
python collect_assignment.py --last_name your-last-name
```
To make sure you and we don't miss any important files or answers, please use the collect_assignment.py script. In case, however, that the script does not work for you, you can manually collect the files and the results. In that case the zip archive that you will send us must have the following structure:

```
lastname_assignment_1.zip
│   python_assignment_1.ipynb
│   python_assignment_1.pdf
│   paper_assignment_1_solved.pdf
│
└───uva_code
│   │   __init__.py
│   │   layers.py
|   |   losses.py
|   |   models.py
|   |   optimizers.py
|   |   solver.py
```

Please make sure that the structure of the file is correct, as the assignments are mostly going to be automatically checked and the specified structure is required. After you create the zip files, please send it to **uva.deeplearning@gmail.com ONLY**. Because of the large number of students if you send it to any other email, we cannot guarantee that it will be corrected in time.

The deadline for the assignment is the **9th of November, 23:59**.
