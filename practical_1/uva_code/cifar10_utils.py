import numpy as np
import os
import cPickle as pickle

"""
This module implements utility functions to download CIFAR10 data. 
You don't need to change anything here.
"""

# Default paths for downloading CIFAR10 data
CIFAR10_FOLDER = 'cifar10/cifar-10-batches-py'
CIFAR10_DOWNLOAD_SCRIPT = 'cifar10/get_cifar10.sh'

def load_cifar10_batch(batch_filename):
  """ 
  Loads single batch of CIFAR10 data. 

  Args:
    batch_filename: Filename of batch to get data from.

  Returns:
    X: CIFAR10 batch data in numpy array with shape (10000, 32, 32, 3).
    Y: CIFAR10 batch labels in numpy array with shape (10000, ).

  """
  with open(batch_filename, 'rb') as f:
    batch = pickle.load(f)
    X = batch['data']
    Y = batch['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_cifar10(cifar10_folder):
  """ 
  Loads CIFAR10 train and test splits.

  Args:
    cifar10_folder: Folder which contains downloaded CIFAR10 data.

  Returns:
    X_train: CIFAR10 train data in numpy array with shape (50000, 32, 32, 3).
    Y_train: CIFAR10 train labels in numpy array with shape (50000, ).
    X_test: CIFAR10 test data in numpy array with shape (10000, 32, 32, 3).
    Y_test: CIFAR10 test labels in numpy array with shape (10000, ).
  
  """
  Xs = []
  Ys = []
  for b in range(1, 6):
    batch_filename = os.path.join(cifar10_folder, 'data_batch_' + str(b))
    X, Y = load_cifar10_batch(batch_filename)
    Xs.append(X)
    Ys.append(Y)    
  X_train = np.concatenate(Xs)
  Y_train = np.concatenate(Ys)
  X_test, Y_test = load_cifar10_batch(os.path.join(cifar10_folder, 'test_batch'))
  return X_train, Y_train, X_test, Y_test

def get_cifar10_raw_data():
  """
  Gets raw CIFAR10 data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz.

  Returns:
    X_train: CIFAR10 train data in numpy array with shape (50000, 32, 32, 3).
    Y_train: CIFAR10 train labels in numpy array with shape (50000, ).
    X_test: CIFAR10 test data in numpy array with shape (10000, 32, 32, 3).
    Y_test: CIFAR10 test labels in numpy array with shape (10000, ).
  """
  if not os.path.exists(CIFAR10_FOLDER):
    os.system(CIFAR10_DOWNLOAD_SCRIPT)

  X_train, Y_train, X_test, Y_test = load_cifar10(CIFAR10_FOLDER)

  return X_train, Y_train, X_test, Y_test

def preprocess_cifar10_data(X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, num_val = 1000):
  """
  Preprocesses CIFAR10 data by subsampling validation and by substracting mean from all images.

  Args:
    X_train_raw: CIFAR10 raw train data in numpy array.
    Y_train_raw: CIFAR10 raw train labels in numpy array.
    X_test_raw: CIFAR10 raw test data in numpy array.
    Y_test_raw: CIFAR10 raw test labels in numpy array.
    num_val: Number of validation samples.

  Returns:
    X_train: CIFAR10 train data in numpy array.
    Y_train: CIFAR10 train labels in numpy array.
    X_val: CIFAR10 validation data in numpy array.
    Y_va;: CIFAR10 validation labels in numpy array.
    X_test: CIFAR10 test data in numpy array.
    Y_test: CIFAR10 test labels in numpy array.
  """
  
  # Subsample validation set from train set
  num_train = X_train_raw.shape[0]
  mask_train = range(num_train - num_val)
  mask_val = range(num_train - num_val, num_train)
  X_val = X_train_raw[mask_val]
  Y_val = Y_train_raw[mask_val]
  X_train = X_train_raw[mask_train]
  Y_train = Y_train_raw[mask_train]
  X_test = X_test_raw.copy()
  Y_test = Y_test_raw.copy()

  # Substract the mean
  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image

  # Reshape data
  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  X_val = np.reshape(X_val, (X_val.shape[0], -1))
  X_test = np.reshape(X_test, (X_test.shape[0], -1))

  return X_train, Y_train, X_val, Y_val, X_test, Y_test