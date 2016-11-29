import numpy as np
from random import randrange

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  fx = f(x)
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h
    fxph = f(x)
    x[ix] = oldval - h
    fxmh = f(x)
    x[ix] = oldval

    grad[ix] = (fxph - fxmh) / (2 * h)
    if verbose:
      print ix, grad[ix]
    it.iternext()

  return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index
    
    oldval = x[ix]
    x[ix] = oldval + h
    pos = f(x).copy()
    x[ix] = oldval - h
    neg = f(x).copy()
    x[ix] = oldval
    
    grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    it.iternext()
  return grad
