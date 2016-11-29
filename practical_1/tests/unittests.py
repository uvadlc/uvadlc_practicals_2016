import unittest
import numpy as np

from uva_code.losses import HingeLoss, CrossEntropyLoss, SoftMaxLoss
from uva_code.layers import LinearLayer, ReLULayer, SigmoidLayer, TanhLayer, ELULayer, SoftMaxLayer

from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class TestLosses(unittest.TestCase):

  def test_hinge_loss(self):
    np.random.seed(42)
    rel_error_max = 1e-6
    
    for test_num in range(10):
      N = np.random.choice(range(1, 100))
      C = np.random.choice(range(1, 20))
      X = np.random.randn(N, C)
      y = np.random.randint(C, size=(N,))

      loss, grads = HingeLoss(X, y)

      f = lambda _: HingeLoss(X, y)[0]
      grads_num = eval_numerical_gradient(f, X, verbose = False, h = 1e-5)
      self.assertLess(rel_error(grads_num, grads), rel_error_max)

  def test_softmax_loss(self):
    np.random.seed(42)
    rel_error_max = 1e-5
    
    for test_num in range(10):
      N = np.random.choice(range(1, 100))
      C = np.random.choice(range(1, 20))
      X = np.random.randn(N, C)
      y = np.random.randint(C, size=(N,))

      loss, grads = SoftMaxLoss(X, y)

      f = lambda _: SoftMaxLoss(X, y)[0]
      grads_num = eval_numerical_gradient(f, X, verbose = False, h = 1e-5)
      self.assertLess(rel_error(grads_num, grads), rel_error_max)

  def test_crossentropy_loss(self):
    np.random.seed(42)
    rel_error_max = 1e-5
    
    for test_num in range(10):
      N = np.random.choice(range(1, 100))
      C = np.random.choice(range(1, 10))
      X = np.random.randn(N, C)
      y = np.random.randint(C, size=(N,))
      X = np.exp(X - np.max(X, axis = 1, keepdims = True))
      X /= np.sum(X, axis = 1, keepdims = True)

      loss, grads = CrossEntropyLoss(X, y)

      f = lambda _: CrossEntropyLoss(X, y)[0]
      grads_num = eval_numerical_gradient(f, X, verbose = False, h = 1e-5)
      self.assertLess(rel_error(grads_num, grads), rel_error_max)

class TestLayers(unittest.TestCase):

  def test_linear_backward(self):
    np.random.seed(42)
    rel_error_max = 1e-5

    for test_num in range(10):
      N = np.random.choice(range(1, 20))
      D = np.random.choice(range(1, 100))
      C = np.random.choice(range(1, 10))
      x = np.random.randn(N, D)
      dout = np.random.randn(N, C)

      layer_params = {'input_size': D, 'output_size': C, 'reg': 0.0, 'weight_scale': 0.001}
      layer = LinearLayer(layer_params)
      layer.initialize()
      layer.set_train_mode()

      out = layer.forward(x)
      dx = layer.backward(dout)
      dw = layer.grads['w']
      dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
      dw_num = eval_numerical_gradient_array(lambda w: layer.forward(x), layer.params['w'], dout)

      self.assertLess(rel_error(dx, dx_num), rel_error_max)
      self.assertLess(rel_error(dw, dw_num), rel_error_max)

  def test_relu_backward(self):
    np.random.seed(42)
    rel_error_max = 1e-6

    for test_num in range(10):
      N = np.random.choice(range(1, 20))
      D = np.random.choice(range(1, 100))
      x = np.random.randn(N, D)
      dout = np.random.randn(*x.shape)

      layer = ReLULayer()
      layer.initialize()
      layer.set_train_mode()

      out = layer.forward(x)
      dx = layer.backward(dout)
      dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)

      self.assertLess(rel_error(dx, dx_num), rel_error_max)

  def test_sigmoid_backward(self):
    np.random.seed(42)
    rel_error_max = 1e-6

    for test_num in range(10):
      N = np.random.choice(range(1, 20))
      D = np.random.choice(range(1, 100))
      x = np.random.randn(N, D)
      dout = np.random.randn(*x.shape)

      layer = SigmoidLayer()
      layer.initialize()
      layer.set_train_mode()
      
      out = layer.forward(x)
      dx = layer.backward(dout)
      dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)

      self.assertLess(rel_error(dx, dx_num), rel_error_max)

  def test_tanh_backward(self):
    np.random.seed(42)
    rel_error_max = 1e-6

    for test_num in range(10):
      N = np.random.choice(range(1, 20))
      D = np.random.choice(range(1, 100))
      x = np.random.randn(N, D)
      dout = np.random.randn(*x.shape)

      layer = TanhLayer()
      layer.initialize()
      layer.set_train_mode()
      
      out = layer.forward(x)
      dx = layer.backward(dout)
      dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)

      self.assertLess(rel_error(dx, dx_num), rel_error_max)

  def test_elu_backward(self):
    np.random.seed(42)
    rel_error_max = 1e-6

    for test_num in range(10):
      N = np.random.choice(range(1, 20))
      D = np.random.choice(range(1, 100))
      x = np.random.randn(N, D)
      dout = np.random.randn(*x.shape)

      layer = ELULayer(layer_params = {"alpha": 0.5})
      layer.initialize()
      layer.set_train_mode()
      
      out = layer.forward(x)
      dx = layer.backward(dout)
      dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)

      self.assertLess(rel_error(dx, dx_num), rel_error_max)

  def test_softmax_backward(self):
    np.random.seed(42)
    rel_error_max = 1e-5

    for test_num in range(10):
      N = np.random.choice(range(1, 20))
      D = np.random.choice(range(1, 100))
      x = np.random.randn(N, D)
      dout = np.random.randn(*x.shape)

      layer = SoftMaxLayer()
      layer.initialize()
      layer.set_train_mode()
      
      out = layer.forward(x)
      dx = layer.backward(dout)
      dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)

      self.assertLess(rel_error(dx, dx_num), rel_error_max)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(TestLosses)
  unittest.TextTestRunner(verbosity=2).run(suite)

  suite = unittest.TestLoader().loadTestsFromTestCase(TestLayers)
  unittest.TextTestRunner(verbosity=2).run(suite)
