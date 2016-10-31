"""
This module implements various optimization functions for the neural networks.
You should fill in code into indicated sections. 
"""

class SGD(object):
  """
  Implements vanilla stochastic gradient descent.
  
  """
  def __call__(self, w, dw, config = None):
    """
    Implements vanilla stochastic gradient descent.

    Args:
      w: Input weights.
      dw: Gradient of the loss with respect to input weights w.
      config: Dictionary with configuration parameters:
        learning_rate - learning rate

    Returns:
      next_w: Updated weights.
      next_config: Updated config.

    """
    if config is None: 
      config = {}
    
    config.setdefault('learning_rate', 1e-2)

    ########################################################################################
    # TODO:                                                                                #
    # Compute new weights according to vanilla SGD update rule. Store new weights in       #
    # next_w, new config of optimizer in next_config variables respectively.               #
    ########################################################################################
    next_w = None
    next_config = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return next_w, next_config