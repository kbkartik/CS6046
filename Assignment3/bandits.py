import numpy as np
import time

class Bandit(object):

  def __init__(self):
    np.random.seed(int(time.time()))

  def generate_loss_or_reward(self):
    raise NotImplementedError

class GaussianBandit(Bandit):

  def __init__(self, mean, variance, n_arms):
    assert n_arms > 0

    """
    n_arms (int): Number of arms
    mean (float), variance (float): Gaussian distribution shape parameters
    """

    self.n_arms = n_arms
    self.mean = mean
    self.var = variance

  def generate_loss_or_reward(self):
    bandit_dist = []
    for i in range(0, self.n_arms):
      bandit_dist.append(np.random.normal(self.mean[i], np.sqrt(self.var[i])))
    return np.array(bandit_dist)