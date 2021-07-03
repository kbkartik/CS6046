import numpy as np
import time

class Bandit(object):

  def __init__(self):
    np.random.seed(int(time.time()))

  def generate_loss_or_reward(self):
    raise NotImplementedError

class epsGreedy_BetaBandit(Bandit):

  def __init__(self, alpha, beta, n_arms):
    assert n_arms > 0 and alpha > 0 and beta > 0

    """
    n_arms (int): Number of arms
    alpha (float), beta (float): Beta distribution shape parameters
    """

    self.n_arms = n_arms
    self.alpha = alpha
    self.beta = beta

  def generate_loss_or_reward(self):
    bandit_dist = []
    for i in range(1, self.n_arms+1):
      bandit_dist.append(np.random.beta(self.alpha, self.beta*i))
    return np.array(bandit_dist)

class EXP3_BetaBandit(Bandit):

  def __init__(self, alpha_1, beta_1, alpha_2, beta_2, n_arms):
    assert n_arms > 0
    assert alpha_1 > 0 and beta_1 > 0 and alpha_2 > 0 and beta_2 > 0

    #np.random.seed(int(time.time()))
    """
    n_arms (int): Number of arms
    alpha_1 (float), beta_1 (float): Beta distribution shape parameters for first 9 arms
    alpha_2 (float), beta_2 (float): Beta distribution shape parameters for 10th arm
    """

    self.n_arms = n_arms
    self.alpha_1 = alpha_1
    self.beta_1 = beta_1
    self.alpha_2 = alpha_2
    self.beta_2 = beta_2

  def generate_loss_or_reward(self):
    bandit_dist = np.append(np.random.beta(self.alpha_1, self.beta_1, size=self.n_arms-1), 
                                 np.random.beta(self.alpha_2, self.beta_2))
    return bandit_dist