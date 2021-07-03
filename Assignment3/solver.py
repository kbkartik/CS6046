import numpy as np
import time

from bandits import GaussianBandit

class Solver(object):
  def __init__(self, bandit):
    np.random.seed(int(time.time()))

    assert isinstance(bandit, GaussianBandit)

    self.bandit = bandit

  def run_one_iteration(self):
    raise NotImplementedError
  
  def reset_epoch_variables(self):
    raise NotImplementedError

  def pick_arm(self):
    raise NotImplementedError

  def update_regret(self, ep_num, itr, arm, bandit_dist_vector):
    """
    ep_num (int): epoch number
    itr (int): iteration number or timestep
    arm (int): index of arm picked
    bandit_dist_vector (np.array(float)): Bandit loss/reward distribution vector
    """
    
    self.cum_z_t += bandit_dist_vector # Cumulative adversary vector
    self.cum_algo_loss += bandit_dist_vector[arm] # Cumulative algorithm loss
    
    # Calculate cumulative regret in terms of "loss" or "reward"
    if self.value_type == "reward":
      self.regret[ep_num, itr] = np.max(self.cum_z_t) - self.cum_algo_loss
    elif self.value_type == "loss":
      self.regret[ep_num, itr] = self.cum_algo_loss - np.min(self.cum_z_t)

    # Calculate mean cumulative regret over epochs
    if ep_num == self.epochs - 1 and itr == self.T - 1:
      itrs = range(1, self.T+1)
      self.mean_regret = np.mean(self.regret.copy(), 0)

  def run(self):
    assert self.bandit is not None

    for ep_num in range(self.epochs):
    
      self.reset_epoch_variables()
      for itr in range(self.T):
        arm, bandit_dist_vector = self.run_one_iteration()
        self.update_regret(ep_num, itr, arm, bandit_dist_vector)