import numpy as np

from banditsolver import Solver
from bandits import epsGreedy_BetaBandit, EXP3_BetaBandit

class EXP3(Solver):
  def __init__(self, bandit, epochs, T, value_type):
    super(EXP3, self).__init__(bandit)

    """
    bandit (Bandit): the bandit environment or adversary
    epochs (int): Number of epochs
    T (int): Number of timesteps/iterations
    value_type (String): Calculate regret in terms of loss/reward
    """

    self.epochs = epochs
    self.T = T
    self.value_type = value_type
    self.regret = np.empty((self.epochs, self.T)) # Regret array for all epochs and timesteps

    # Optimal learning rate for EXP3
    self.eta_lr = np.sqrt((2*np.log(self.bandit.n_arms))/(self.T*self.bandit.n_arms))

  def reset_epoch_variables(self):
    self.weights = np.ones((self.bandit.n_arms,))
    self.cum_z_t = np.zeros((self.bandit.n_arms,)) # Cumulative adversary vector
    self.cum_algo_loss = 0 # Cumulative algorithm loss

  def pick_arm(self):
    # Pick arm based on arm probabilities
    return np.random.choice(range(0, self.bandit.n_arms), replace=False, p=self.probabilities)
  
  def run_one_iteration(self):

    # Normalize weights
    self.probabilities = self.weights/np.sum(self.weights)
    arm = self.pick_arm()   

    bandit_dist_vector = self.bandit.generate_loss_or_reward()

    fake_loss_or_reward = np.zeros((self.bandit.n_arms,))
    fake_loss_or_reward[arm] = bandit_dist_vector[arm]/self.probabilities[arm]

    # Update weights
    if self.value_type == 'loss':
      self.weights = self.weights * np.exp(-1 * self.eta_lr * fake_loss_or_reward)
    else:
      self.weights = self.weights * np.exp((self.eta_lr * fake_loss_or_reward)/self.bandit.n_arms)

    return arm, bandit_dist_vector