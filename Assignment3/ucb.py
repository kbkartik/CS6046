import numpy as np

from solver import Solver
from bandits import GaussianBandit

class UCB(Solver):
  def __init__(self, bandit, epochs, T, value_type):
    super(UCB, self).__init__(bandit)

    """
    bandit (Bandit): the bandit environment or adversary
    epochs (int): Number of epochs
    T (int): Number of timesteps/iterations
    value_type (String): Calculate regret in terms of loss/reward
    """

    self.epochs = epochs
    self.value_type = value_type
    self.T = T
    self.regret = np.empty((self.epochs, self.T))

  def reset_epoch_variables(self):
    self.arm_freq = np.ones((self.bandit.n_arms,))
    self.reward_estimate = self.bandit.generate_loss_or_reward() # Round robin init
    self.ucb = np.zeros(self.bandit.n_arms,)
    self.cum_z_t = np.zeros((self.bandit.n_arms,)) # Cumulative adversary vector
    self.cum_algo_loss = 0 # Cumulative algorithm loss

  def pick_arm(self):

    self.ucb = self.reward_estimate + np.sqrt((2*np.log(self.T))/self.arm_freq)
    return np.argmax(self.ucb)

  def run_one_iteration(self):

    arm = self.pick_arm()
    bandit_dist_vector = self.bandit.generate_loss_or_reward()

    # Update arm frequency and average loss
    self.arm_freq[arm] += 1
    self.reward_estimate[arm] += (1 / self.arm_freq[arm]) * (bandit_dist_vector[arm] - self.reward_estimate[arm])

    return arm, bandit_dist_vector