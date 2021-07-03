import numpy as np

from solver import Solver
from bandits import GaussianBandit

class Thompson(Solver):
  def __init__(self, bandit, epochs, T, value_type):
    super(Thompson, self).__init__(bandit)

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
    self.arm_freq = np.zeros((self.bandit.n_arms,))
    self.prior_means = np.zeros((self.bandit.n_arms,))
    self.cum_z_t = np.zeros((self.bandit.n_arms,)) # Cumulative adversary vector
    self.cum_algo_loss = 0 # Cumulative algorithm loss

  def pick_arm(self):

    generate_params = []
    for i in range(0, self.bandit.n_arms):
      generate_params.append(np.random.normal(self.prior_means[i], np.sqrt(1/(self.arm_freq[i]+1))))

    return np.argmax(generate_params)

  def run_one_iteration(self):

    arm = self.pick_arm()
    bandit_dist_vector = self.bandit.generate_loss_or_reward()

    # Update arm frequency and prior mean
    self.arm_freq[arm] += 1
    #self.prior_means[arm] += (1 / self.arm_freq[arm]) * (bandit_dist_vector[arm] - self.prior_means[arm])
    self.prior_means[arm] = (self.prior_means[arm] * self.arm_freq[arm] + bandit_dist_vector[arm]) / (1+self.arm_freq[arm])

    return arm, bandit_dist_vector