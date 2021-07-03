import numpy as np

from banditsolver import Solver
from bandits import epsGreedy_BetaBandit, EXP3_BetaBandit

class EpsilonGreedy(Solver):
  def __init__(self, bandit, eps, epochs, T, value_type):
    super(EpsilonGreedy, self).__init__(bandit)

    """
    bandit (Bandit): the bandit environment or adversary
    eps (float): epsilon value
    epochs (int): Number of epochs
    T (int): Number of timesteps/iterations
    value_type (String): Calculate regret in terms of loss/reward
    """

    assert (eps == 'RR') or (0. <= eps <= 1.0)
    if eps == 'RR':
      self.eps = 0
      self.round_robin = True
    else:
      self.eps = eps
      self.round_robin = False
    
    self.epochs = epochs
    self.T = T
    self.value_type = value_type
    self.regret = np.empty((self.epochs, self.T))
    self.orig_eps = eps

  def reset_epoch_variables(self):
    self.round_robin_complete = False
    self.arm_freq = np.zeros((self.bandit.n_arms,))
    self.average_reward = np.zeros((self.bandit.n_arms,))
    self.cum_z_t = np.zeros((self.bandit.n_arms,)) # Cumulative adversary vector
    self.cum_algo_loss = 0 # Cumulative algorithm loss

  def run_round_robin(self):
    # Run round robin for first 10 iterations
    for rr in range(self.bandit.n_arms):
      if self.arm_freq[rr] == 0:
        if rr == self.bandit.n_arms - 1:
          self.round_robin_complete = True
          #self.eps = 0
        return rr

  def pick_arm(self):
    explore = np.random.binomial(1, self.eps)
    if explore == 1 or np.sum(self.average_reward) == 0:
      # Random exploration  
      return np.random.randint(0, self.bandit.n_arms)
    else:
      # Pick the best one
      return np.argmax(self.average_reward)

  def run_one_iteration(self):
    
    if not(self.round_robin) or self.round_robin_complete:
      arm = self.pick_arm()
    else:
      # Pick arm in round robin fashion
      # and then set self.eps = 0.
      arm = self.run_round_robin()

    bandit_dist_vector = self.bandit.generate_loss_or_reward()

    # Update arm frequency and average reward
    self.arm_freq[arm] += 1
    self.average_reward[arm] = (1 - (1.0 / self.arm_freq[arm])) * self.average_reward[arm] + (1.0 / self.arm_freq[arm]) * bandit_dist_vector[arm]

    return arm, bandit_dist_vector