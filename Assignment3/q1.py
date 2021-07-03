import numpy as np
import time
import matplotlib.pyplot as plt

class Bandit(object):

  def __init__(self):
    np.random.seed(int(time.time()))

  def generate_loss_or_reward(self):
    raise NotImplementedError

class GaussianBandit(Bandit):

  def __init__(self, mean, variance, n_arms, T):
    assert n_arms > 0

    """
    n_arms (int): Number of arms
    mean (float), variance (float): Gaussian distribution shape parameters
    """

    self.n_arms = n_arms
    self.mean = mean
    self.var = variance
    self.T = T

  def generate_loss_or_reward(self):
    bandit_dist = np.random.normal(self.mean[0], np.sqrt(self.var[0]), size=(self.T, 1))
    bandit_dist = np.append(bandit_dist, np.random.normal(self.mean[1], np.sqrt(self.var[1]), size=(self.T, 1)), axis=1)
    return bandit_dist

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

  def update_regret(self, ep_num, itr):
    """
    ep_num (int): epoch number
    itr (int): iteration number or timestep
    """
    
    # Calculate cumulative regret in terms of "loss"
    if self.value_type == "loss":
      self.regret[ep_num, itr] = self.cum_algo_loss #- np.min(self.cum_z_t)

    # Calculate mean cumulative regret over epoch
    if ep_num == self.epochs - 1 and itr == self.T - 1:
      self.mean_regret = np.mean(self.regret.copy(), 0)

  def run(self):
    assert self.bandit is not None

    for ep_num in range(self.epochs):

      self.reset_epoch_variables()
      for itr in range(self.T):
        self.t0 = itr + 1
        self.run_one_iteration()
        self.update_regret(ep_num, itr)


class ETC(Solver):
  def __init__(self, bandit, epochs, T, value_type):
    super(ETC, self).__init__(bandit)

    """
    bandit (Bandit): the bandit environment or adversary
    epochs (int): Number of epochs
    T (int): Number of timesteps/iterations
    value_type (String): Calculate regret in terms of loss/reward
    """

    self.epochs = epochs
    self.value_type = value_type
    self.T = T
    self.t0 = 0
    self.regret = np.empty((self.epochs, self.T))

  def reset_epoch_variables(self):
    self.bandit_array = self.bandit.generate_loss_or_reward()

  def pick_arm(self):

    if self.t0 % 2 != 0:
      self.arm_freq = np.ones((self.bandit.n_arms,)) * int((self.t0 - 1)/2)
      rand_arm = np.random.binomial(1, 0.5)
      self.arm_freq[rand_arm] += 1
    else:
      self.arm_freq = np.ones((self.bandit.n_arms,)) * int(self.t0/self.bandit.n_arms)

  def run_one_iteration(self):
    self.cum_algo_loss = 0 # Cumulative algorithm loss
    self.arm_loss = np.zeros((self.bandit.n_arms,))

    self.pick_arm()
    
    # Getting timesteps for selected arms
    arm1_range = np.arange(0, self.arm_freq[0], dtype=int)
    if len(arm1_range) == 1:
      arm1_range = arm1_range[0]
    arm2_range = np.arange(self.arm_freq[0], self.t0, dtype=int)
    if len(arm2_range) == 1:
      arm2_range = arm2_range[0]

    # Getting arm losses and cumulative pseudo-regret
    if self.arm_freq[0] != 0:
      self.arm_loss[0] = np.sum(self.bandit_array[arm1_range, 0])
      self.cum_algo_loss += np.sum(self.bandit_array[arm1_range, 0])
    if self.arm_freq[1] != 0:
      self.arm_loss[1] = np.sum(self.bandit_array[arm2_range, 1])
      self.cum_algo_loss += np.sum(self.bandit_array[arm2_range, 1])

    # Best arm after exploration
    best_arm_after_t0 = np.argmin(self.arm_loss)
    
    # Adding exploitation loss
    self.cum_algo_loss += np.sum(self.bandit_array[self.t0:, best_arm_after_t0])



def setT0(T, delta):

  # Calculate T0
  T0 = (4/np.power(delta, 2)) * np.log((T * np.power(delta, 2))/4)
  if int(T0) % 2 == 0:
    return int(T0)
  else:
    return int(T0) + 1


T = 1000 # No. of iterations
epochs = 1000 # No. of epochs
d = 2 # No. of experts
delta = 0.1
T0 = setT0(T, delta)

# Parameters for bandit (reward) distribution
means = [0, 0.1]

def q1a_run():
  var = [1, 1]

  # Declare bandit type
  gb = GaussianBandit(means, var, d, T)

  # Run ETC
  model_etc = ETC(gb, epochs, T, "loss")
  model_etc.run()

  # Plot graphs
  plt.figure(figsize=(7, 7), dpi=90);
  itrs = range(1, model_etc.regret.shape[1]+1)

  sd = np.std(model_etc.regret, 0)
  cum_mean_regret = np.mean(model_etc.regret, 0)
  plt.plot(itrs, cum_mean_regret, label="regret", color='black')
  plt.fill_between(itrs, cum_mean_regret - sd, cum_mean_regret + sd, alpha=0.8, color='yellow')
  plt.xlabel("Number of iterations")
  plt.ylabel("Regret")
  plt.title("ETC regret")
  plt.legend(loc=1)
  plt.savefig('Q1_a.png');

def q1b_run():
  var = [[5, 0.03], [5, 4.9], [0.03, 5], [0.08, 0.07]]
  colors = ['black', 'blue', 'red', 'yellow']
  plt.figure(figsize=(7, 7), dpi=90);
  for i, v in enumerate(var):
    # Declare bandit type
    gb = GaussianBandit(means, var[i], d, T)

    # Run ETC
    model_etc = ETC(gb, epochs, T, "loss")
    model_etc.run()

    cum_mean_regret = np.mean(model_etc.regret, 0)
    itrs = range(1, T+1)
    plt.plot(itrs, cum_mean_regret, label="variance:(%.2f, %.2f)"%(var[i][0], var[i][1]), color=colors[i])
    plt.xlabel("Number of iterations")
    plt.ylabel("Regret")
    plt.title("ETC regret")
    plt.legend(loc=1)
  plt.savefig('Q1_b.png');

# Run Part 2 of Question 1
q1a_run()

# Run Part 4 of Question 1
q1b_run()