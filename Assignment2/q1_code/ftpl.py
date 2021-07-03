import numpy as np

from solver import Solver

class RandomizedRegularizer(Solver):
  def __init__(self, dim, T, epochs, dataset):
    self.dim = dim
    self.T = T
    self.epochs = epochs
    self.dataset = dataset
    self.regret = np.empty((self.epochs, self.T))

    # Optimal eta value for noise set as per Kalai & Vempala, 2004
    self.noise_eta = 1/np.sqrt(2/(self.T*np.linalg.norm(self.dataset, ord=1)*np.max(self.dataset)))
    #self.noise_eta = 1/np.sqrt(np.log(self.dim)/self.T)
    #self.noise_eta = 1/np.sqrt(self.T)
    
  def reset_epoch_variables(self):
    self.initialize = True
    self.cum_z_t = np.zeros((self.dim,)) # Cumulative adversary vector for calculating regret
    self.cum_z_t_ur = np.zeros((self.dim,)) # Cumulative adversary vector for update rule
    self.R = np.random.uniform(0, self.noise_eta, size=self.dim) # Random vector sampled from uniform distribution
    self.cum_algo_loss = 0 # Cumulative algorithm loss

  def run_one_iteration(self):
    if self.initialize:
      self.weights = np.ones((self.dim,))
      self.p_t = self.weights/self.dim
      self.initialize = False
    else:
      # Get new decision vector
      self.p_t = np.zeros((self.dim,))
      self.p_t[np.argmin(self.cum_z_t_ur)] = 1

    # Calculate loss at timestep t
    loss_t = np.dot(self.p_t, self.z_t)
    self.cum_algo_loss += loss_t

    # Update cumulative loss
    self.cum_z_t += self.z_t
    self.cum_z_t_ur += self.z_t + self.R
    
    # Calculate cumulative regret
    regret_t = self.cum_algo_loss - np.min(self.cum_z_t)
    
    return regret_t