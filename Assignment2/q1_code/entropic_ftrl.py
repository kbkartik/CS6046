import numpy as np

from solver import Solver

class EntropicRegularizer(Solver):
  def __init__(self, dim, T, eta, epochs, dataset):
    self.dim = dim
    self.eta = eta
    self.T = T
    self.epochs = epochs
    self.dataset = dataset
    self.regret = np.empty((self.epochs, self.T))
    
  def reset_epoch_variables(self):
    self.initialize = True
    self.cum_z_t = np.zeros((self.dim,)) # Cumulative adversary vector for calculating regret
    self.cum_algo_loss = 0 # Cumulative algorithm loss

  def run_one_iteration(self):
    if self.initialize:
      self.weights = np.ones((self.dim,))
      self.p_t = self.weights/self.dim

    # Calculate loss at timestep t
    loss_t = np.dot(self.p_t, self.z_t)
    self.cum_algo_loss += loss_t
    
    # Get new decision vector
    if self.initialize:
      self.p_t = np.exp(-1*self.eta*self.z_t)/np.sum(np.exp(-1*self.eta*self.z_t))
      self.initialize = False
    else:
      self.p_t = np.exp(-1*self.eta*self.cum_z_t)/np.sum(np.exp(-1*self.eta*self.cum_z_t))

    # Update cumulative adversary vector
    self.cum_z_t += self.z_t
    
    # Calculate regret
    regret_t = self.cum_algo_loss - np.min(self.cum_z_t)
    
    return regret_t