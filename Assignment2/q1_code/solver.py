import numpy as np
import time

class Solver(object):
  def __init__(self):
    np.random.seed(int(time.time()))

  def update_weights(self):
    raise NotImplementedError

  def reset_epoch_variables(self):
    raise NotImplementedError

  def update_regret(self, regret_t, ep_num, itr):
    self.regret[ep_num, itr] = regret_t
    
    if ep_num == self.epochs - 1 and itr == self.T - 1:
      itrs = range(1, self.T+1)
      self.regret_mean = np.mean(self.regret.copy()/itrs, 0)

  def run(self):
    for ep_num in range(self.epochs):
      self.reset_epoch_variables()
      for i in range(self.T):
        self.z_t = self.dataset[i]
        regret_t = self.run_one_iteration()
        self.update_regret(regret_t, ep_num, i)