import numpy as np
import matplotlib.pyplot as plt

from ucb import UCB
from thompson import Thompson
from bandits import GaussianBandit

def plot_graph(model_cum_regrets, dels):
    """
    model: model learnt by algorithm
    image_name (String): Image name for graph
    """

    plt.figure(figsize=(7, 7), dpi=80)
    plt.plot(dels, model_cum_regrets[:, 0], label="UCB", color='black')
    plt.plot(dels, model_cum_regrets[:, 1], label="TS", color='blue')
    
    plt.xlabel("$\Delta$")
    plt.ylabel("Regret")
    plt.title(r"Regret: UCB vs Thompson Sampling (TS) for different $\Delta$'s")
    plt.legend(loc='upper right')
    #plt.tight_layout()
    plt.savefig('Q2_b.png', dpi=300)

T = 10000 # No. of iterations
epochs = 100 # No. of epochs
d = 2 # No. of experts
dels = np.linspace(0.1, 1, num=10)
cum_regrets = []

for i in range(len(dels)):
  # Parameters for bandit (reward) distribution
  means = [0, dels[i]]
  var = [1, 1]

  # Declare bandit type
  gb = GaussianBandit(means, var, d)

  # Run UCB
  model_ucb = UCB(gb, epochs, T, "reward")
  model_ucb.run()

  # Run TS
  model_TS = Thompson(gb, epochs, T, "reward")
  model_TS.run()

  UCB_cum_regret = np.mean(model_ucb.regret, 0)[-1]
  TS_cum_regret = np.mean(model_TS.regret, 0)[-1]
  cum_regrets.append([UCB_cum_regret, TS_cum_regret])

# Plot graphs
plot_graph(np.array(cum_regrets), dels)

