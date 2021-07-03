import numpy as np
import matplotlib.pyplot as plt

from ucb import UCB
from thompson import Thompson
from bandits import GaussianBandit

T = 10000 # No. of iterations
epochs = 100 # No. of epochs
d = 2 # No. of experts

# Parameters for bandit (reward) distribution
means = [0, 0.1]
var = [1, 1]

# Declare bandit type
gb = GaussianBandit(means, var, d)

# Run UCB
model_ucb = UCB(gb, epochs, T, "reward")
model_ucb.run()

# Run TS
model_TS = Thompson(gb, epochs, T, "reward")
model_TS.run()

# Plot graphs
plt.figure(figsize=(7, 7), dpi=80)
plt.plot(range(1, T+1), model_ucb.mean_regret, label="UCB", color='black')
plt.plot(range(1, T+1), model_TS.mean_regret, label="TS", color='blue')

plt.xlabel("T")
plt.ylabel("Regret")
plt.title(r"Regret: UCB vs Thompson Sampling (TS) for $\Delta=0.1$")
plt.legend(loc='upper right')
#plt.tight_layout()
plt.savefig('Q2_a.png', dpi=300)