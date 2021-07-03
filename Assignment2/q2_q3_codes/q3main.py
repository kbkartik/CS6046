import numpy as np
import matplotlib.pyplot as plt

from egreedy import EpsilonGreedy
from exp3 import EXP3
from bandits import epsGreedy_BetaBandit, EXP3_BetaBandit


T = 100000 # No. of iterations
epochs = 100 # No. of epochs
d = 10 # No. of experts
itrs = range(1, T+1)

# Parameters for bandit (reward) distribution
alpha = 5
beta = 5

# Different value of epsilon and list to store each epsilon's regret
epsilon = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 'RR']
e_greedy_regret = []

# Declare bandit type
b = epsGreedy_BetaBandit(alpha, beta, d)

# Run EpsilonGreedy
for eps in epsilon:
  model_e_greedy = EpsilonGreedy(b, eps, epochs, T, "reward")
  model_e_greedy.run()
  e_greedy_regret.append(model_e_greedy.regret_mean)

# Run EXP3
model_exp3 = EXP3(b, epochs, T, "reward")
model_exp3.run()
exp3_regret_mean = model_exp3.regret_mean


# Plot graphs
plt.figure(figsize=(14, 8), dpi=100)

for e, e_regret in zip(epsilon, e_greedy_regret):
  if e == 'RR':
    plt.plot(itrs, e_regret, label="$\epsilon=0.0$\n(round_robin)")
  else:
    plt.plot(itrs, e_regret, label="$\epsilon=%.1f$"%e)

plt.plot(itrs, exp3_regret_mean, label="EXP3")

plt.xlabel("Number of iterations")
plt.ylabel("Regret")
plt.title("Comparison of $\epsilon$-greedy and EXP3 regret")
plt.legend(loc=1)
plt.savefig('Q3.png')