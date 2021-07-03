import numpy as np
import matplotlib.pyplot as plt

from egreedy import EpsilonGreedy
from exp3 import EXP3
from bandits import epsGreedy_BetaBandit, EXP3_BetaBandit

def plot_graph(model, title, image_name):

    """
    model: model learnt by algorithm
    title (String): Title for graph
    image_name (String): Image name for graph
    """

    sd = np.std(model.regret, 0)
    cum_regret_mean = np.mean(model.regret, 0)
    plt.figure(figsize=(10, 10));
    plt.plot(itrs, cum_regret_mean, label="regret", color='black')
    plt.fill_between(itrs, cum_regret_mean - sd, cum_regret_mean + sd, alpha=0.8, color='yellow')
    plt.xlabel("Number of iterations")
    plt.ylabel("Regret")
    plt.title(title)
    plt.savefig(image_name)


T = 10000 # No. of iterations
epochs = 1000 # No. of epochs
d = 10 # No. of experts
itrs = range(1, T+1)

# Beta distribution parameters
alpha_1 = 5
beta_1 = 5
alpha_2 = 5
beta_2 = 10

# Declare bandit type
b = EXP3_BetaBandit(alpha_1, beta_1, alpha_2, beta_2, d)

# Run EXP3
model_exp3 = EXP3(b, epochs, T, "loss")
model_exp3.run()

# Plot graph
plot_graph(model_exp3, r"EXP3 Regret $(\alpha_1=\beta_1=\alpha_2=5,\beta_2=10)$", 'EXP3_regret.png')

# Changing shape parameters of Beta distribution parameters
# to visualize how variance of regret is affected
alpha = 5
beta = 5

# Declare bandit type
b = epsGreedy_BetaBandit(alpha, beta, d)

# Run EXP3 with new distribution parameters
model_exp3g = EXP3(b, epochs, T, "loss")
model_exp3g.run()

# Plot graph
plot_graph(model_exp3g, r"EXP3 Regret for loss distribution Beta($\alpha=5, \beta=5*i$); $i:$ arm index", 'EXP3_regret_with_different_shape_parameters.png')