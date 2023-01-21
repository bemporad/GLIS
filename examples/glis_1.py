"""
2D examples solved using GLIS (for visualization)
    - with box constraints only
    - check examples with known constraints in the file 'glis_known_constraints.py'
    - check examples with unknown constraints in the file 'glis_unknown_constraints.py'

Note:
    - Problems can be solved by
        - integrating/feeding the simulator/fun directly into the GLIS solver
            - simulator/fun:
                - input: a sample to test (provided by GLIS)
                - output: the evaluation
            - the intermediate steps within the simulator/fun are unknown to the GLIS (black-box)
        - incrementally (i.e., provide the function evaluation at each iteration)
    - User-defined nonlinear transformation of obj. fun. during the solving process is possible
    - RBF (default) or IDW surrogate fun. can be used to fit the surrogate fun.
        - different RBF models can be used (default: inverse_quadratic)
    - Templates of the aforementioned solving procedures are noted in this file

Authors: A. Bemporad, M. Zhu
"""

import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
from src.glis.glis import GLIS

benchmark = "camelsixhumps"
# benchmark="ackley"

savefigs = False

if benchmark == "camelsixhumps":
    # Camel six-humps function
    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])
    fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                     x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
    xopt0 = np.array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
    fopt0 = -1.0316  # unconstrained optimum
    max_evals = 60

elif benchmark == "ackley":
    lb = -5.0 * np.ones(2)
    ub = 5.0 * np.ones(2)
    fun = lambda x: -20.0 * np.exp(-.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) - np.exp(
        0.5 * (np.cos(2.0 * np.pi * x[0]) + np.cos(2.0 * np.pi * x[1]))) + np.exp(1.0) + 20.0

    # compute optimum/optimizer by PSO
    xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
                       minfunc=1e-12, maxiter=10000)
    max_evals = 100

key = 2
np.random.seed(key)  # rng default for reproducibility
####################################################################################
print("Solve the problem by feeding the simulator/fun directly into the GLIS solver")
# Solve global optimization problem
prob1 = GLIS(bounds=(lb, ub), n_initial_random=10)
xopt1, fopt1 = prob1.solve(fun, max_evals)
X1 = np.array(prob1.X)
fbest_seq1 = prob1.fbest_seq
##########################################

# Plot
print("Optimization finished. Draw the plot")
[x_plot, y_plot] = np.meshgrid(np.arange(lb[0], ub[0], .01), np.arange(lb[1], ub[1], .01))
z_plot = np.zeros(x_plot.shape)
for i in range(0, x_plot.shape[0]):
    for j in range(0, x_plot.shape[1]):
        z_plot[i, j] = fun(np.array([x_plot[i, j], y_plot[i, j]]))

plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
fig.tight_layout(pad=2, h_pad=2, w_pad=2)
ax[0].contour(x_plot, y_plot, z_plot, 100, alpha=.4)
ax[0].plot(X1[:prob1.n_initial_random, 0], X1[:prob1.n_initial_random, 1], "*", color=[0.9258, 0.6914, 0.1250],
           markersize=4)
ax[0].plot(X1[prob1.n_initial_random:, 0], X1[prob1.n_initial_random:, 1], "*", color=[0, 0.9, 0], markersize=4)
ax[0].plot(xopt0[0], xopt0[1], "o", color=[0, 0.4470, 0.7410], markersize=6)
ax[0].plot(xopt1[0], xopt1[1], "*", color=[0.8500, 0.3250, 0.0980], markersize=6)
ax[0].set_xlim(lb[0], ub[0])
ax[0].set_ylim(lb[1], ub[1])
ax[0].set_title("samples")

ax[1].plot(np.arange(0, max_evals), fbest_seq1, color=[0.8500, 0.3250, 0.0980])
ax[1].plot(np.arange(0, max_evals), fopt0 * np.ones(max_evals))
ax[1].set_title("best function value")
ax[1].set_xlabel("queries")
ax[1].grid()

if savefigs:
    plt.savefig("glis-1.png", dpi=300)
plt.show()

np.random.seed(key)  # reset seed
####################################################################################
print("Solve the problem incrementally (i.e., provide the function evaluation at each iteration)")
# solve same problem, but incrementally
prob2 = GLIS(bounds=(lb, ub), n_initial_random=10)
x2 = prob2.initialize()
for k in range(max_evals):
    f = fun(x2)
    x2 = prob2.update(f)
X2 = np.array(prob2.X[
              :-1])  # it is because in prob.update, it will calculate the next point to query (the last x2 is calculated at max_evals +1)
xopt2 = prob2.xbest
fopt2 = prob2.fbest
##########################################
# assert np.linalg.norm(X1-X2)==0.0 and np.all(xopt1==xopt2) and fopt1==fopt2


np.random.seed(key)  # reset seed
####################################################################################
print("Solve global optimization problem under a nonlinear transformation of objective function")
# Solve global optimization problem under a nonlinear transformation of objective function
prob3 = GLIS(bounds=(lb, ub), n_initial_random=10, obj_transform=lambda f: np.log(f + 2.))
xopt3, fopt3 = prob3.solve(fun, max_evals)
fbest_seq3 = prob3.fbest_seq
##########################################
# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
fig.tight_layout(pad=2, h_pad=2, w_pad=2)
plt.plot(np.arange(0, max_evals), fbest_seq1, color=[0.8500, 0.3250, 0.0980], label=r'$\min f(x)$')
plt.plot(np.arange(0, max_evals), fbest_seq3, color=[0.9258, 0.6914, 0.1250], label=r'$\min\log(f(x)+2)$')
plt.plot(np.arange(0, max_evals), fopt0 * np.ones(max_evals))
plt.title("best function value")
plt.xlabel("queries")
plt.legend(fontsize=12)
plt.grid()

if savefigs:
    plt.savefig("glis-2.png", dpi=300)

plt.show()

np.random.seed(key)  # reset seed
####################################################################################
print("Solve global optimization problem using IDW surrogates")
# Solve global optimization problem using IDW surrogates
prob4 = GLIS(bounds=(lb, ub), n_initial_random=10, rbf='IDW')
xopt4, fopt4 = prob4.solve(fun, max_evals)
##########################################


np.random.seed(key)  # reset seed
####################################################################################
print("Solve global optimization problem using non-default Gaussian RBF function")
# Solve global optimization problem using non-default Gaussian RBF function
from src.glis.glis import gaussian

prob5 = GLIS(bounds=(lb, ub), n_initial_random=10, rbf=gaussian, rbf_epsil=3.0)
xopt5, fopt5 = prob5.solve(fun, max_evals)
##########################################
