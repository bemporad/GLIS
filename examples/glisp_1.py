"""
2D examples solved by GLISp
    - with box constraints only
    - check examples with known constraints in the file 'glisp_known_constraints.py'
    - check examples with unknown constraints in the file 'glisp_unknown_constraints.py'

Notes:
    - The preference expression step may be
        - integrated/fed directly into GLISp
        - incrementally provided to GLISp
    - RBF surrogate is used
        - The user may choose to recalibrate the RBF parameters during the solving process
    - For exploration step, one may use the following methods:
        - IDW
        - probability of improvement
    Templates of the aforementioned solving procedures are noted in this file

Authors: A. Bemporad, M. Zhu
"""

import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
from src.glis.glis import GLISp

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
    comparetol = 1e-4
    n_initial_random = 10

elif benchmark == "ackley":
    lb = -5.0 * np.ones(2)
    ub = 5.0 * np.ones(2)
    fun = lambda x: -20.0 * np.exp(-.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) - np.exp(
        0.5 * (np.cos(2.0 * np.pi * x[0]) + np.cos(2.0 * np.pi * x[1]))) + np.exp(1.0) + 20.0

    # compute optimum/optimizer by PSO
    xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
                       minfunc=1e-12, maxiter=10000)

    max_evals = 40
    comparetol = 1e-4
    n_initial_random = 10

max_prefs = max_evals - 1


####################################################################################
# Define synthetic preference function mapping (x1,x2) to {-1,0,1}
def pref_fun(x1, x2):
    f1 = fun(x1)
    f2 = fun(x2)
    if f1 <= f2 - comparetol:
        pref = -1
    elif f1 >= f2 + comparetol:
        pref = 1
    else:
        pref = 0
    return pref


##########################################


key = 3
np.random.seed(key)  # rng default for reproducibility
####################################################################################
print("Solve the problem by feeding the  preference expression step directly into the GLISp solver")
# Solve global optimization problem
prob1 = GLISp(bounds=(lb, ub), n_initial_random=n_initial_random)
xopt1 = prob1.solve(pref_fun, max_prefs)
X1 = np.array(prob1.X)
fbest_seq1 = list(map(fun, X1[prob1.ibest_seq]))
##########################################

# Plot
print("Optimization finished. Draw the plot")
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
fig.tight_layout(pad=2, h_pad=2, w_pad=2)

[x_plot, y_plot] = np.meshgrid(np.arange(lb[0], ub[0], .01), np.arange(lb[1], ub[1], .01))
z_plot = np.zeros(x_plot.shape)
for i in range(0, x_plot.shape[0]):
    for j in range(0, x_plot.shape[1]):
        z_plot[i, j] = fun(np.array([x_plot[i, j], y_plot[i, j]]))

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
    plt.savefig("glisp-1.png", dpi=300)
plt.show()

np.random.seed(key)
####################################################################################
print("Solve the problem incrementally (i.e., provide the preference at each iteration)")
# solve same problem, but incrementally
prob2 = GLISp(bounds=(lb, ub), n_initial_random=n_initial_random)
xbest2, x2 = prob2.initialize()  # get first two random samples
for k in range(max_prefs):
    pref = pref_fun(x2, xbest2)  # evaluate preference
    x2 = prob2.update(pref)
    xbest2 = prob2.xbest
X2 = np.array(prob2.X[:-1])
xopt2 = xbest2
fbest_seq2 = list(map(fun, X2[prob2.ibest_seq]))


##########################################
# assert np.linalg.norm(X1-X2)==0.0 and np.all(xopt1==xopt2)


####################################################################################
# Smart synthetic preference function that avoids recomputing the same value
# fun(xbest) multiple times:
def pref_fun_smart(x1, x2):
    pref_fun_smart.X.append(x1)
    f1 = fun(x1)
    pref_fun_smart.F.append(f1)

    X = np.array(pref_fun_smart.X)
    i = np.where(np.all(np.abs(X - x2) == 0.0, axis=1))[0]  # does x2 already exist in list ?
    if i.size == 0:
        pref_fun_smart.X.append(x2)
        f2 = fun(x2)
        pref_fun_smart.F.append(f2)
    else:
        f2 = pref_fun_smart.F[i[0].item()]

    if f1 <= f2 - comparetol:
        pref = -1
    elif f1 >= f2 + comparetol:
        pref = 1
    else:
        pref = 0
    return pref


pref_fun_smart.X = list()  # initialize preference function
pref_fun_smart.F = list()

key = 2
np.random.seed(key)
##########################################
print("Solve the problem use an alternative synthetic pref fun.")
prob3 = GLISp(bounds=(lb, ub), n_initial_random=n_initial_random, RBFcalibrate=True)
xopt3 = prob3.solve(pref_fun_smart, max_prefs)
X3 = np.array(prob3.X)
fbest_seq3 = list(map(fun, X3[prob3.ibest_seq]))

# assert np.linalg.norm(X1-X3)==0.0 and np.all(xopt1==xopt3)
