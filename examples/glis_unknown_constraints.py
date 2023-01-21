"""
2D examples with unknown constraints solved using GLIS (for visualization)

Note:
    - Templates of the different solving procedures are available. Please check notes in the file 'glis_1.py'

Authors: A. Bemporad, M. Zhu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from pyswarm import pso
from glis.solvers import GLIS
from math import cos, sin, exp

benchmark = "MBC"  # Mishra's Bird function constrained

savefigs = False

if benchmark == "MBC":
    # Mishra's Bird function constrained
    lb = np.array([-10.0, -6.5])
    ub = np.array([-2, 0.0])
    fun = lambda x: sin(x[1]) * exp((1 - cos(x[0])) ** 2) + cos(x[0]) * exp((1 - sin(x[1])) ** 2) + (x[0] - x[1]) ** 2
    xopt0 = np.array([-3.1302468, -1.5821422])  # unconstrained optimizer
    fopt0 = -106.7645367  # unconstrained optimum

    xopt0_const = np.array([-9.3669, -1.62779])  # constrained optimizer
    fopt0_const = -48.4060  # constrained optimum

    isUnknownConstrained = True
    isUnknownSatisfactionConstrained = False

    if isUnknownConstrained:
        g_unkn_fun = lambda x: (x[0] + 9) ** 2 + (x[1] + 3) ** 2 - 9
    else:
        g_unkn_fun = lambda x: 0

    if isUnknownSatisfactionConstrained:  # add the necessary eqns if relavent
        s_unkn_fun = lambda x: 0
    else:
        s_unkn_fun = lambda x: 0

    max_evals = 50
    n_initial_random = 13


####################################################################################
# Define synthetic feasibility check function
def eval_feas(x, has_syn_unknown_const=isUnknownConstrained):
    if has_syn_unknown_const is None:
        feasible = True
    else:
        feasible = g_unkn_fun(x) < 1.e-6

    return feasible


##########################################


####################################################################################
# Define synthetic satisfactory check function
def eval_sat(x, has_syn_unknown_satfun=isUnknownSatisfactionConstrained):
    if has_syn_unknown_satfun is None:
        satisfactory = True
    else:
        satisfactory = s_unkn_fun(x) < 1.e-6

    return satisfactory


##########################################


key = 2
np.random.seed(key)  # rng default for reproducibility
##########################################
# Solve global optimization problem
print("Solve the problem by feeding the simulator/fun directly into the GLIS solver")
if max_evals > 30 and isUnknownConstrained:
    scale_delta = True
prob1 = GLIS(bounds=(lb, ub), n_initial_random=n_initial_random, has_unknown_constraints=isUnknownConstrained,
             has_satisfaction_fun=isUnknownSatisfactionConstrained, scale_delta=scale_delta)
xopt1, fopt1 = prob1.solve(fun, max_evals, unknown_constraint_fun=eval_feas, satisfactory_fun=eval_sat)
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

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
fig.tight_layout(pad=2, h_pad=2, w_pad=2)
ax.contour(x_plot, y_plot, z_plot, 100, alpha=.4)
ax.plot(X1[:prob1.n_initial_random, 0], X1[:prob1.n_initial_random, 1], "*", color=[0.9258, 0.6914, 0.1250],
        markersize=4)
ax.plot(X1[prob1.n_initial_random:, 0], X1[prob1.n_initial_random:, 1], "*", color=[0, 0.9, 0], markersize=4)
ax.plot(xopt0_const[0], xopt0_const[1], "o", color=[0, 0.4470, 0.7410], markersize=6)
ax.plot(xopt1[0], xopt1[1], "*", color=[0.8500, 0.3250, 0.0980], markersize=6)
ax.set_title("samples")
ax.set_xlim(lb[0], ub[0])
ax.set_ylim(lb[1], ub[1])

patches = []

if benchmark == "MBC":
    import matplotlib.patches as mpatches

    th = np.arange(0, 2 * np.pi, .01)
    N = th.size
    V = np.zeros((N, 2))
    for i in range(0, N):
        V[i, 0] = -9 + 3 * cos(th[i])
        V[i, 1] = -3 + 3 * sin(th[i])
    circle = mpatches.Polygon(xy=V, closed=True)
    patches.append(circle)

polygon = mpatches.Polygon(xy=V, closed=True)
patches.append(polygon)

collection = matplotlib.collections.PatchCollection(patches, edgecolor=[0, 0, 0], facecolor=[.5, .5, .5], alpha=0.6)
ax.add_collection(collection)

if savefigs:
    plt.savefig("glis_unkn_const.png", dpi=300)

plt.show()
