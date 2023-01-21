"""
2D examples with unknown constraints solved using GLISp (for visualization)

Note:
    - Templates of the different solving procedures are available. Please check notes in the file 'glisp_1.py'

Authors: A. Bemporad, M. Zhu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from pyswarm import pso
from src.glis.glis import GLISp
from math import cos, sin, exp

# benchmark="MBC"   # Mishra's Bird function constrained
# benchmark='CHC' #CamelSixHumps function with feasibility constraints
benchmark = 'CHSC'  # CamelSixHumps function with feasibility and satisfactory constraints

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
    comparetol = 1e-4
    n_initial_random = 13
    delta = 1.

elif benchmark == "CHC":
    # CamelSixHumps function with feasibility constraints
    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])
    fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                     x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
    xopt0 = np.array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
    fopt0 = -1.0316  # unconstrained optimum
    xopt0_const = np.array([0.21305, 0.57424])  # constrained optimizers
    fopt0_const = -0.58445  # constrained optimum

    isUnknownConstrained = True
    isUnknownSatisfactionConstrained = False

    if isUnknownConstrained:
        Aineq_unkn = np.array([[1.6295, 1],
                               [-1, 4.4553],
                               [-4.3023, -1],
                               [-5.6905, -12.1374],
                               [17.6198, 1]])

        bineq_unkn = np.array([[3.0786, 2.7417, -1.4909, 1, 32.5198]])
        g_nl_unkn = lambda x: np.array([x[0] ** 2 + (x[1] + 0.1) ** 2 - .5])

        g_unkn_fun = lambda x: sum(np.maximum((Aineq_unkn.dot(x) - bineq_unkn).flatten("c"), 0.0)) + sum(
            np.maximum(g_nl_unkn(x), 0))
    else:
        g_unkn_fun = lambda x: 0

    if isUnknownSatisfactionConstrained:  # add the necessary eqns if relavent
        s_unkn_fun = lambda x: 0
    else:
        s_unkn_fun = lambda x: 0

    max_evals = 100
    comparetol = 1e-4
    n_initial_random = 30
    delta = 2.

elif benchmark == "CHSC":
    # CamelSixHumps function with feasibility and satisfactory constraints
    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])
    fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                     x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
    xopt0 = np.array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
    fopt0 = -1.0316  # unconstrained optimum
    xopt0_const = np.array([0.0781, 0.6562])  # constrained optimizers
    fopt0_const = -0.9050  # constrained optimum

    isUnknownConstrained = True
    isUnknownSatisfactionConstrained = True

    if isUnknownConstrained:
        g_unkn_fun = lambda x: x[0] ** 2 + (x[1] + 0.04) ** 2 - .8
    else:
        g_unkn_fun = lambda x: 0

    if isUnknownSatisfactionConstrained:  # add the necessary eqns if relavent
        Aineq_unkn = np.array([[1.6295, 1],
                               [0.5, 3.875],
                               [-4.3023, -4],
                               [-2, 1],
                               [0.5, -1]])

        bineq_unkn = np.array([[3.0786, 3.324, -1.4909, 0.5, 0.5]])
        s_unkn_fun = lambda x: sum(np.maximum((Aineq_unkn.dot(x) - bineq_unkn).flatten("c"), 0.0))
    else:
        s_unkn_fun = lambda x: 0

    max_evals = 50
    comparetol = 1e-4
    n_initial_random = 13
    delta = 1.

max_prefs = max_evals - 1


####################################################################################
def pref_fun_unkn_const(x1, x2):
    """
    Synthetic preference function that avoids recomputing the same value fun(xbest) multiple times:

    Express preference based on fun. eval. AND constraint satisfaction
    """
    pref_fun_unkn_const.X.append(x1)
    f1 = fun(x1)
    fes1 = eval_feas(x1, has_syn_unknown_const=isUnknownConstrained)
    soft1 = eval_sat(x1, has_syn_unknown_satfun=isUnknownSatisfactionConstrained)
    pref_fun_unkn_const.F.append(f1)
    pref_fun_unkn_const.Fes.append(fes1)
    pref_fun_unkn_const.Sat.append(soft1)

    X = np.array(pref_fun_unkn_const.X)
    i = np.where(np.all(np.abs(X - x2) == 0.0, axis=1))[0]  # does x2 already exist in list ?
    if i.size == 0:
        pref_fun_unkn_const.X.append(x2)
        f2 = fun(x2)
        fes2 = eval_feas(x2, has_syn_unknown_const=isUnknownConstrained)
        soft2 = eval_sat(x2, has_syn_unknown_satfun=isUnknownSatisfactionConstrained)
        pref_fun_unkn_const.F.append(f2)
        pref_fun_unkn_const.Fes.append(fes2)
        pref_fun_unkn_const.Sat.append(soft2)
    else:
        f2 = pref_fun_unkn_const.F[i[0].item()]
        fes2 = pref_fun_unkn_const.Fes[i[0].item()]
        soft2 = pref_fun_unkn_const.Sat[i[0].item()]

    # Make comparisons
    if isUnknownConstrained and isUnknownSatisfactionConstrained:
        if f1 <= f2 - comparetol:
            if (fes1 and soft1) or [fes1, fes2, soft1, soft2].count(True) == 0:
                pref = -1
            else:
                pref = 1
        elif f1 >= f2 + comparetol:
            if (fes2 and soft2) or [fes1, fes2, soft1, soft2].count(True) == 0:
                pref = 1
            else:
                pref = -1
        else:
            pref = 0
    elif isUnknownConstrained:
        if (fes1 == 1 and fes2 == 1):
            if f1 < f2 - comparetol:
                pref = -1
            elif f1 > f2 + comparetol:
                pref = 1
            else:
                pref = 0
        elif (fes1 == 1 and fes2 == 0):
            pref = -1
        elif (fes1 == 0 and fes2 == 1):
            pref = 1
        else:
            if g_unkn_fun(x1) < g_unkn_fun(x2) - comparetol:
                pref = -1
            elif g_unkn_fun(x1) > g_unkn_fun(x2) - comparetol:
                pref = 1
            else:
                pref = 0
    elif isUnknownSatisfactionConstrained:
        if (soft1 == 1 and soft2 == 1):
            if f1 < f2 - comparetol:
                pref = -1
            elif f1 > f2 + comparetol:
                pref = 1
            else:
                pref = 0
        elif (soft1 == 1 and soft2 == 0):
            pref = -1
        elif (soft1 == 0 and soft2 == 1):
            pref = 1
        else:
            if s_unkn_fun(x1) < s_unkn_fun(x2) - comparetol:
                pref = -1
            elif s_unkn_fun(x1) > s_unkn_fun(x2) - comparetol:
                pref = 1
            else:
                pref = 0
    return pref


##########################################


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
# Define synthetic feasibility/satisfactory check function
def eval_sat(x, has_syn_unknown_satfun=isUnknownSatisfactionConstrained):
    if has_syn_unknown_satfun is None:
        satisfactory = True
    else:
        satisfactory = s_unkn_fun(x) < 1.e-6
    return satisfactory


##########################################


key = 3
np.random.seed(key)  # rng default for reproducibility
##########################################
# Solve global optimization problem
# initialize preference function
pref_fun_unkn_const.X = list()
pref_fun_unkn_const.F = list()
pref_fun_unkn_const.Fes = list()
pref_fun_unkn_const.Sat = list()

if max_evals > 30 and isUnknownConstrained:
    scale_delta = True
prob1 = GLISp(bounds=(lb, ub), n_initial_random=n_initial_random, has_unknown_constraints=isUnknownConstrained,
              has_satisfaction_fun=isUnknownSatisfactionConstrained, scale_delta=scale_delta,
              RBFcalibrate=True, delta=delta)
xopt1 = prob1.solve(pref_fun_unkn_const, max_prefs, unknown_constraint_fun=eval_feas, satisfactory_fun=eval_sat)
X1 = np.array(prob1.X)
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

elif benchmark == "CHC":
    import matplotlib.patches as mpatches
    from math import sqrt

    th = np.arange(0, 2 * np.pi, .01)
    N = th.size
    V = np.zeros((N, 2))
    for i in range(0, N):
        V[i, 0] = 0 + sqrt(0.5) * cos(th[i])
        V[i, 1] = -0.1 + sqrt(0.5) * sin(th[i])
    circle = mpatches.Polygon(xy=V, closed=True)
    patches.append(circle)

    V = np.array([[0.4104, -0.2748], [0.1934, 0.6588], [1.3286, 0.9136],
                  [1.8412, 0.0783], [1.9009, -0.9736]])

    polygon = mpatches.Polygon(V, True)
    patches.append(polygon)

elif benchmark == "CHSC":
    import matplotlib.patches as mpatches
    from math import sqrt

    th = np.arange(0, 2 * np.pi, .01)
    N = th.size
    V = np.zeros((N, 2))
    for i in range(0, N):
        V[i, 0] = 0 + sqrt(0.8) * cos(th[i])
        V[i, 1] = -0.04 + sqrt(0.8) * sin(th[i])
    circle = mpatches.Polygon(xy=V, closed=True)
    patches.append(circle)

    V = np.array([[1.48, 0.667], [0.168, 0.836], [-0.041, 0.417],
                  [0.554, -0.223], [1.68, 0.34]])

    polygon = mpatches.Polygon(xy=V, closed=True)
    patches.append(polygon)

polygon = mpatches.Polygon(xy=V, closed=True)
patches.append(polygon)

collection = matplotlib.collections.PatchCollection(patches, edgecolor=[0, 0, 0], facecolor=[.5, .5, .5], alpha=0.6)
ax.add_collection(collection)

if savefigs:
    plt.savefig("glisp_unkn_const.png", dpi=300)

plt.show()
