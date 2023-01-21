"""
Examples with higher dimensions solved by GLISp

Note:
    - In this file, we solve the problem by
        - integrating/feeding the simulator/fun directly into the GLISp solver
        - with or without a user-defined nonlinear transformation of obj. fun.
    - Other formats are possible, check the file 'glisp_1.py' for more details

Authors: A. Bemporad, M. Zhu
"""

import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
from glis import GLISp

benchmark = "hartman6"
# benchmark="rosenbrock8"

if benchmark == "hartman6":
    nvars = 6
    lb = np.zeros(nvars)
    ub = np.ones(nvars)
    alphaH = np.array([1.0, 1.2, 3.0, 3.2])
    AH = np.array([[10, 3, 17, 3.5, 1.7, 8],
                   [0.05, 10, 17, 0.1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]])
    PH = 1.e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                           [2329, 4135, 8307, 3736, 1004, 9991],
                           [2348, 1451, 3522, 2883, 3047, 6650],
                           [4047, 8828, 8732, 5743, 1091, 381]])


    def fun(x):
        xx = x.reshape(-1)
        f = 0.0
        for j in range(4):
            aux = 0.0
            for i in range(6):
                aux = aux + (xx[i] - PH[j, i]) ** 2 * AH[j, i]
            f -= np.exp(-aux) * alphaH[j]
        return f


    fopt0 = -3.32237  # optimum
    xopt0 = np.array([.20169, .150011, .476874, .275332, .311652, .6573])  # optimizer
    max_evals = 80
    comparetol = 1e-4
    obj_transform = None

elif benchmark == "rosenbrock8":
    nvars = 8
    lb = -30. * np.ones(nvars)
    ub = -lb


    def fun(x):
        xx = x.reshape(-1)
        f = 0.0
        for j in range(7):
            f += 100. * (xx[j + 1] - xx[j] ** 2) ** 2 + (1. - xx[j]) ** 2
        return f  # optimize the logarithm of the function


    max_evals = 80
    comparetol = 1e-4
    obj_transform = lambda f: np.log(f)

    # compute optimum/optimizer by PSO
    xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
                       minfunc=1e-12, maxiter=10000)

max_prefs = max_evals - 1


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
        f2 = pref_fun_smart.F[i.item()]

    if f1 <= f2 - comparetol:
        pref = -1
    elif f1 >= f2 + comparetol:
        pref = 1
    else:
        pref = 0
    return pref


##########################################


key = 0
np.random.seed(key)  # rng default for reproducibility
##########################################
# Solve global optimization problem
pref_fun_smart.X = list()  # initialize preference function
pref_fun_smart.F = list()
prob = GLISp(bounds=(lb, ub), obj_transform=obj_transform)
xopt = prob.solve(pref_fun_smart, max_prefs)
X = np.array(prob.X)
fbest_seq = list(map(fun, X[prob.ibest_seq]))
cpu_time = prob.time_total
##########################################

print("Elapsed time: %5.4f s" % cpu_time)

plt.figure(figsize=(6, 4))
plt.rcParams['text.usetex'] = True

plt.plot(np.arange(0, max_evals), fbest_seq, color=[0.8500, 0.3250, 0.0980])
plt.plot(np.arange(0, max_evals), fopt0 * np.ones(max_evals))
plt.title("best function value")
plt.xlabel("queries")
plt.grid()
plt.show()
