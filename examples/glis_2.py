"""
Examples with higher dimensions solved by GLIS

Note:
    - In this file, we solve the problem by
        - integrating/feeding the simulator/fun directly into the GLIS solver
        - with or without a user-defined nonlinear transformation of obj. fun.
    - Other formats are possible, check the file 'glis_1.py' for more details

Authors: A. Bemporad, M. Zhu
"""

import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
from src.glis.glis import GLIS

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
    obj_transform = None

elif benchmark=="rosenbrock8":
    nvars = 8
    lb = -30.*np.ones(nvars)
    ub = -lb
    def fun(x):
        xx = x.reshape(-1)
        f = 0.0
        for j in range(7):
            f += 100.*(xx[j+1]-xx[j]**2)**2+(1.-xx[j])**2
        return f # optimize the logarithm of the function
    max_evals = 80
    obj_transform = lambda f: np.log(f)

    # compute optimum/optimizer by PSO
    xopt0, fopt0 = pso(fun, lb, ub, swarmsize=200,
                       minfunc=1e-12, maxiter=10000)



key=0
np.random.seed(key) #rng default for reproducibility
##########################################
# Solve global optimization problem
prob = GLIS(bounds=(lb, ub), delta=0.1, obj_transform=obj_transform)
xopt1, fopt1 = prob.solve(fun, max_evals)
X=np.array(prob.X)
fbest_seq = prob.fbest_seq
cpu_time=prob.time_total
##########################################

print("Elapsed time: %5.4f s" % cpu_time)

plt.figure(figsize=(6,4))
plt.rcParams['text.usetex'] = True

plt.plot(np.arange(0, max_evals), fbest_seq, color=[0.8500, 0.3250, 0.0980])
plt.plot(np.arange(0,max_evals),fopt0*np.ones(max_evals))
plt.title("best function value")
plt.xlabel("queries")
plt.grid()
plt.show()


