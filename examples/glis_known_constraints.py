"""
2D examples solved using GLIS (for visualization)
    - with KNOWN linear and/or nonlinear inequality constraints

Solve the problem with
    - feasible initial samples
    - infeasible initial samples

Authors: A. Bemporad, M. Zhu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from glis import GLIS

benchmark = "camelsixhumps"

feas_sampl_solve = True  # solve the problem with feasible initial samples
infeas_sampl_solve = True  # solve the problem with infeasible initial samples

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

    A = np.array([[1.6295, 1], [-1, 4.4553], [-4.3023, -1], [-5.6905, -12.1374], [17.6198, 1]])
    b = np.array([3.0786, 2.7417, -1.4909, 1, 32.5198])
    g = lambda x: np.array([x[0] ** 2 + (x[1] + 0.1) ** 2 - .5])

    xopt0_const = np.array([0.21305, 0.57424])  # constrained optimizers
    fopt0_const = -0.58445  # constrained optimum

key = 2

# for plotting
[x_plot, y_plot] = np.meshgrid(np.arange(lb[0], ub[0], .01), np.arange(lb[1], ub[1], .01))
z_plot = np.zeros(x_plot.shape)
for i in range(0, x_plot.shape[0]):
    for j in range(0, x_plot.shape[1]):
        z_plot[i, j] = fun(np.array([x_plot[i, j], y_plot[i, j]]))
plt.rcParams['text.usetex'] = True

if feas_sampl_solve:
    np.random.seed(key)  # rng default for reproducibility
    ####################################################################################
    print("Solve the problem 'camelsixhumps' with feasible initial sample")
    # Solve global optimization problem
    prob1 = GLIS(bounds=(lb, ub), A=A, b=b, g=g, n_initial_random=10)
    xopt_const1, fopt_const1 = prob1.solve(fun, max_evals)
    X1 = np.array(prob1.X)
    fbest_seq1 = prob1.fbest_seq
    ##########################################

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    fig.tight_layout(pad=2, h_pad=2, w_pad=2)
    ax[0].contour(x_plot, y_plot, z_plot, 100, alpha=.4)
    ax[0].plot(X1[:prob1.n_initial_random, 0], X1[:prob1.n_initial_random, 1], "*", color=[0.9258, 0.6914, 0.1250],
               markersize=4)
    ax[0].plot(X1[prob1.n_initial_random:, 0], X1[prob1.n_initial_random:, 1], "*", color=[0, 0.9, 0], markersize=4)
    ax[0].plot(xopt0_const[0], xopt0_const[1], "o", color=[0, 0.4470, 0.7410], markersize=6)
    ax[0].plot(xopt_const1[0], xopt_const1[1], "*", color=[0.8500, 0.3250, 0.0980], markersize=6)
    ax[0].set_title("feasible initial samples")
    ax[0].set_xlim(lb[0], ub[0])
    ax[0].set_ylim(lb[1], ub[1])

    V = np.array([[0.4104, -0.2748], [0.1934, 0.6588], [1.3286, 0.9136],
                  [1.8412, 0.0783], [1.9009, -0.9736]])

    patches = list()
    polygon = matplotlib.patches.Polygon(V, closed=True)
    patches.append(polygon)

    th = np.arange(0, 2 * np.pi, .01)
    N = th.size
    V = np.zeros((N, 2))
    for i in range(0, N):
        V[i, 0] = 0 + np.sqrt(.5) * np.cos(th[i])
        V[i, 1] = -.1 + np.sqrt(.5) * np.sin(th[i])
    circle = matplotlib.patches.Polygon(V, closed=True)
    patches.append(circle)

    collection = matplotlib.collections.PatchCollection(patches, edgecolor=[0, 0, 0], facecolor=[.5, .5, .5], alpha=0.6)
    ax[0].add_collection(collection)

    ax[1].plot(np.arange(0, max_evals), fbest_seq1, color=[0.8500, 0.3250, 0.0980])
    ax[1].set_title("best function value")
    ax[1].grid()
    ax[1].set_xlabel("queries")

    if savefigs:
        plt.savefig("glis_known_const_fea_sample.png", dpi=300)

    plt.show()

if infeas_sampl_solve:
    np.random.seed(key)  # rng default for reproducibility
    ####################################################################################
    print("Solve the problem 'camelsixhumps' with infeasible initial sample")
    prob2 = GLIS(bounds=(lb, ub), A=A, b=b, g=g, n_initial_random=10, feasible_sampling=False)
    xopt_const2, fopt_const2 = prob2.solve(fun, max_evals)
    X2 = np.array(prob2.X)
    fbest_seq2 = prob2.fbest_seq
    ##########################################

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    fig.tight_layout(pad=2, h_pad=2, w_pad=2)

    ax[0].contour(x_plot, y_plot, z_plot, 100, alpha=.4)
    ax[0].plot(X2[:prob2.n_initial_random, 0], X2[:prob2.n_initial_random, 1], "*", color=[0.9258, 0.6914, 0.1250],
               markersize=4)
    ax[0].plot(X2[prob2.n_initial_random:, 0], X2[prob2.n_initial_random:, 1], "*", color=[0, 0.9, 0], markersize=4)
    ax[0].plot(xopt0_const[0], xopt0_const[1], "o", color=[0, 0.4470, 0.7410], markersize=6)
    ax[0].plot(xopt_const2[0], xopt_const2[1], "*", color=[0.8500, 0.3250, 0.0980], markersize=6)
    ax[0].set_title("infeasible initial samples")
    ax[0].set_xlim(lb[0], ub[0])
    ax[0].set_ylim(lb[1], ub[1])

    V = np.array([[0.4104, -0.2748], [0.1934, 0.6588], [1.3286, 0.9136],
                  [1.8412, 0.0783], [1.9009, -0.9736]])

    patches = list()
    polygon = matplotlib.patches.Polygon(V, closed=True)
    patches.append(polygon)

    th = np.arange(0, 2 * np.pi, .01)
    N = th.size
    V = np.zeros((N, 2))
    for i in range(0, N):
        V[i, 0] = 0 + np.sqrt(.5) * np.cos(th[i])
        V[i, 1] = -.1 + np.sqrt(.5) * np.sin(th[i])
    circle = matplotlib.patches.Polygon(V, closed=True)
    patches.append(circle)

    collection = matplotlib.collections.PatchCollection(patches, edgecolor=[0, 0, 0], facecolor=[.5, .5, .5], alpha=0.6)
    ax[0].add_collection(collection)

    ax[1].plot(np.arange(0, max_evals), fbest_seq2, color=[0.8500, 0.3250, 0.0980])
    ax[1].set_title("best function value")
    ax[1].grid()
    ax[1].set_xlabel("queries")

    if savefigs:
        plt.savefig("glis_known_const_infea_sample.png", dpi=300)

    plt.show()
