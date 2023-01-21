"""
GLIS - (GL)obal optimization solvers using (I)nverse distance weighting and
radial basis function (S)urrogates.

(C) 2019-2023 Alberto Bemporad, Mengjia Zhu

GLIS is a package for finding the global (GL) minimum of a function that is expensive
to evaluate, possibly under constraints, using inverse (I) distance weighting and
surrogate (S) radial basis functions.

The package implements two main algorithms:

1) GLIS, to solve derivative-free constrained global optimization problems
where the objective function a black-box:

min  f(x)
s.t. lb <= x <=ub, A*x <=b, g(x)<=0

using the global optimization algorithm described in [1]. The approach is
particularly useful when f(x) is time-consuming to evaluate, as it
attempts at minimizing the number of function evaluations.

2) GLISp, to globally optimize a function $f$ whose value cannot be even evaluated but,
given two points x and y, it is possible to query whether f(x) is better or worse than
f(y). More generally, one can only evaluate a preference function pi(x,y)

pi(x,y) = -1 if x is better than y
pi(x,y) =  1 if x is worse than y
pi(x,y) =  0 if x is as good as y

and want to solve the following preference-based optimization problem:

find a feasible vector x* such that pi(x*,x)<=0 for all feasible vectors x.

GLISp is particularly useful to solve optimization problems that involve human assessments.
In fact, there is no need to quantify an objective function f, which instead remain
underlying in the head of the decision-maker expressing the preferences.

See more in the README.md file.

[1] A. Bemporad, "Global optimization via inverse weighting and radial basis functions,"
    Computational Optimization and Applications, vol. 77, pp. 571–595, 2020.

[2] A. Bemporad and D. Piga, “Active preference learning based on radial basis functions,”
    Machine Learning, vol. 110, no. 2, pp. 417–448, 2021.

[3] M. Zhu, D. Piga, and A. Bemporad, “C-GLISp: Preference-based global optimization under
    unknown constraints with applications to controller calibration,” IEEE Trans. Contr.
    Systems Technology, vol. 30, no. 3, pp. 2176–2187, Sept. 2022.

"""

import numpy as np
from pyswarm import pso  # the "pyswarms" package could be used here too
import contextlib
import io
from scipy.optimize import linprog
from pyDOE import lhs
import time
from numba import njit
import warnings
from glis.rbf import inverse_quadratic


@njit
def facquisition(xs, Xs, F, X_all, F_all, useRBF, rbf_xs, W, delta_E, dF,
                 delta_G, delta_S, scale_delta, N, maxevals, alpha, iw_ibest,
                 has_unknown_constraints, has_satisfaction_fun, UnknownFeasible, UnknownSatisfactory,
                 isfeas_seq, constrpenalty_value, W_unkn, rbf_xs_unkn, F_unkn):
    """
    Acquisition function to minimize to get next sample in GLIS (surrogate + exploration)

    Note: in case samples that are infeasible wrt unknown constraints exist or if infeasible initial sampling is allowed
        - here Xs only collects the K feasible ones, and W has dimension K.
        - X_all and F_all collect all the samples (feasible and infeasible), while only the feasible ones are used to construct the surrogate
    """

    d = np.sum((Xs - xs) ** 2, axis=1)

    if np.all(isfeas_seq):  # if samples are all feasible
        d_all = d
    else:
        # to account for all X that have been sampled (including the infeasible ones,
        # since the distance info is used to estimate the probability of feasibility)
        d_all = np.sum((X_all - xs) ** 2, axis=1)

    ii = np.where(d_all < 1e-12)
    if ii[0].size > 0:
        fhat = (F_all[ii[0][0]]).item()
        fhat_unkn = (F_unkn[ii[0][0]]).item()
        if (not isfeas_seq[ii[0][0]]):  # if the sample is infeasible
            fhat = (constrpenalty_value * dF).item()
        dhat = 0.
        if has_unknown_constraints:
            Ghat = UnknownFeasible[ii[0][0]]
        else:
            Ghat = 1.
        if has_satisfaction_fun:
            Shat = UnknownSatisfactory[ii[0][0]]
        else:
            Shat = 1.
    else:
        if W is None:  # when all the current samples are infeasible
            #           in this case, aux_all will be used
            fhat = 0.
            w = np.array([0.]).reshape((1, -1))  # placeholder necessary for  @njit
            sw = 0.
            aux = 0.

            if has_unknown_constraints:
                w_unkn = (np.exp(-d_all) / d_all).reshape((1, -1))
                sw_unkn = np.sum(w_unkn)
                if useRBF:
                    v_infes = rbf_xs_unkn
                    fhat_unkn = (np.sum(v_infes * W_unkn)).item()
                else:
                    fhat_unkn = (np.sum(F_unkn * w_unkn) / sw_unkn).item()
            else:
                fhat_unkn = np.zeros((1)).item()
        else:
            w = (np.exp(-d) / d).reshape((1, -1))
            sw = np.sum(w)
            aux = 1. / sum(1. / d)

            if useRBF:
                v = rbf_xs
                fhat = (np.sum(v * W)).item()
            else:
                fhat = (np.sum(F * w) / sw).item()

            if has_unknown_constraints:
                w_unkn = (np.exp(-d_all) / d_all).reshape((1, -1))
                sw_unkn = np.sum(w_unkn)
                if useRBF:
                    v_infes = rbf_xs_unkn
                    fhat_unkn = (np.sum(v_infes * W_unkn)).item()
                else:
                    fhat_unkn = (np.sum(F_unkn * w_unkn) / sw_unkn).item()
            else:
                fhat_unkn = np.zeros((1)).item()

        if np.all(isfeas_seq):
            w_all = w
            sw_all = sw
            aux_all = aux
        else:
            w_all = (np.exp(-d_all) / d_all).reshape((1, -1))
            sw_all = np.sum(w_all)
            aux_all = 1. / sum(1. / d_all)

        # when considering the IDW exploration function, take all the points sampled into account
        if not scale_delta:
            # for comparison, used in the original GLIS and when N_max <= 30 in C-GLIS
            dhat = delta_E * np.arctan(aux_all)
            if W is not None:  # when there exists feasible samples
                dhat = dhat * 2. / np.pi * dF + alpha * np.sqrt(np.sum(w * (F - fhat) ** 2) / sw)
        else:
            dhat = delta_E * ((1. - N / maxevals) * np.arctan(aux_all * iw_ibest) + N / maxevals *
                              np.arctan(aux_all))
            if W is not None:  # when there exists feasible samples
                dhat = dhat * 2.0 / np.pi * dF + alpha * np.sqrt(np.sum(w * (F - fhat) ** 2) / sw)

        # to account for the unknown constraints
        if has_unknown_constraints:
            Ghat = (np.sum(UnknownFeasible * w_all) / sw_all).item()
        else:
            Ghat = 1.

        if has_satisfaction_fun:
            Shat = (np.sum(UnknownSatisfactory * w_all) / sw_all).item()
        else:
            Shat = 1.

    return fhat - dhat + (delta_G * (1. - Ghat) + delta_S * (1. - Shat)) * fhat_unkn


@njit
def facquisition_pref_surrogate(xs, Xs, rbf_xs, W, delta_E, dF,
                                delta_G, delta_S, scale_delta, N, maxevals, iw_ibest,
                                has_unknown_constraints, has_satisfaction_fun, UnknownFeasible, UnknownSatisfactory):
    """
    Acquisition function to minimize to get next sample in GLISp using (RBF surrogate + IDW exploration)

    Note: for the preference-based case, infeasibility is not a issue because the surrogate
    should already account for it based on the pairwise comparisons expressed (infeasibility
    is part of the decision-making process when express preferences)
    """

    d = np.sum((Xs - xs) ** 2, axis=1)

    fhat = (np.sum(rbf_xs * W)).item()  # surrogate at xs (note that rbf_xs has dimension K too)

    ii = np.where(d < 1e-12)
    if ii[0].size > 0:
        dhat = 0.
        if has_unknown_constraints:
            Ghat = UnknownFeasible[ii[0][0]]
        else:
            Ghat = 1.
        if has_satisfaction_fun:
            Shat = UnknownSatisfactory[ii[0][0]]
        else:
            Shat = 1.
    else:
        w = np.exp(-d) / d
        sw = sum(w)

        aux = 1. / sum(1. / d)
        if not scale_delta:
            # for comparision, used in the original GLISp and when N_max <= 30 in C-GLISp
            dhat = (delta_E * np.arctan(aux)).item()
        else:
            dhat = (delta_E * ((1. - N / maxevals) * np.arctan(aux * iw_ibest) + N / maxevals *
                               np.arctan(aux))).item()

        # to account for the unknown constraints
        if has_unknown_constraints:
            Ghat = (np.sum(UnknownFeasible * w) / sw).item()
        else:
            Ghat = 1.

        if has_satisfaction_fun:
            Shat = (np.sum(UnknownSatisfactory * w) / sw).item()
        else:
            Shat = 1.

    f = fhat / dF.item() - dhat
    f += delta_G * (1. - Ghat) + delta_S * (1. - Shat)

    return f


@njit
def facquisition_pref_prob_improvement(v, W, sepvalue):
    """
    Acquisition function based on probability of improvement for preference-based optimization
    """

    PHI_W = np.sum(v * W).item()

    lm1 = max(PHI_W + sepvalue, 0.0)
    l0 = max(0, PHI_W - sepvalue, -PHI_W - sepvalue)
    l1 = max(sepvalue - PHI_W, 0.0)
    c0 = 1.0
    cm1 = 1.0
    c1 = 1.0
    em1 = exp(-cm1 * lm1)
    f = -em1 / (em1 + exp(-c0 * l0) + exp(-c1 * l1))

    return f


class GLIS_base(object):
    """
    Base class of GLIS to set up the problem and provide the default values to related parameters
    """

    def __init__(self, bounds=None, n_initial_random=None,
                 alpha=1.0, delta=0.5, rbf=None, rbf_epsil=1.0,
                 svdtol=1.e-6, A=None, b=None, g=None, obj_transform=None,
                 shrink_range=True, constraint_penalty=1000,
                 feasible_sampling=True, scale_delta=False, expected_max_evals=100,
                 display=1, PSOiters=2000, PSOswarmsize=20, PSOminfunc=1e-4,
                 has_unknown_constraints=False, has_satisfaction_fun=False):

        if not isinstance(bounds, tuple) or not len(bounds) == 2:
            raise (Exception('You must provide a tuple (lb,ub) of finite lower and upper bounds'))

        lb = np.array(bounds[0]).reshape(-1)
        ub = np.array(bounds[1]).reshape(-1)

        nvars = lb.size
        if n_initial_random is None:
            n_initial_random = 2 * nvars

        if rbf is None:
            rbf = inverse_quadratic
        if isinstance(rbf, str) and rbf.lower() == 'idw':
            # None = use IDW interpolation
            useRBF = 0  # 1 = use RBFs, 0 = use IDW interpolation
        else:
            useRBF = 1

        if A is None:
            A = np.zeros((0, nvars))  # matrix A defining linear inequality constraints
        if b is None:
            b = np.zeros((0, 1))  # right hand side of constraints A*x <= b

        isLinConstrained = (b.shape[0] > 0)
        isNLConstrained = (g is not None)
        if not isLinConstrained and not isNLConstrained:
            feasible_sampling = False

        gs = g
        dd = (ub - lb).reshape(-1) / 2.0  # compute dd,d0 even if scalevars=0 so to return them
        d0 = (ub + lb).reshape(-1) / 2.0

        # Rescale problem variables in [-1,1]
        lb = -np.ones(nvars)
        ub = np.ones(nvars)

        if isLinConstrained:
            b = b - A @ d0
            A = A * dd

        if isNLConstrained:
            gs = lambda x: g(x * dd + d0)

        self.constrpenalty_value = constraint_penalty

        if isLinConstrained and isNLConstrained:
            self.constrpenalty = lambda xs: constraint_penalty * (
                    np.sum(np.maximum((A @ xs - b).reshape(-1), 0.) ** 2) +
                    np.sum(np.maximum(gs(xs), 0.) ** 2))
        elif isLinConstrained and not isNLConstrained:
            self.constrpenalty = lambda xs: constraint_penalty * (np.sum(np.maximum((A @ xs - b).reshape(-1), 0.) ** 2))
        elif not isLinConstrained and isNLConstrained:
            self.constrpenalty = lambda xs: constraint_penalty * np.sum(np.maximum(gs(xs), 0.) ** 2)
        else:
            self.constrpenalty = lambda xs: 0.

        if shrink_range:
            # possibly shrink lb,ub to constraints
            if not isNLConstrained and isLinConstrained:
                flin = np.zeros(nvars)
                for i in range(nvars):
                    flin[i] = 1.0
                    res = linprog(flin, A_ub=A, b_ub=b, bounds=np.hstack((lb.reshape(-1, 1), ub.reshape(-1, 1))))
                    if not res.success:
                        raise (Exception(res.message))
                    lb[i] = max(lb[i], res.fun)
                    flin[i] = -1.0
                    res = linprog(flin, A_ub=A, b_ub=b, bounds=np.hstack((lb.reshape(-1, 1), ub.reshape(-1, 1))))
                    ub[i] = min(ub[i], -res.fun)
                    flin[i] = 0.0

            elif isNLConstrained:
                NLpenaltyfun = lambda x: sum(np.maximum(g(x), 0) ** 2)
                if isLinConstrained:
                    LINpenaltyfun = lambda x: sum(np.maximum((A.dot(x) - b).reshape(-1), 0) ** 2)
                else:
                    LINpenaltyfun = lambda x: 0

                for i in range(0, nvars):
                    obj_fun = lambda x: x[i] + 1.0e4 * (NLpenaltyfun(x) + LINpenaltyfun(x))
                    if display < 2:
                        with contextlib.redirect_stdout(io.StringIO()):
                            z, cost = pso(obj_fun, lb, ub, swarmsize=30,
                                          minfunc=1.e-8, maxiter=2000)
                    else:
                        z, cost = pso(obj_fun, lb, ub, swarmsize=30,
                                      minfunc=1.e-8, maxiter=2000)
                    lb[i] = max(lb[i], z[i])

                    obj_fun = lambda x: -x[i] + 1.0e4 * (NLpenaltyfun(x) + LINpenaltyfun(x))

                    if display < 2:
                        with contextlib.redirect_stdout(io.StringIO()):
                            z, cost = pso(obj_fun, lb, ub, swarmsize=30,
                                          minfunc=1e-8, maxiter=2000)
                    else:
                        z, cost = pso(obj_fun, lb, ub, swarmsize=30,
                                      minfunc=1e-8, maxiter=2000)
                    ub[i] = min(ub[i], z[i])

        self.nvars = nvars  # number of optimization variables
        self.lb = lb  # lower bounds (in scaled vars)
        self.ub = ub  # upper bounds (in scaled vars)
        self.A = A.reshape(-1, nvars)  # linear constraint matrix (in scaled vars)
        self.b = b.reshape(-1)  # linear constraint rhs (in scaled vars)
        self.g = gs  # nonlinear constraint function (in scaled vars)
        self.obj_transform = obj_transform  # nonlinear transformation, minimize obj_transform(f(x)) instead of f(x)
        self.isObjTransformed = not (self.obj_transform is None)
        self.transformed_F = list()
        self.feasible_sampling = feasible_sampling  # require that initial samples are feasible
        self.isLinConstrained = isLinConstrained
        self.isNLConstrained = isNLConstrained
        self.n_initial_random = int(n_initial_random)  # number of random initial samples
        self.alpha = alpha
        self.delta = delta
        self.constraint_penalty = constraint_penalty
        self.display = display
        self.svdtol = svdtol
        self.dd = dd
        self.d0 = d0
        self.useRBF = useRBF
        self.rbf = rbf
        self.rbf_epsil = rbf_epsil
        self.scale_delta = scale_delta
        self.expected_max_evals = expected_max_evals
        self.PSOiters = PSOiters
        self.PSOswarmsize = PSOswarmsize
        self.PSOminfunc = PSOminfunc
        self.has_unknown_constraints = has_unknown_constraints
        self.has_satisfaction_fun = has_satisfaction_fun
        self.time_fun_eval = list()
        self.time_opt_acquisition = list()
        self.time_fit_surrogate = list()
        self.UnknownFeasible = list()
        self.UnknownSatisfactory = list()
        self.KnownFeasible = list()
        self.X = list()
        self.F = list()
        self.Fmin = np.inf
        self.Fmax = -np.inf
        self.iter = 0
        self.xnext = None
        self.fbest = np.inf
        self.ibest = None
        self.xbest = None
        self.fbest_seq = list()
        self.ibest_seq = list()
        self.isfeas_seq = list()
        self.isInitialized = False

    def initialize_(self):
        """
        Initialize problem and return initial set X of random samples to query:
            - X = initialize(GLIS)
            - dimension:  (GLIS.n_initial_random)-by-(GLIS.nvars)
        """

        if not self.feasible_sampling:
            Xs = 2. * lhs(self.nvars, self.n_initial_random, "m") - 1.
            if (not self.isLinConstrained and not self.isNLConstrained):
                self.KnownFeasible = list(np.ones(self.n_initial_random))
            else:
                for i in range(self.n_initial_random):
                    self.KnownFeasible.append(self.isKnownFeasible(Xs[i]))
        else:
            nn = self.n_initial_random
            nk = 0
            while (nk < self.n_initial_random):
                XXs = 2. * lhs(self.nvars, nn, "m") - 1.

                ii = np.ones(nn)
                for i in range(nn):
                    if self.isLinConstrained:
                        ii[i] = np.all(self.A @ XXs[i] <= self.b)
                    if self.isNLConstrained:
                        ii[i] = ii[i] and np.all(self.g(XXs[i]) <= 0.)

                nk = sum(ii)
                if (nk == 0):
                    nn = int(20 * nn)
                elif (nk < self.n_initial_random):
                    nn = int(np.ceil(min(20, 1.1 * self.n_initial_random / nk) * nn))
            ii = np.where(ii)
            Xs = XXs[ii[0][0:self.n_initial_random]]
            self.KnownFeasible = list(np.ones(self.n_initial_random))
        self.X = list(Xs * self.dd + self.d0)
        return Xs

    def get_delta_adpt(self, Xs, constraint_set, delta_const_default):
        """
        Obtain the adaptive exploration parameter related to unknown constraint or satisfaction function
        """

        ind = constraint_set.shape[0]
        sqr_error_feas = np.zeros(ind)
        for i in range(0, ind):
            xx = Xs[i]
            Xi = np.vstack((np.array(Xs[0:i]), np.array(Xs[i + 1:ind])))
            constraint_set_i = np.concatenate((constraint_set[0:i], constraint_set[i + 1:ind]))
            Feas_xx = constraint_set[i]
            d = np.sum((Xi - xx) ** 2, axis=-1)  # in case of repeating samples
            w = np.sum(-d) / d
            sw = sum(w)
            ghat = np.sum(constraint_set_i * w) / sw
            sqr_error_feas[i] = (ghat - Feas_xx) ** 2

        std_feas = (np.sum(sqr_error_feas) / (ind - 1)) ** (1 / 2)
        delta_adpt = (1 - std_feas) * delta_const_default

        return delta_adpt

    def isKnownFeasible(self, xs):
        """
        Check the feasibility of sample xs w.r.t known constraints
        """

        isfeas = True
        if self.isLinConstrained:
            isfeas = isfeas and np.all(self.A @ xs <= self.b)
        if self.isNLConstrained:
            isfeas = isfeas and np.all(self.g(xs) <= 0)
        return isfeas


class GLIS(GLIS_base):
    """
    Solve the global optimization problem

        min  f(x)

        s.t. lb <= x <=ub,
             A*x <=b, g(x)<=0

    using the global optimization algorithm described in [1].

     Parameters
    ----------

    bounds : tuple (1-D array,1-D array)
        Arrays of lower and upper bounds on x, lb <= x <= ub, bounds = (lb,ub)

    n_initial_random : int
        Number of generated initial random samples

    alpha: float, optional
        Hyper-parameter used in the acquisition function to promote the informativeness
        of the new sample that will be queried, related to surrogate uncertainty

    delta: float, optional
        Hyper-parameter used in the acquisition function to promote the diversity
        of the new sample that will be queried, only related to the position of the samples acquired so far

    rbf: function, optional
        RBF interpolant used to construct the surrogate. Default RBFs can be imported:

        from glis import gaussian, inverse_quadratic, multiquadric, thin_plate_spline, linear, inverse_multi_quadric

        For example, in case of Gaussian RBF the function is defined as

        rbf(x1,x2)=exp(-(rbf_epsilon*||x1-x2||_2^2))

    rbf_epsil: float, optional
        RBF parameter

    svdtol: float, optional
        tolerance used in SVD decomposition when fitting the RBF function

    A: 2D-array, optional
        Left-hand-side of known linear constraints A*x <= b imposed during acquisition

    b: 1D-array, optional
        Right-hand-side of known linear constraints A*x <= b imposed during acquisition

    g: function, optional
        Known nonlinear function on the optimization vector, g(x) <= 0 imposed during acquisition

    obj_transform: function, optional
        Nonlinear transformation of the objective function to minimize

    shrink_range: bool, optional
        If True, the given bounds `bounds` are shrunk to the bounding box of the feasible
        constrained set X = {x: A*x <= b, g(x) <= 0}

    constraint_penalty: float, optional
        penalty used to penalize the violation of the constraints A*x <= b, g(x) <= 0
        during acquisition

    feasible_sampling: bool, optional
        If True all the initial samples satisfy the constraints A*x <= b, g(x) <= 0

    scale_delta: bool, optional
        If True, scale "delta" during the iterations, as described in [3]

    expected_max_evals: int, optional
        Expected maximum number of queries (defaulted to `max_evals` when using `GLIS.solve()`

    display: int, optional
        Verbosity level (0 = none)

    PSOiters: int, optional
        Maximum number of iterations of PSO solver

    PSOswarmsize: int, optional
        Number of particles used by PSO solver

    PSOminfunc: int, optional
        Tolerance used by PSO solver

    has_unknown_constraints: bool, optional
        True if the problem has unknown constraints, as described in [2]

    has_satisfaction_fun: bool, optional
        True if the queries include a declaration that the acquired sample
        is satisfactory, as described in [2]


    Returns:
    --------
    GLIS object


    References:
    --------
    [1] A. Bemporad, "Global optimization via inverse weighting and radial basis functions,"
    Computational Optimization and Applications, vol. 77, pp. 571–595, 2020.

    [2] M. Zhu, D. Piga, and A. Bemporad, “C-GLISp: Preference-based global optimization under
    unknown constraints with applications to controller calibration,” IEEE Trans. Contr.
    Systems Technology, vol. 30, no. 3, pp. 2176–2187, Sept. 2022.

    """

    def __init__(self, bounds, *args, **kwargs):
        super(GLIS, self).__init__(bounds=bounds, *args, **kwargs)

    def initialize(self):
        """
        Initialize the problem
            - obtain the initial samples to query
            - preallocate the RBF coefficient matrix for the initial samples uf RBF surrogate is used

        return:
            self.xnext: 1D array
                the initial sample to query
        """

        Xs = self.initialize_()  # get initial samples to query

        if self.useRBF:
            self.M = np.zeros((self.n_initial_random, self.n_initial_random))
            for i in range(self.n_initial_random):
                for j in range(i, self.n_initial_random):
                    mij = self.rbf(Xs[i], Xs[j], self.rbf_epsil)
                    self.M[i, j] = mij
                    self.M[j, i] = mij

            if self.has_unknown_constraints:
                self.M_unkn = self.M.copy()

        self.xnext = self.X[0]
        self.isInitialized = True
        return self.xnext  # initial sample to query (unscaled)

    def get_rbf_weights(self, F_, M_):
        """
        Obain the RBF weights for the surrogate

        Input:
            F_: 1D array
                function evaluation at N samples
            M_: N-by-N array
                RBF coefficient matrix w.r.t N samples

        Return:
            W: 1D array
                RBF weights for N samples
        """

        # Solve M_*W = F_ using SVD
        U, dS, Vt = np.linalg.svd(M_)
        ii = np.where(dS >= self.svdtol)
        if np.any(ii):  # if ii is not empty, ii might be empty if all the current samples are infeasible
            ns = max(ii[0]) + 1
            W = Vt[0:ns, ].T * (1. / dS[0:ns].reshape(-1)) @ (U[:, 0:ns].T @ F_)
        else:
            W = None
        return W

    def update(self, f_val, feasible=True, satisfactory=True):
        """
        - Update the relevant variables w.r.t the newly queried sample
        - And then solve the optimization problem on the updated acquisition function to obtain the next point to query

        - Note:
            - randomly generated samples are always feasible wrt known constraints if self.feasible_sampling = True
            - actively generated samples are always feasible wrt known constraints

        Input:
            f_val: float
                evaluation of the objective function at x (the last queried point)
            feasible: bool
                feasibility evaluation of x (the last queried point)
                    w.r.t the unknown constraints (if exist)
            satisfactory: bool
                satisfactory evaluation of x (the last queried point)
                    w.r.t the unknown satisfactory functions (if exist)
        Return:
            self.xnext: 1D-array
                The next point to query
        """

        self.F.append(f_val)
        f0 = f_val
        if self.isObjTransformed:
            f0 = self.obj_transform(f_val)
            self.transformed_F.append(f0)
            F = np.array(self.transformed_F)
        else:
            F = np.array(self.F)

        self.UnknownFeasible.append(feasible)
        self.UnknownSatisfactory.append(satisfactory)

        x = self.xnext  # this was computed at the previous call after n_initial_random iterations

        if self.iter < self.n_initial_random:
            isfeas = bool(self.KnownFeasible[self.iter])
            self.time_opt_acquisition.append(0.)
            self.time_fit_surrogate.append(0.)
        else:
            isfeas = True  # actively generated samples are always feasible wrt known constraints
            self.KnownFeasible.append(isfeas)
        if self.has_unknown_constraints:
            isfeas = isfeas and feasible
        # if self.has_satisfaction_fun:
        #     isfeas = isfeas and satisfactory
        if isfeas and f_val < self.fbest:
            # update optimal solution, if feasible wrt to all constraints and satisfactory
            self.fbest = f_val
            self.ibest = self.iter
            self.xbest = x.copy()
        self.ibest_seq.append(self.ibest)
        self.fbest_seq.append(self.fbest)
        self.isfeas_seq.append(isfeas)
        ind_feas = [i for i, x in enumerate(self.isfeas_seq) if x == True]

        if isfeas:
            self.Fmax = max(self.Fmax, f0)
            self.Fmin = min(self.Fmin, f0)

        if self.display > 0:
            if isfeas:
                print("Iteration N = %4d, best = %7.4f, current = %7.4f, x = [" % (self.iter + 1, self.fbest, f_val),
                      end="")
            else:
                print(
                    "Iteration N = %4d, best = %7.4f, current = infeasible sample, x = [" % (self.iter + 1, self.fbest),
                    end="")
            for j in range(self.nvars):
                print("%7.4f" % x[j], end="")
                if j < self.nvars - 1:
                    print(", ", end="")
            print("]")

        if (self.iter == self.n_initial_random - 1) and self.useRBF:
            # Possibly remove rows/columns corresponding to infeasible samples
            # This step is necessary even when Unknown constraints are not present (for the case, when feasible_sampling = False)
            self.M = self.M[ind_feas, :].T[ind_feas, :].T

        if self.iter >= self.n_initial_random - 1:
            # Active sampling: prepare vector xnext to query
            Xs_all = (np.array(self.X) - self.d0) / self.dd

            delta_E = self.delta

            if self.has_unknown_constraints:
                delta_G_default = self.delta
                delta_G = self.get_delta_adpt(Xs_all, np.array(self.UnknownFeasible), delta_G_default)
            else:
                delta_G = 0.
            if self.has_satisfaction_fun:
                delta_S_default = self.delta / 2.
                delta_S = self.get_delta_adpt(Xs_all, np.array(self.UnknownSatisfactory), delta_S_default)
            else:
                delta_S = 0.

            dF = self.Fmax - self.Fmin
            if dF == -np.inf:  # no feasible samples found so far
                dF_ = np.array([1.])
            else:
                dF_ = dF.copy()

            if self.scale_delta and (len(ind_feas) > 0):
                d_ibest = np.sum(
                    (np.vstack((Xs_all[0:self.ibest], Xs_all[self.ibest + 1:self.iter + 1])) - Xs_all[self.ibest]) ** 2,
                    axis=-1)
                ii = np.where(d_ibest < 1.e-12)
                if ii[0].size > 0:
                    iw_ibest = 0.
                else:
                    iw_ibest = sum(1. / d_ibest)
            else:
                iw_ibest = 0.

            F_all = np.array(self.F)
            F = F_all[ind_feas].copy()  # only keeps values f(x) corresponding to feasible samples x
            Xs = Xs_all[ind_feas, :].copy()  # RBF or IDW only defined wrt feasible samples in Xs

            # Update RBF matrix M
            if self.iter >= self.n_initial_random:
                if self.useRBF and isfeas:
                    # N = self.iter + 1
                    N = len(ind_feas)
                    self.M = np.vstack((np.hstack((self.M, np.zeros((N - 1, 1)))), np.zeros((1, N))))
                    # Just update last row and column of M
                    for h in range(N - 1):
                        mij = self.rbf(Xs[h], Xs[N - 1], self.rbf_epsil)
                        self.M[h, N - 1] = mij
                        self.M[N - 1, h] = mij
                    self.M[N - 1, N - 1] = 1.0
                if self.useRBF and self.has_unknown_constraints:
                    N = self.iter + 1
                    self.M_unkn = np.vstack((np.hstack((self.M_unkn, np.zeros((N - 1, 1)))), np.zeros((1, N))))
                    # Just update last row and column of M
                    for h in range(N - 1):
                        mij = self.rbf(Xs_all[h], Xs_all[N - 1], self.rbf_epsil)
                        self.M_unkn[h, N - 1] = mij
                        self.M_unkn[N - 1, h] = mij
                    self.M_unkn[N - 1, N - 1] = 1.0

            t0 = time.time()
            if self.useRBF:
                W = self.get_rbf_weights(F,
                                         self.M)  # update weights using current F and matrix M (only consider the feasible samples)
            else:
                W = np.zeros(self.iter)
            self.time_fit_surrogate.append(time.time() - t0)

            # Related to unknown constraints
            F_unkn = F_all.copy()
            if self.useRBF and self.has_unknown_constraints:
                ind_infeas = [i for i, x in enumerate(self.isfeas_seq) if x == False]
                F_unkn[ind_infeas] = np.ones((len(ind_infeas))) * (
                            self.constrpenalty_value * dF_)  # for infeasible ones, penalty values are assigned to the fun. eval
                W_unkn = self.get_rbf_weights(F_unkn,
                                              self.M_unkn)  # update weights using current F and matrix M (consider all the samples)
            else:
                W_unkn = np.zeros(self.iter)  # place holder

            def acquisition(xs):
                if self.useRBF:
                    rbf_xs = self.rbf(Xs, xs, self.rbf_epsil)
                else:
                    rbf_xs = 0.0

                if self.useRBF and self.has_unknown_constraints:
                    rbf_xs_unkn = self.rbf(Xs_all, xs, self.rbf_epsil)
                else:
                    rbf_xs_unkn = 0.0

                return facquisition(xs, Xs, F, Xs_all, F_all, self.useRBF, rbf_xs, W, delta_E, dF_, delta_G, delta_S,
                                    self.scale_delta, float(self.iter), float(self.expected_max_evals),
                                    self.alpha, iw_ibest,
                                    self.has_unknown_constraints, self.has_satisfaction_fun,
                                    np.array(self.UnknownFeasible), np.array(self.UnknownSatisfactory),
                                    np.array(self.isfeas_seq), np.array(self.constrpenalty_value),
                                    W_unkn, rbf_xs_unkn, F_unkn) \
                    + dF_ * self.constrpenalty(xs)

            t0 = time.time()
            if self.display < 2:
                with contextlib.redirect_stdout(io.StringIO()):
                    z, cost = pso(acquisition, self.lb, self.ub, swarmsize=self.PSOswarmsize,
                                  minfunc=dF_ * self.PSOminfunc, maxiter=self.PSOiters)
            else:
                z, cost = pso(acquisition, self.lb, self.ub, swarmsize=self.PSOswarmsize,
                              minfunc=dF_ * self.PSOminfunc, maxiter=self.PSOiters)
            self.time_opt_acquisition.append(time.time() - t0)

            xsnext = z.T
            self.xnext = xsnext * self.dd + self.d0
            self.X.append(self.xnext)

        else:
            self.xnext = self.X[self.iter + 1]

        self.iter += 1

        return self.xnext

    def solve(self, fun, max_evals, unknown_constraint_fun=None, satisfactory_fun=None):
        """
        If the simulator/fun and the unknwn_constraint_fun, satisfactory_fun, if exist, have already be integrated with the GLIS solver,
            - use solve() to solve the problem directly

        Input:
            fun: the simulator/fun/...
                - Input: sample to query
                - Output: performance/function evaluation
            max_evals: int
                maximum number of function evaluations
            unknown_constraint_fun:
                - Input: sample to query
                - Output (bool): True if feasible; False if infeasible
            satisfactory_fun:
                - Input: sample to query
                - Output (bool): True if satisfactory; False if unsatisfactory

        Return:
            self.xbest: 1D-array
                the best x sampled
            self.fbest: float
                function evaluation at xbest
        """

        t_all = time.time()
        self.expected_max_evals = max_evals
        x = self.initialize()  # x is unscaled

        for k in range(max_evals):
            t0 = time.time()

            # evaluate fun/performance
            f_val = fun(x)

            # evaluate unknown feasibility/satisfactory, if exist, of new x
            if self.has_unknown_constraints:
                feasible = unknown_constraint_fun(x)
            else:
                feasible = True
            if self.has_satisfaction_fun:
                satisfactory = satisfactory_fun(x)
            else:
                satisfactory = True

            self.time_fun_eval.append(time.time() - t0)

            x = self.update(f_val, feasible, satisfactory)

        self.X = self.X[
                 :-1]  # it is because in prob.update, it will calculate the next point to query (the last x2 is calculated but not assessed at max_evals +1)
        self.time_total = time.time() - t_all
        return self.xbest, self.fbest


class GLISp(GLIS_base):
    """
    Solve the preference-based global optimization problem

        find feasible x* such that pi(x*,x)<=0 for all feasible x

    using the global optimization algorithm described in [1], where

        pi(x,y) = -1 if x is better than y

        pi(x,y) =  1 if x is worse than y

        pi(x,y) =  0 if x is as good as y

    and feasibility is defined by the constraints lb <= x <=ub, A*x <=b, g(x)<=0

     Parameters
    ----------
    bounds : tuple (1-D array,1-D array)
        Arrays of lower and upper bounds on x, lb <= x <= ub, bounds = (lb,ub)

    n_initial_random : int
        Number of generated initial random samples

    alpha: float, optional
        Hyper-parameter used in the acquisition function to promote the informativeness
        of the new sample that will be queried, related to surrogate uncertainty

    delta: float, optional
        Hyper-parameter used in the acquisition function to promote the diversity
        of the new sample that will be queried, only related to the position of the samples acquired so far

    rbf: function, optional
        RBF interpolant used to construct the surrogate. Default RBFs can be imported:

        from glis.rbf import gaussian, inverse_quadratic, multiquadric, thin_plate_spline, linear, inverse_multi_quadric

        For example, in case of Gaussian RBF the function is defined as

        rbf(x1,x2)=exp(-(rbf_epsilon*||x1-x2||_2^2))

    rbf_epsil: float, optional
        RBF parameter

    svdtol: float, optional
        tolerance used in SVD decomposition when fitting the RBF function

    A: 2D-array, optional
        Left-hand-side of known linear constraints A*x <= b imposed during acquisition

    b: 1D-array, optional
        Right-hand-side of known linear constraints A*x <= b imposed during acquisition

    g: function, optional
        Known nonlinear function on the optimization vector, g(x) <= 0 imposed during acquisition

    obj_transform: function, optional
        Nonlinear transformation of the objective function to minimize

    shrink_range: bool, optional
        If True, the given bounds `bounds` are shrunk to the bounding box of the feasible
        constrained set X = {x: A*x <= b, g(x) <= 0}

    constraint_penalty: float, optional
        penalty used to penalize the violation of the constraints A*x <= b, g(x) <= 0
        during acquisition

    feasible_sampling: bool, optional
        If True all the initial samples satisfy the constraints A*x <= b, g(x) <= 0

    scale_delta: bool, optional
        If True, scale "delta" during the iterations, as described in [3]

    expected_max_evals: int, optional
        Expected maximum number of queries (defaulted to `max_evals` when using `GLIS.solve()`

    display: int, optional
        Verbosity level (0 = none)

    PSOiters: int, optional
        Maximum number of iterations of PSO solver

    PSOswarmsize: int, optional
        Number of particles used by PSO solver

    PSOminfunc: int, optional
        Tolerance used by PSO solver

    has_unknown_constraints: bool, optional
        True if the problem has unknown constraints, as described in [2]

    has_satisfaction_fun: bool, optional
        True if the queries include a declaration that the acquired sample
        is satisfactory, as described in [2]

    acquisition_method : str, optional
        Acquisition method, either "surrogate" or "prob_improvement"

    epsDeltaF : float, optional
        Lower bound on the range of the surrogate function

    RBFcalibrate : bool, optional
        If True, introduce recalibration of the RBF parameter epsilon as
        epsilon*theta, where theta is chosen by cross validation during GLISp iterations

    RBFcalibrationSteps : int 1-D array, optional
        array of step indices at which recalibration must be performed

    thetas : float 1-D array, optional
        Array of values of theta tested during recalibration.

    itheta: int
        the index of the current theta in thetas

    theta: float
        value of the current theta: thetas[itheta]

    sepvalue : float, optional
        Amount of separation fhat(x1)-fhat(x2) imposed in the surrogate function fhat
        when imposing preference constraints.

    Returns:
    --------
    GLISp object

    References:
    --------
    [1] A. Bemporad, "Global optimization via inverse weighting and radial basis functions,"
    Computational Optimization and Applications, vol. 77, pp. 571–595, 2020.

    [2] A. Bemporad and D. Piga, “Active preference learning based on radial basis functions,”
    Machine Learning, vol. 110, no. 2, pp. 417–448, 2021.

    [3] M. Zhu, D. Piga, and A. Bemporad, “C-GLISp: Preference-based global optimization under
    unknown constraints with applications to controller calibration,” IEEE Trans. Contr.
    Systems Technology, vol. 30, no. 3, pp. 2176–2187, Sept. 2022.

    """

    def __init__(self, bounds, acquisition_method="surrogate", epsDeltaF=1.e-4,
                 RBFcalibrate=True, RBFcalibrationSteps=None, thetas=None, sepvalue=None, *args, **kwargs):
        super(GLISp, self).__init__(bounds=bounds, *args, **kwargs)
        if not (acquisition_method == "surrogate" or acquisition_method == "prob_improvement"):
            raise (Exception('Supported acquisition methods are "surrogate" and "prob_improvement"'))
        if (self.isLinConstrained or self.isNLConstrained) and not self.feasible_sampling:
            raise (Exception("Must generate feasible initial samples, please set 'feasible_sampling = True'"))
        if not self.obj_transform is None:
            warnings.warn("This is preference-based optimization, argument 'obj_transform' ignored")
        self.acquisition_method = acquisition_method
        self.epsDeltaF = epsDeltaF
        if not self.useRBF:
            raise (Exception("IDW not supported in GLISp, only RBF surrogates"))
        self.RBFcalibrate = RBFcalibrate
        self.sepvalue = sepvalue
        self.RBFcalibrationSteps = RBFcalibrationSteps  # array of indices k such that the RBF must be recalibrate when k samples have been compared
        self.thetas = thetas
        self.I = list()
        self.Ieq = list()

    def get_rbf_weights(self, M, n, I, Ieq, ibest):
        """
        Fit RBF satisfying comparison constraints at sampled points
            - with L1-regularization on the rbf coefficients via LP

        Optimization vector x=[beta_p;beta_m;epsil] where:
            - beta = beta_p - beta_m = rbf coefficients
            - epsil = vector of slack vars, one per constraint

        Input:
            M: n-by-n array
                coefficient matrix of the RBF surrogate w.r.t n samples
                Note: here self.M is not used since in the rbf_recalibration, the M can vary
            n: int
                number of samples considered
            I: list of a list, e.g., [[a,b], [c,a],...[f,g]]
                contains indices of the pairwise compared samples
                where [a,b] means the decision maker prefers a over b, i.e., f(x(a))<f(x(b))
            Ieq: list of a list, e.g., [[a,b], [c,a],...[f,g]]
                contains indices of the pairwise compared samples
                where [a,b] means the decision maker do not have a preference between sample x(a) and x(b), i.e., f(x(a))=f(x(b))
            ibest: int
                the index of the current best sample
                Note: here self.ibest is not used since in the rbf_recalibration, the index of ibest can vary

        Return:
            beta: 1D array
                RBF weights for n samples
        """
        sepvalue = self.sepvalue
        normalize = 0  # imposing constraints to normalize surrogate seems redundant

        m = I.shape[0]
        meq = Ieq.shape[0]
        # A = np.zeros((m + 2 * meq + m + meq, n + m + meq))
        # b = np.zeros((m + 2 * meq + m + meq, 1))
        A = np.zeros((m + 2 * meq, 2 * n + m + meq))
        b = np.zeros((m + 2 * meq, 1))
        for k in range(m):
            i = I[k][0]
            j = I[k][1]
            # f(x(i))<f(x(j))
            # sum_h(beta(h)*phi(x(i,:),x(h,:))<=sum_h(beta(h)*phi(x(j,:),x(h,:))+eps_k-sepvalue
            A[k, 0:n] = M[i, 0:n] - M[j, 0:n]
            A[k, n:2 * n] = - A[k, 0:n]
            A[k, 2 * n + k] = -1.0
            b[k] = -sepvalue

        # |f(x(i))-f(x(j))|<=comparetol
        # --> f(x(i))<=f(x(j))+comparetol+epsil
        # --> f(x(j))<=f(x(i))+comparetol+epsil
        # sum_h(beta(h)*phi(x(i,:),x(h,:))<=sum_h(beta(h)*phi(x(j,:),x(h,:))+sepvalue+epsil
        # sum_h(beta(h)*phi(x(j,:),x(h,:))<=sum_h(beta(h)*phi(x(i,:),x(h,:))+sepvalue+epsil
        for k in range(0, meq):
            i = Ieq[k][0]
            j = Ieq[k][1]
            A[m + 2 * k, 0:n] = M[i, 0:n] - M[j, 0:n]
            A[m + 2 * k, n:2 * n] = - A[m + 2 * k, 0:n]
            A[m + 2 * k, 2 * n + m + k] = -1.0
            b[m + 2 * k] = sepvalue
            A[m + 2 * k + 1, 0:n] = - A[m + 2 * k, 0:n]
            A[m + 2 * k + 1, n:2 * n] = A[m + 2 * k + 1, 0:n]
            A[m + 2 * k + 1, 2 * n + m + k] = -1.0
            b[m + 2 * k + 1] = sepvalue

        if normalize:
            # Add constraints to avoid trivial solution surrogate=flat:
            #    sum_h(beta.*phi(x(ibest,:),x(h,:))) = 0
            #    sum_h(beta.*phi(x(ii,:),x(h,:))) = 1

            # Look for sample where function is worse,i.e., f(ii) is largest
            ii = I[0][1]
            for k in range(0, m):
                if I[k][0] == ii:
                    ii = I[k][1]
            Aeq = np.zeros((2, 2 * n + m + meq))
            beq = np.zeros((2, 1))
            Aeq[0, 0:n] = M[ibest, 0:n]
            Aeq[0, n:2 * n] = - Aeq[0, 0:n]
            Aeq[1, 0:n] = M[ii, 0:n]
            Aeq[1, n:2 * n] = - Aeq[1, 0:n]
            beq[0] = 0.0
            beq[1] = 1.0
        else:
            Aeq = np.zeros((0, 2 * n + m + meq))
            beq = np.zeros((0, 1))

        c = 1.e-6 * np.ones((2 * n + m + meq, 1))  # L1-regularization = 1.e-6

        # Penalty on constraint violations (more penalty on violations involving xbest)
        c[2 * n:2 * n + m + meq] = 1.0
        if I.size > 0:
            c[2 * n + np.where(np.any(I == ibest, axis=1))[0]] = 10.0
        if Ieq.size > 0:
            c[2 * n + np.where(np.any(Ieq == ibest, axis=1))[0]] = 10.0

        # In scipy.optimize.linprog by default all variable must be nonnegative.
        # bounds = np.zeros((2*n+m+meq,2)); bounds[:,1]=np.inf # redundant
        res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, method='highs')

        try:
            beta = res.x[0:n] - res.x[n:2 * n]
        except:
            c[:2 * n] = 1.e-3
            res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, method='highs')
            if res.x is None:
                raise (Exception(res.message))
            else:
                beta = res.x[0:n] - res.x[n:2 * n]

        # if not res.success:
        #     raise (Exception(res.message))
        return beta

    def rbf_calibrate(self, Xs):
        """
        Calibrate scaling of epsil parameter as epsil = theta*self.rbf_epsil in RBF by cross-validation

        Input:
            Xs: N-by-nvars array
                Samples used for cross-validation (N: number of samples)

        Return:
            None

        Updated:
            self.itheta
            self.theta
        """

        N = self.iter
        ibest = self.ibest
        itheta = self.itheta
        thetas = self.thetas
        sepvalue = self.sepvalue
        I = np.array(self.I)
        Ieq = np.array(self.Ieq)
        MM = self.MM

        if self.display > 0:
            print("Recalibrating RBF: ", end='')

        # MM is a list of length numel(thetas) of 2D matrices
        # used to save previously computed RBF values
        nth = thetas.size
        imax = 0
        successmax = -1

        for k in range(nth):
            epsilth = self.rbf_epsil * thetas[k]

            # Update matrix containing RBF values for all thetas
            if not (k == itheta):  # RBF values already computed for current theta, no need to update
                iM = MM[k].shape[0]  # current dimension of MM[k]
                MM[k] = np.vstack((np.hstack((MM[k], np.zeros((iM, N + 1 - iM)))), np.zeros((N + 1 - iM, N + 1))))
                for j in range(iM, N + 1):
                    for h in range(N + 1):
                        MM[k][j, h] = self.rbf(Xs[j, :], Xs[h, :], epsilth)
                        MM[k][h, j] = MM[k][j, h]

            Ncomparisons = 0
            success = 0

            for i in range(N + 1):  # remove each sample but the current best from the set for cross-validation
                if not (i == ibest):
                    Xsi = np.vstack((Xs[0:i], Xs[i + 1:N + 1]))
                    # ibest may change after removing one sample
                    if ibest > i:
                        newibest = ibest - 1
                    else:
                        newibest = ibest

                    Ii = np.empty((0, 2)).astype('int')
                    ni = 0
                    for j in range(I.shape[0]):
                        if not (I[j, 0] == i) and not (I[j, 1] == i):
                            Ii = np.vstack((Ii, I[j, :]))
                            if I[j, 0] > i:
                                Ii[ni, 0] = Ii[ni, 0] - 1
                            if I[j, 1] > i:
                                Ii[ni, 1] = Ii[ni, 1] - 1
                            ni = ni + 1

                    Ieqi = np.empty((0, 2)).astype('int')
                    ni = 0
                    for j in range(Ieq.shape[0]):
                        if not (Ieq[j, 0] == i) and not (Ieq[j, 1] == i):
                            Ieqi = np.vstack((Ieqi, Ieq[j, :]))
                            if Ieq[j, 0] > i:
                                Ieqi[ni, 0] = Ieqi[ni, 0] - 1
                            if Ieq[j, 1] > i:
                                Ieqi[ni, 1] = Ieqi[ni, 1] - 1
                            ni = ni + 1

                    ind = list(range(0, i)) + list(range(i + 1, N + 1))
                    Mi = MM[k][ind][:, ind]

                    Wi = self.get_rbf_weights(Mi, N, Ii, Ieqi, newibest)
                    # Compute RBF @X[i,:]
                    FH = np.zeros(N + 1)
                    FH[ind] = np.dot(Mi, Wi).reshape(-1)  # rbf at samples

                    v = self.rbf(Xsi[0:N, :], Xs[i, :], epsilth)
                    FH[i] = np.sum(v * Wi)

                    # Cross validation
                    for j in range(I.shape[0]):
                        if (I[j, 0] == i) or (I[j, 1] == i):
                            Ncomparisons = Ncomparisons + 1
                            i1 = I[j, 0]
                            i2 = I[j, 1]
                            if FH[i1] <= FH[i2] - sepvalue:
                                success = success + 1

                    for j in range(Ieq.shape[0]):
                        if (Ieq[j, 0] == i) or (Ieq[j, 1] == i):
                            Ncomparisons = Ncomparisons + 1
                            i1 = Ieq[j, 0]
                            i2 = Ieq[j, 1]
                            if np.abs(FH[i1] - FH[i2]) <= sepvalue:
                                success = success + 1
            if (self.display > 0):
                print(".", end='')

            success = success / Ncomparisons * 100.0
            # NOTE: normalization is only for visualization purposes

            # Find theta such that success is max, and closest to 1 among maximizers
            if (success > successmax) or (success == successmax and
                                          (thetas[k] - 1) ** 2 < (thetas[imax] - 1) ** 2):
                imax = k
                successmax = success

        self.itheta = imax
        self.theta = thetas[self.itheta]

        if self.display > 0:
            print(" done.")

        return

    def initialize(self):
        """
        Initialize the problem
            - obtain the initial samples to query
            - preallocate the RBF coefficient matrix for the initial samples

        return:
            self.xbest, self.xnext: first pair of samples to compare with
        """

        Xs = self.initialize_()  # get initial samples to query

        if self.sepvalue is None:
            self.sepvalue = 1. / self.expected_max_evals

        if self.RBFcalibrationSteps is None:
            n = self.n_initial_random
            self.RBFcalibrationSteps = np.array([n, n + round((self.expected_max_evals - n) / 4),
                                                 n + round((self.expected_max_evals - n) / 2),
                                                 n + round(3 * (self.expected_max_evals - n) / 4)])
        # remove self-calibration steps smaller than n_initial_random
        self.RBFcalibrationSteps = self.RBFcalibrationSteps[np.where(self.RBFcalibrationSteps >= n)[0]]
        if self.thetas is None:
            thetas = np.logspace(-1, 1, 11, False)
            self.thetas = thetas[:-1]
        self.itheta = np.argmin(np.abs(self.thetas - 1.))  # start with the value of theta closest to 1

        self.theta = self.thetas[self.itheta]

        self.MM = [np.empty((0, 0))] * self.thetas.size

        epsilth = self.rbf_epsil * self.theta
        self.MM[self.itheta] = np.zeros((self.n_initial_random, self.n_initial_random))
        M = self.MM[self.itheta]
        # Note: contrary to GLIS, we keep also infeasible samples in RBF matrix here
        for i in range(self.n_initial_random):
            for j in range(i, self.n_initial_random):
                mij = self.rbf(Xs[i], Xs[j], epsilth)
                M[i, j] = mij
                M[j, i] = mij

        # we arbitrarily consider the current best = first sample Xs[0] and
        # will ask for comparing Xs[0] wrt Xs[1] next:
        self.xbest = Xs[0] * self.dd + self.d0
        self.ibest = 0
        self.ibest_seq.append(self.ibest)
        self.xnext = Xs[1] * self.dd + self.d0
        self.iter = 1

        return self.xbest, self.xnext

    def update(self, pref_val, feasible=True, satisfactory=True):
        """
        - Update the relevant variables w.r.t the newly queried sample
        - And then solve the optimization problem on the updated acquisition function to obtain the next point to query

        - Note:
            - randomly generated samples are always feasible wrt known constraints if self.feasible_sampling = True
            - actively generated samples are always feasible wrt known constraints

        Input:
            pref_val: int
                pairwise comparison result w.r.t x (the last queried point) and current best point
            feasible: bool
                feasibility evaluation of x (the last queried point)
                    w.r.t the unknown constraints (if exist)
            satisfactory: bool
                satisfactory evaluation of x (the last queried point)
                    w.r.t the unknown satisfactory functions (if exist)
        Return:
            self.xnext: 1D-array
                The next point to query
        """

        x = self.xnext  # this was computed at the previous call after n_initial_random iterations
        N = self.iter  # current sample being examined

        if self.iter < self.n_initial_random:
            isfeas = bool(self.KnownFeasible[self.iter])
            self.time_opt_acquisition.append(0.)
            self.time_fit_surrogate.append(0.)
        else:
            isfeas = True  # actively generated samples are always feasible wrt known constraints
            self.KnownFeasible.append(isfeas)
        if self.has_unknown_constraints:
            isfeas = isfeas and feasible

        self.isfeas_seq.append(isfeas)
        self.UnknownFeasible.append(feasible)
        self.UnknownSatisfactory.append(satisfactory)

        if pref_val == -1:  # the expert has decided the preference, no matter what are unknown constraints/satisfactory values
            # update optimal solution
            self.I.append([N, self.ibest])
            self.ibest = N
            self.xbest = x.copy()
        else:
            if pref_val == 1:
                self.I.append([self.ibest, N])
            else:
                self.Ieq.append([N, self.ibest])
        self.ibest_seq.append(self.ibest)

        if self.display > 0:
            if self.ibest == N:
                txt = '(***improved x!)'
            else:
                txt = '(no improvement)'
            print("Query #%3d %s: testing x = [" % (N, txt), end="")
            for j in range(self.nvars):
                print("%7.4f" % x[j], end="")
                if j < self.nvars - 1:
                    print(", ", end="")
            print("]")

        if N >= self.n_initial_random - 1:
            # Active sampling: prepare vector xnext to query
            Xs = (np.array(self.X) - self.d0) / self.dd

            if self.RBFcalibrate and np.isin(N + 1, self.RBFcalibrationSteps):
                self.rbf_calibrate(Xs)

            delta_E = self.delta

            if self.has_unknown_constraints:
                delta_G_default = self.delta
                delta_G = self.get_delta_adpt(Xs, np.array(self.UnknownFeasible), delta_G_default)
            else:
                delta_G = 0.
            if self.has_satisfaction_fun:
                delta_S_default = self.delta / 2.
                delta_S = self.get_delta_adpt(Xs, np.array(self.UnknownSatisfactory), delta_S_default)
            else:
                delta_S = 0.

            dF = self.Fmax - self.Fmin

            M = self.MM[self.itheta]  # current RBF matrix

            t0 = time.time()
            # update weights associated with RBF matrix M and current preference info
            W = self.get_rbf_weights(M, N + 1, np.array(self.I), np.array(self.Ieq), self.ibest)

            self.time_fit_surrogate.append(time.time() - t0)
            FH = M @ W  # surrogate at current samples
            dF = max(np.max(FH) - np.min(FH), self.epsDeltaF)

            if self.scale_delta:
                d_ibest = np.sum((np.vstack((Xs[0:self.ibest], Xs[self.ibest + 1:N + 1])) - Xs[self.ibest]) ** 2,
                                 axis=-1)
                ii = np.where(d_ibest < 1.e-12)
                if ii[0].size > 0:
                    iw_ibest = 0.
                else:
                    iw_ibest = sum(1. / d_ibest)
            else:
                iw_ibest = 0.

            if self.acquisition_method == 'prob_improvement':
                # xs_best=(self.xbest-self.d0)/self.dd # scaled current best solution
                rbf_xbest = self.MM[self.itheta][
                    self.ibest]  # same as self.rbf(Xs, xs_best, self.rbf_epsil * self.theta)

                def acquisition(xs):
                    rbf_xs = self.rbf(Xs, xs.T, self.rbf_epsil * self.theta)
                    v = rbf_xs - rbf_xbest
                    return facquisition_pref_prob_improvement(v, W, self.sepvalue) + self.constrpenalty(xs)
            else:
                def acquisition(xs):
                    rbf_xs = self.rbf(Xs, xs.T, self.rbf_epsil * self.theta)
                    return facquisition_pref_surrogate(xs.T, Xs, rbf_xs, W, delta_E, dF, delta_G, delta_S,
                                                       self.scale_delta, float(N), float(self.expected_max_evals),
                                                       iw_ibest,
                                                       self.has_unknown_constraints, self.has_satisfaction_fun,
                                                       np.array(self.UnknownFeasible),
                                                       np.array(self.UnknownSatisfactory)) \
                        + self.constrpenalty(xs.T)

            t0 = time.time()
            if self.display < 2:
                with contextlib.redirect_stdout(io.StringIO()):
                    z, cost = pso(acquisition, self.lb, self.ub, swarmsize=self.PSOswarmsize,
                                  minfunc=dF * self.PSOminfunc, maxiter=self.PSOiters)
            else:
                z, cost = pso(acquisition, self.lb, self.ub, swarmsize=self.PSOswarmsize,
                              minfunc=dF * self.PSOminfunc, maxiter=self.PSOiters)
            self.time_opt_acquisition.append(time.time() - t0)

            xsnext = z.T
            self.xnext = xsnext * self.dd + self.d0
            self.X.append(self.xnext)

            # Update RBF matrix M
            N += 1
            epsilth = self.rbf_epsil * self.theta
            self.MM[self.itheta] = np.vstack(
                (np.hstack((self.MM[self.itheta], np.zeros((N, 1)))), np.zeros((1, N + 1))))
            M = self.MM[self.itheta]
            # Just update last row and column of M
            for h in range(N):
                mij = self.rbf(Xs[h], xsnext, epsilth)
                M[h, N] = mij
                M[N, h] = mij
            M[N, N] = 1.0
        else:
            self.xnext = self.X[self.iter + 1]

        self.iter += 1

        return self.xnext

    def solve(self, pref_fun, max_prefs, unknown_constraint_fun=None, satisfactory_fun=None):
        """
        If the pref_fun (decision-maker process) and the unknwn_constraint_fun, satisfactory_fun, if exist, have already be integrated with the GLISp solver,
            - use solve() to solve the problem directly

        Note: Here max_prefs = number of preferences expressed on max_prefs+1 samples

        Input:
            pref_fun: decision-maker process
                - Input: two samples to compare with
                - Output (int): preference evaluation {-1, 0, 1}
            max_prefs: int
                maximum number of pairwise comparisons
            unknown_constraint_fun:
                - Input: sample to query
                - Output (bool): True if feasible; False if infeasible
            satisfactory_fun:
                - Input: sample to query
                - Output (bool): True if satisfactory; False if unsatisfactory

        Return:
            self.xbest: 1D-array
                the best x sampled
        """

        t_all = time.time()
        self.expected_max_evals = max_prefs + 1
        xbest, x = self.initialize()  # x, self.xbest are unscaled. Initially, xbest is always the first random sample

        # Is current best feasible/satisfactory wrt unknown constraints/satisfactory function?
        if self.has_unknown_constraints:
            feasible = unknown_constraint_fun(xbest)
        else:
            feasible = True
        if self.has_satisfaction_fun:
            satisfactory = satisfactory_fun(xbest)
        else:
            satisfactory = True

        self.UnknownFeasible.append(feasible)
        self.UnknownSatisfactory.append(satisfactory)

        for k in range(max_prefs):
            t0 = time.time()

            # evaluate preference
            pref_val = pref_fun(x, xbest)

            # evaluate unknown feasibility/satisfactory, if exist, of new x
            if self.has_unknown_constraints:
                feasible = unknown_constraint_fun(x)
            else:
                feasible = True
            if self.has_satisfaction_fun:
                satisfactory = satisfactory_fun(x)
            else:
                satisfactory = True

            self.time_fun_eval.append(time.time() - t0)

            x = self.update(pref_val, feasible, satisfactory)
            xbest = self.xbest

        self.X = self.X[
                 :-1]  # it is because in prob.update, it will calculate the next point to query (the last x2 is calculated at max_evals +1)
        self.time_total = time.time() - t_all
        return self.xbest

    def rbf_recalibrate(self):
        # Method to force recalibration of RBF's epsilon = theta*self.rbf_epsilon outside solve()
        if self.iter < self.n_initial_random:
            warnings.warn("Recalibration only possible after n_initial_random = %d steps." % self.n_initial_random)
            return self.theta

        Xs = (np.array(self.X) - self.d0) / self.dd
        self.rbf_calibrate(Xs)
        return self.theta
