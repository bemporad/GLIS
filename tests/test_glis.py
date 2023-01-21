# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package
#
# Authors: A. Bemporad, M. Zhu

import unittest
import numpy as np
from glis.solvers import GLIS, GLISp


class Test_glis(unittest.TestCase):

    def test_solve_incremental(self):
        """
        test if GLIS.solve gives same results as loops with GLIS.update
        """

        # Camel six-humps function
        lb = np.array([-2.0, -1.0])
        ub = np.array([2.0, 1.0])
        fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                         x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
        xopt0 = np.array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
        fopt0 = -1.0316  # unconstrained optimum
        max_evals = 60

        key = 2
        np.random.seed(key)  # rng default for reproducibility
        ####################################################################################
        print("Solve the problem by feeding the simulator/fun directly into the GLIS solver")
        # Solve global optimization problem
        prob1 = GLIS(bounds=(lb, ub), n_initial_random=10)
        xopt1, fopt1 = prob1.solve(fun, max_evals)
        X1 = np.array(prob1.X)
        ##########################################

        np.random.seed(key)  # reset seed
        ####################################################################################
        print("Solve the problem incrementally (i.e., provide the function evaluation at each iteration)")
        # solve same problem, but incrementally
        prob2 = GLIS(bounds=(lb, ub), n_initial_random=10)
        x2 = prob2.initialize()
        for k in range(max_evals):
            f = fun(x2)
            x2 = prob2.update(f)
        X2 = np.array(prob2.X[:-1])  # it is because in prob.update, it will calculate the next point to query (the last x2 is calculated at max_evals +1)
        xopt2 = prob2.xbest
        fopt2 = prob2.fbest
        ##########################################
        # assert np.linalg.norm(X1-X2)==0.0 and np.all(xopt1==xopt2) and fopt1==fopt2

        self.assertEqual(X1.tolist(), X2.tolist())
        self.assertEqual(xopt1.tolist(), xopt2.tolist())
        self.assertEqual(fopt1, fopt2)

    def test_feasible_samples_known_const(self):
        """
        test if all the samples are feasible w.r.t known constraints when feasible_sampling = True
        test if the active learning samples are feasible w.r.t known constraints when feasible_sample = False
        """

        # Camel six-humps function
        lb = np.array([-2.0, -1.0])
        ub = np.array([2.0, 1.0])
        fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                         x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
        xopt0 = np.array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
        fopt0 = -1.0316  # unconstrained optimum
        max_evals = 60
        n_initial_random = 10

        g = lambda x: np.array([x[0] ** 2 + (x[1] + 0.1) ** 2 - .5])


        key = 2
        np.random.seed(key) #rng default for reproducibility
        ####################################################################################
        print("Solve the problem 'camelsixhumps' with feasible initial sample")
        # Solve global optimization problem
        prob1 = GLIS(bounds=(lb, ub), g=g, n_initial_random=n_initial_random,feasible_sampling=True)
        xopt_const1, fopt_const1 = prob1.solve(fun, max_evals)
        X1 = prob1.X
        kn_feas_calculated = list()
        for i in range(max_evals):
            kn_feas_calculated.append(g(X1[i])<1.0e-3)  # numerical tolerance
        ##########################################
        self.assertEqual(prob1.isfeas_seq.count(True), kn_feas_calculated.count(True))


        np.random.seed(key)  # rng default for reproducibility
        ####################################################################################
        print("Solve the problem 'camelsixhumps' with infeasible initial sample")
        prob2 = GLIS(bounds=(lb, ub), g=g, n_initial_random=n_initial_random, feasible_sampling=False)
        xopt_const2, fopt_const2 = prob2.solve(fun, max_evals)
        X2 = prob2.X
        kn_feas_calculated_2 = list()
        for i in range(max_evals):
            kn_feas_calculated_2.append(g(X2[i])<1.0e-3)  # numerical tolerance
        ##########################################
        self.assertEqual(prob2.isfeas_seq.count(True), kn_feas_calculated_2.count(True))


if __name__ == '__main__':
    unittest.main()
