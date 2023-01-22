# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package
#
# Authors: A. Bemporad, M. Zhu

import unittest
import numpy as np
from glis.solvers import GLIS, GLISp


class Test_glis_glisp(unittest.TestCase):

    def test_solve_incremental(self):
        """
        test if GLIS/GLISp prob.solve gives same results as loops with GLIS/GLISp prob.update
        """
        GLIS_test = True
        GLISp_test = True

        # Camel six-humps function
        lb = np.array([-2.0, -1.0])
        ub = np.array([2.0, 1.0])
        fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2 +
                         x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
        xopt0 = np.array([[0.0898, -0.0898], [-0.7126, 0.7126]])  # unconstrained optimizers, one per column
        fopt0 = -1.0316  # unconstrained optimum
        max_evals = 60
        n_initial_random = 10

        if GLIS_test:
            key = 2
            np.random.seed(key)  # rng default for reproducibility
            ####################################################################################
            print("Solve the problem by feeding the simulator/fun directly into the GLIS solver")
            # Solve global optimization problem
            prob1 = GLIS(bounds=(lb, ub), n_initial_random=n_initial_random)
            xopt1, fopt1 = prob1.solve(fun, max_evals)
            X1 = np.array(prob1.X)
            ##########################################

            np.random.seed(key)  # reset seed
            ####################################################################################
            print("Solve the problem incrementally (i.e., provide the function evaluation at each iteration)")
            # solve same problem, but incrementally
            prob2 = GLIS(bounds=(lb, ub), n_initial_random=n_initial_random)
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

        if GLISp_test:
            ##########################################
            # Smart synthetic preference function that avoids recomputing the same value
            # fun(xbest) multiple times:
            comparetol = 1e-4
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
            #########################
            max_prefs = max_evals - 1
            #########################
            key = 2
            np.random.seed(key)  # rng default for reproducibility
            ####################################################################################
            print("Solve the problem by feeding the  preference expression step directly into the GLISp solver")
            # Solve global optimization problem
            pref_fun_smart.X = list()  # initialize preference function
            pref_fun_smart.F = list()
            prob3 = GLISp(bounds=(lb, ub), n_initial_random=n_initial_random)
            xopt3 = prob3.solve(pref_fun_smart, max_prefs)
            X3 = np.array(prob3.X)
            ##########################################

            np.random.seed(key)  # reset seed
            ####################################################################################
            print("Solve the problem incrementally (i.e., provide the preference at each iteration)")
            # solve same problem, but incrementally
            pref_fun_smart.X = list()  # initialize preference function
            pref_fun_smart.F = list()
            prob4 = GLISp(bounds=(lb, ub), n_initial_random=n_initial_random)
            xbest4, x4 = prob4.initialize()  # get first two random samples
            for k in range(max_prefs):
                pref = pref_fun_smart(x4, xbest4)  # evaluate preference
                x4 = prob4.update(pref)
                xbest4 = prob4.xbest
            X4 = np.array(prob4.X[:-1])
            ##########################################
            self.assertEqual(prob3.ibest_seq[:10], prob4.ibest_seq[:10])



    def test_feasible_samples_known_const(self):
        """
        test if all the samples are feasible w.r.t known constraints when feasible_sampling = True
        test if the active learning samples are feasible w.r.t known constraints when feasible_sample = False
        """
        GLIS_test = True
        GLISp_test = True

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

        if GLIS_test:
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

        if GLISp_test:
            # define the synthetic preference function
            comparetol = 1e-4
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
            #########################
            max_prefs = max_evals - 1
            #########################

            key = 2
            np.random.seed(key) #rng default for reproducibility
            ####################################################################################
            print("Solve the problem 'camelsixhumps' with feasible initial sample")
            # Solve global optimization problem
            pref_fun_smart.X = list()  # initialize preference function
            pref_fun_smart.F = list()
            prob3 = GLISp(bounds=(lb, ub), g=g, n_initial_random=n_initial_random)
            xopt_const1 = prob3.solve(pref_fun_smart, max_prefs)
            X3 = np.array(prob3.X)
            kn_feas_calculated_3 = list()
            for i in range(max_evals):
                kn_feas_calculated_3.append(g(X3[i])<1.0e-3)  # numerical tolerance
            ##########################################
            self.assertEqual(prob3.isfeas_seq.count(True), kn_feas_calculated_3.count(True))


if __name__ == '__main__':
    unittest.main()
