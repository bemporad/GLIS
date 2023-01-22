# GLIS / GLISp / C-GLISp

<img src="http://cse.lab.imtlucca.it/~bemporad/glis/images/glis-logo.png" alt="drawing" width=40%/>

# Contents

* [Package description](#description)
    * [GLIS](#glis)
    * [GLISp](#glisp)
    * [C-GLIS(p)](#cglisp)

* [Installation](#install)

* [Basic usage](#basic-usage)
    * [Global optimization (GLIS)](#basic-glis)
    * [Preference-based optimization (GLISp)](#basic-glisp)

* [Advanced options](#advanced)
    * [Surrogate function](#surrogate)
    * [Acquisition function](#acquisition)
    * [RBF recalibration](#recalibration)
    * [Unknown constraints and satisfactory samples](#unknown)
    * [Objective function transformation](#transformation)
    * [Other options](#options)

* [Contributors](#contributors)

* [Citing GLIS](#bibliography)

* [License](#license)


<a name="description"></a>
## Package description 

**GLIS** is a package for finding the global (**GL**) minimum of a function that is expensive to evaluate, possibly under constraints, using inverse (**I**) distance weighting and surrogate (**S**) radial basis functions. Compared to Bayesian optimization, GLIS is very competitive and often computationally lighter.

The package implements two main algorithms, described here below.

<a name="glis"></a>
### GLIS

The GLIS algorithm solves the following constrained derivative-free global optimization problem

$$\min_x  f(x)$$

$$l \leq x\leq u$$

$$Ax \leq b,\ g(x)\leq 0$$

The approach is particularly useful when $f(x)$ is time-consuming to evaluate, as it attempts at minimizing the number of function evaluations by actively learning a surrogate of $f$. 

Finite bounds $l \leq x\leq u$ are required to limit the search within a bounded set, the remaining constraints are optional.

The algorithm is based on the following paper:

<a name="cite-Bem20"><a>
> [1] A. Bemporad, "[Global optimization via inverse weighting and radial basis functions](http://cse.lab.imtlucca.it/~bemporad/publications/papers/coap-glis.pdf)," *Computational Optimization and Applications*, vol. 77, pp. 571–595, 2020. [[bib entry](#ref1)]


<a name="glisp"></a>
### GLISp

GLISp solves global optimization problems in which the function $f$ cannot be evaluated but, given two samples $x$, $y$, it is possible to query whether $f(x)$ is better or worse than $f(y)$. More generally, one can only evaluate a *preference function* $\pi(x,y)$

<p align="center">
$\pi(x,y) = -1$ if $x$ is better than $y$
</p>

<p align="center">
$\pi(x,y) = 1$ if $x$ is worse than $y$
</p>

<p align="center">
$\pi(x,y) = 0$ if $x$ is as good as $y$,
</p>

and want to solve the following preference-based optimization problem:

<p align="center">
find $x^*$ such that $\pi(x^*,x)\leq 0$  $\ \forall x$
</p>

with $x^*,x$ satisfying the constraints $l \leq x\leq u$, 
and, if present, $Ax \leq b$, $g(x)\leq 0$.

GLISp is particularly useful to solve optimization problems that involve human assessments. In fact, there is no need to explicitly quantify an *objective function* $f$, which instead remains unexpressed in the head of the decision-maker determining the preferences. A typical example is solving a multi-objective optimization problem where the exact priority of each objective is not clear.

The algorithm is based on the following paper:

<a name="cite-BP21"><a>
> [2] A. Bemporad, D. Piga, "[Active preference learning based on radial basis functions](http://cse.lab.imtlucca.it/~bemporad/publications/papers/mlj_glisp.pdf)," *Machine Learning*, vol. 110, no. 2, pp. 417-448, 2021. [[bib entry](#ref2)]

<a name="cglisp"></a>
### C-GLISp

C-GLISp is an extension of GLIS and GLISp to handle *unknown* constraints on $x$, $x\in  X_u$, where the shape of the set $X_u$ is completely unknown and one can only query whether a certain $x\in X_u$ or $x\not\in X_u$.
A typical case is when $f(x)$ is the result of an experiment or simulation, parameterized by $x$, and one labels $x\not\in X_u$ if the experiment could not be executed. The algorithm also supports labeling samples $x$ as *satisfactory* or not, for example an experiment could be carried out but the outcome was not considered satisfactory. Both additional information (feasibility with respect to unknown constraints and satisfaction) are used by GLIS or GLISp to drive the search of the optimal solution.

The algorithm is based on the following paper:

<a name="cite-ZPB22"><a>
> [3] M. Zhu, D. Piga, A. Bemporad, "[C-GLISp: Preference-based global optimization under unknown constraints with applications to controller calibration](http://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeecst-c-glisp.pdf),” *IEEE Trans. Contr. Systems Technology*, vol. 30, no. 3, pp. 2176–2187, Sept. 2022. [[bib entry](#ref3)]

<a name="install"></a>
## Installation

~~~python
pip install glis
~~~

A MATLAB version of GLIS/GLISp is also available for download [here](http://cse.lab.imtlucca.it/~bemporad/glis).


<a name="basic-usage"></a>
## Basic usage

<a name="basic-glis"></a>
### Global optimization (GLIS)

Minimize a function $f$ of a vector $x\in\mathbb{R}^n$
within the finite bounds *lb* $\leq x\leq$ *ub*:

~~~python
from glis.solvers import GLIS
prob = GLIS(bounds=(lb, ub), n_initial_random=10) # initialize GLIS object
xopt, fopt = prob.solve(fun, max_evals)  # solve optimization problem
~~~

where `fun` is a Python function that, given a sample to query, returns $f(x)$. For example, `fun` can be a  function invoking a simulator and returning the key performance index to minimize. The parameter `n_initial_random` is the number of random samples taken at initialization by Latin Hypercube Sampling (LHS), and `max_evals` is the total allowed budget of function evaluations. The code returns the optimizer `xopt` and the corresponding minimum value `fopt=fun(xopt)` found.

If it becomes possible to obtain additional samples after running the optimization, the latter can be continued as follows:

~~~python
x = prob.xnext     # next sample to query
f = fun(x)         # function evaluation
x = prob.update(f) # update GLIS object
xopt = prob.xbest  # updated optimizer
fopt = prob.fbest  # updated optimum
~~~

Alternatively, for a full step-by-step optimization without explicitly passing the function handle `fun` to GLIS, use the following code structure to solve the problem step by step:

~~~python
from glis.solvers import GLIS
prob = GLIS(bounds=(lb, ub), n_initial_random=10)  # initialize GLIS object
x = prob.initialize()  # get first sample to query
for k in range(max_evals):
    f = fun(x)
    x = prob.update(f)
xopt = prob.xopt # final optimizer
fopt = prob.fopt # final optimum
~~~

#### Example

Minimize the *camel-six-humps* function 

$$f(x_1,x_2) = \left(4 - 2.1x_1^2 + \frac{x_1^4}{3}\right)
	     x_1^2+x_1x_2+4(x_2^2-1)x_2^2$$
	     
for $-2\leq x_1\leq 2$, $-1\leq x_2\leq 1$. The function has the global minimum $f(x^*) = -1.0316$ attained at $x^* = (0.0898, -0.7126)$ and $x^*= (-0.0898, 0.7126)$. GLIS minimizes $f(x)$ as follows:

~~~python
lb = np.array([-2.0, -1.0])
ub = np.array([2.0, 1.0])
fun = lambda x: ((4.0 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * 
    x[0] ** 2 + x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)
prob = GLIS(bounds=(lb, ub), n_initial_random=10)
xopt, fopt = prob.solve(fun, max_evals=60)
~~~

In this case we obtain `xopt` = [0.09244155, -0.7108227], `fopt` =-1.0316. Note that the final result depends on the outcome of the initial random sampling phase, so final values found by GLIS may differ from one run to another.

The sequence `xseq` of samples acquired, the corresponding
function values `fseq`, and the sequence *fbest\_seq* of 
best values found during the iterations can be retrieved as follows:

~~~python
xseq=np.array(prob.X)
fseq=np.array(prob.F)
fbest_seq = prob.fbest_seq
~~~

Here below is a plot of the samples `xseq` and best values `fbest_seq` found by GLIS:

<img src="http://cse.lab.imtlucca.it/~bemporad/glis/images/glis-1.png" alt="drawing" width=100%/>

The yellow stars are the initial samples randomly generated by LHS, the green stars are the samples queried by GLIS during the active learning stage, the blue circles are the known global optimal solutions, and the red star is the optimizer identified by GLIS.

Next, we add linear constraints $Ax\leq b$ and the nonlinear constraint $g(x)=x_1^2+(x_2+0.1)^2-\frac{1}{2}\leq 0$

~~~python
    A = np.array([[1.6295, 1],[-1, 4.4553],[-4.3023, -1],[-5.6905, -12.1374],[17.6198, 1]])
    b = np.array([3.0786, 2.7417, -1.4909, 1, 32.5198])
    g = lambda x: np.array([x[0] ** 2 + (x[1] + 0.1) ** 2 - .5])
    prob = GLIS(bounds=(lb, ub), A=A, b=b, g=g, n_initial_random=10)
    xopt, fopt = prob.solve(fun, max_evals)
~~~

GLIS determines now a new optimizer within the given constraints:

<img src="http://cse.lab.imtlucca.it/~bemporad/glis/images/glis_known_const_fea_sample.png" alt="drawing" width=100%/>

In this case, the feasible region is the intersection of an ellipsoid and a polytope. Note that there is no requirement in GLIS that the  constraints define a convex set.

More examples of numerical benchmark testing using GLIS can be found in the `examples` folder.

<a name="basic-glisp"></a>
### Preference-based global optimization (GLISp)

To solve a preference-based optimization problem with preference function $\pi(x_1,x_2)$, $x_1,x_2\in\mathbb{R}^n$
within the finite bounds `lb` $\leq x\leq$ `ub` use the following code:

~~~python
from glis.solvers import GLISp
prob = GLISp(bounds=(lb, ub), n_initial_random=10)    # initialize GLISp object
xopt = prob.solve(pref_fun, max_prefs)                # solve problem
~~~

where `pref_fun` is the Python function implementing $\pi(x_1,x_2)$, `n_initial_random` is the number of random samples taken at initialization, and `max_prefs` is the total allowed budget of preference queries collected from `max_prefs+1` samples. The code returns the most preferred vector `xopt` found.

If additional preference queries can be done after running the optimization, the latter can be continued as follows:

~~~python
xopt = prob.xbest         # current best sample found
x = prob.xnext            # next sample to compare to xopt
pref = pref_fun(x, xbest) # evaluate preference function
x = prob.update(pref)     # update GLISp object and get next sample to possibly query
xbest = prob.xbest        # updated optimizer
~~~

Alternatively, for a full step-by-step optimization without explicitly passing the function handle `pref_fun` to GLISp, use the following code structure to solve the problem step by step:

~~~python
from glis.solvers import GLISp
prob = GLISp(bounds=(lb, ub), n_initial_random=10) # initialize GLISp object
xbest, x = prob.initialize()  # get first two random samples
for k in range(max_prefs):
    pref = pref_fun(x, xbest) # evaluate preference
    x = prob.update(pref)
    xbest = prob.xbest
xopt=xbest                    # final optimizer
~~~

A synthetic preference function `pref_fun` can be defined from a function `fun` as follows:

~~~python
def pref_fun(x1,x2):
	# Synthetic preference function mapping (x1,x2) to {-1,0,1}
    tol = 1.e-3
    f1 = fun(x1)
    f2 = fun(x2)
    if f1 <= f2 - tol:
        pref = -1
    elif f1 >= f2 + tol:
        pref = 1
    else:
        pref = 0
    return pref
~~~

#### Example (cont'd)

We apply GLISp for optimizing the camel-six-humps function $f(x)$ by acquiring the preferences
$\pi(x_1,x_2)=-1$ if $f(x_1)\leq f(x_2)-10^{-4}$, 
$\pi(x_1,x_2)=1$ if $f(x_1)\geq f(x_2)+10^{-4}$, 
$\pi(x_1,x_2)=0$ if $|f(x_1)-f(x_2)|<10^{-4}$,
obtaining `xopt` = (-0.09967807,  0.71635488) (corresponding to $f($`xopt`$)$ = -1.0312).

Here below is a plot of the samples queried with bounds constraints:

<img src="http://cse.lab.imtlucca.it/~bemporad/glis/images/glisp-1.png" alt="drawing" width=100%/>

and, with additional linear and nonlinear constraints:

<img src="http://cse.lab.imtlucca.it/~bemporad/glis/images/glisp_known_const_fea_sample.png" alt="drawing" width=100%/>


More examples of numerical benchmark testing using GLISp can be found in the `examples` folder.

<a name="advanced"></a>
## Advanced options

<a name="surrogate"></a>
### Surrogate function

By default, GLIS/GLISp use the *inverse quadratic* RBF

$$rbf(x_1,x_2)=\frac{1}{1+(\epsilon||x_1-x_2||_2)^2}$$

with $\epsilon=1$ to construct the surrogate of the objective function. To use a different RBF, for example the Gaussian RBF

$$rbf(x_1,x_2)=e^{-(\epsilon||x_1-x_2||_2)^2}$$

use the following code:

~~~python
from glis.rbf import gaussian
prob = GLIS(bounds=(lb, ub), n_initial_random=10, rbf=gaussian, rbf_epsil=3.0)
xopt, fopt = prob.solve(fun, max_evals)
~~~

The following RBFs are available in `glis`:

~~~python
from glis.rbf import gaussian, inverse_quadratic, multiquadric, thin_plate_spline, linear, inverse_multi_quadric
~~~

In alternative to RBF functions, in GLIS we can use inverse distance weighting (IDW) surrogates:

~~~python
prob = GLIS(bounds=(lb, ub), n_initial_random=10, rbf='IDW')
xopt, fopt = prob.solve(fun, max_evals)
~~~

Although IDW functions are simpler to evaluate, usually RBF surrogates perform better.


<a name="acquisition"></a>
### Acquisition function

GLIS acquires a new sample $x_k$ by solving the following nonlinear
programming problem

$$\min_x  a(x)=\hat f(x) -\alpha s^2(x) - \delta\Delta F z(x)$$

$$l \leq x\leq u$$

$$Ax \leq b,\ g(x)\leq 0$$

where $\hat f$ is the surrogate (RBF or IDW) function, $s^2(x)$ the IDW variance function, and $z(x)$ the IDW exploration function. GLIS attempts at finding a point $x_k$ where $f(x)$ is expected to have the lowest value ( $\min \hat f(x)$ ), getting $x_k$ where the surrogate is estimated to be most uncertain ( $\max s^2(x)$ ), and exploring new areas of the feasible space ( $\max z(x)$ ). The hyperparameters $\alpha$ and $\delta$ determine the tradeoff  ( $\Delta F$ is the current range of values of $f(x)$ collected so far and is used as a normalization factor).

GLIS uses Particle Swarm Optimization (PSO) to determine the minimizer $x_k$ of the acquisition problem, whose objective function $a(x)$ is cheap to evaluate.

By default, GLIS takes $\alpha=1$ and $\delta=\frac{1}{2}$. Increasing these values promotes *exploration* of the sample space, and particular increasing $\delta$ promotes *diversity* of the samples, indipendently on the function values $f(x)$ acquired, while increasing $\alpha$ promotes the informativeness of the samples and heavily depends on the constructed surrogate function $\hat f$.

To change the default values of the hyper-parameters $\alpha$ and $\delta$, use the following code:

~~~python
prob = GLIS(bounds=(lb, ub), alpha=0.5, delta=0.1)
~~~

GLISp performs acquisition in a similar way than GLIS. The surrogate $\hat f$ is determined by determining the combination of RBF coefficients, through linear programming, that make the resulting $\hat f$ satisfy the collected preference constraints. The parameter $\alpha$ is ignored.

GLISp also supports, in alternative, the acquisition based on the maximimization of the *probability of improvement*, as defined in [[2]](#cite-BP21). This can be specified as follows:

~~~python
prob = GLISp(bounds=(lb, ub), acquisition_method="prob_improvement")
~~~

By default, `acquisition_method = "surrogate"`.


<a name="recalibration"><a>

### RBF recalibration

The performance of GLISp can be usually improved by recalibrating
the RBF parameter $\epsilon$. This is achieved by performing leave-one-out cross-validation on the available samples to find the scaling $\epsilon\leftarrow\theta\epsilon$ providing the surrogate function that best satisfies the given preference constraints:

~~~python
prob = GLISp(bounds=(lb, ub), RBFcalibrate=True, RBFcalibrationSteps=steps, thetas=thetas)
~~~

where `steps` is an array of step indices at which recalibration must be performed, and `thetas` is the array of values of $\theta$ tested during recalibration.

To force the recalibration of the RBF at any time, use the command `prob.rbf_recalibrate()`, which computes the optimal value `prob.theta` of the scaling factor $\theta$.


<a name="unknown"><a>

### Unknown constraints and satisfactory samples

As detailed in [[3]](#cite-ZPB22), GLIS/GLISp can handle *unknown* constraints on $x$, where the shape of $X$ is unknown, and support labeling samples $x$ as *satisfactory* or not. To instruct the solver to collect such extra information during queries, use the following code:

~~~python
prob = GLIS(bounds=(lb, ub), n_initial_random=13, has_unknown_constraints=True, has_satisfaction_fun=True)
xopt, fopt = prob.solve(fun, max_evals=50, unknown_constraint_fun=f1, satisfactory_fun=f2)
~~~

where `f1` and `f2` are the Python functions of $x$ determining, respectively, whether $x$ is feasibile with respect to unknown constraints and satisfactory. The value returned by `f1` and `f2` must be Boolean (`True` = feasible/satisfactory, `False` otherwise).

To solve the same problem in iterative form in GLIS, use the following example:

~~~python
prob = GLIS(bounds=(lb, ub), n_initial_random=n_initial_random)
x = prob.initialize()
for k in range(max_evals):
    f = fun(x)
    fes = f1(x)
    sat = f2(x)
    x = prob.update(f, fes, sat)
xopt=prob.xopt 
fopt=prob.fopt 
~~~

A numerical benchmark with unknown constraints solved by GLIS can be found in file `glis_unknown_constraints.py` in the  `examples` folder

while in GLISp:

~~~python
prob = GLISp(bounds=(lb, ub), n_initial_random=n_initial_random, RBFcalibrate=True)
xbest, x = prob.initialize()  # get first two random samples
prob.eval_feas_sat(xbest, unknown_constraint_fun=f1, satisfactory_fun=f2)
for k in range(max_prefs):
    pref = pref_fun(x, xbest)  # evaluate preference
    prob.eval_feas_sat(x, unknown_constraint_fun=f1, satisfactory_fun=f2)
    x = prob.update(pref)
    xbest = prob.xbest
xopt = xbest
~~~

Numerical benchmarks with unknown constraints solved by C-GLISp (detailed in [[3]](#cite-ZPB22) ) can be found in `glisp_unknown_constraints.py` in the `examples` folder.

<a name="transformation"><a>

### Objective function transformation
In GLIS, when the objective function has very large and very small values, it is possible to fit the surrogate of a nonlinear transformation of the objective rather the raw objective values. For example, in the *camel-six-humps* example we want to build the surrogate $\hat f(x)\approx \log(f(x+2))$ rather than $\hat f(x)\approx f(x)$. In GLIS, you can specify the transformation function as an optional argument:

~~~python
prob = GLIS(bounds=(lb, ub), obj_transform=lambda f: np.log(f+2.), n_initial_random=10)
xopt, fopt = prob.solve(fun, max_evals)
~~~

Compare the best objective values found $f(x)$ of the *camel-six-humps* function:

<p align="center">
<img src="http://cse.lab.imtlucca.it/~bemporad/glis/images/glis-2.png" alt="drawing" width=50%/>
</p>
 
<a name="options"><a>

### Other options

Further options in executing GLIS/GLISp are detailed below:

`svdtol` tolerance used in SVD decomposition when fitting the RBF function.

`shrink_range` flag, if True the given bounds `bounds` are shrunk to the bounding box of the feasible constrained set $X=\{x: Ax\leq b, g(x)\leq 0\}$.

`constraint_penalty` penalty used to penalize the violation of the constraints $Ax\leq b$, $g(x)\leq 0$ during acquisition.

`feasible_sampling` flag, if True all the initial samples satisfy the constraints $Ax\leq b$, $g(x)\leq 0$.

`scale_delta` flag, scale $\delta$ during the iterations, as described in [[3]](#cite-ZPB22).

`expected_max_evals` expected maximum number of queries (defaulted to `max_evals` when using `GLIS.solve()`.

`display` verbosity level: 0 = none (default).

`PSOiters`, `PSOswarmsize`, `PSOminfunc`: parameters used by the PSO solver from the [`pyswarm`](https://pythonhosted.org/pyswarm/) 
package used by GLIS.

`sepvalue` (GLISp only): amount of separation $\hat f(x_1)-\hat f(x_2)$ imposed in the surrogate function when imposing preference constraints.

`epsDeltaF` (GLISp only): lower bound on the range of the surrogate function.

                 
<a name="contributors"><a>
## Contributors

This package was coded by Alberto Bemporad and Mengjia Zhu. Marco Forgione and Dario Piga also contributed to the development of the package.


This software is distributed without any warranty. Please cite the above papers if you use this software.

<a name="bibliography"><a>
## Citing GLIS

<a name="ref1"></a>

```
@article{Bem20,
    author={A. Bemporad},
    title={Global optimization via inverse distance weighting and radial basis functions},
    journal={Computational Optimization and Applications},
    volume=77,
    pages={571--595},
    year=2020
}
```

<a name="ref2"></a>

```
@article{BP21,
    title={Active preference learning based on radial basis functions},
    author={A. Bemporad and D. Piga},
    journal={Machine Learning},
    volume=110,
    number=2,
    pages={417--448},
    year=2021
}
```

<a name="ref3"></a>

```
@article{ZPB22,
    author={M. Zhu and D. Piga and A. Bemporad},
    title={{C-GLISp}: Preference-Based Global Optimization under Unknown Constraints with Applications to Controller Calibration},
    journal={IEEE Transactions on Control Systems Technology},
    month=sep,
    volume=30,
    number=3,
    pages={2176--2187},
    year=2022
}
```

<a name="license"><a>
## License

Apache 2.0

(C) 2019-2023 A. Bemporad, M. Zhu
