import cvxpy
import pyomo.environ as pyo
import pyomo.core.base.constraint as pyo_const
import numpy as np
import sympy
import warnings
import sys
import modelling_functions as mf

def pyomo_to_costmat(pyo_obj: pyo.Objective) -> tuple[np.ndarray, float]:
    """
    Extracts the cost matrix from quadratic binary pyomo objective function.
    """

    bimap, sympy_obj = pyo.expr.sympy_tools.sympyify_expression(pyo_obj.expr)

    # circumventing the sympy.Poly construction and getting directly the dictionary representation turns out to be significantly faster
    sympy_poly, _ = sympy.polys.polyutils.dict_from_expr(sympy_obj)

    n_vars = len(bimap.pyomo2sympy)

    Q = np.zeros(shape=(n_vars, n_vars))
    C = 0
    for ind, coeff in sympy_poly.items():
        ind = [i for i, x in enumerate(list(ind)) if x != 0]

        if len(ind) == 0:
            C += coeff
            continue

        assert(len(ind) in [1, 2])

        Q[min(ind)][max(ind)] += coeff
        
    return Q, C


def pyomo_to_cvxpy(pyo_obj: pyo.Objective, sense: int = pyo.minimize, verbose: bool = False) -> tuple[cvxpy.Problem, cvxpy.Variable]:
    """
    Creates unconstrained cvxpy problem from quadratic binary pyomo objective function.
    """
    Q, C = pyomo_to_costmat(pyo_obj)

    if sense == pyo.maximize:
        Q = - Q
        C = - C

    _, fixed_vars = mf.roof_duality(Q, False)
    Q, C = mf.replace_fixed_vars((Q, C), fixed_vars)

    # Trivial cases:

    if np.all(Q >= 0):
        return cvxpy.Problem(cvxpy.Minimize(0 + C)), cvxpy.Variable()
    
    if np.all(Q <= 0):
        Y = np.ones_like(Q)
        opt = np.trace(Q@Y)
        return cvxpy.Problem(cvxpy.Minimize(opt + C)), cvxpy.Variable()
    
    if np.count_nonzero(Q - np.diag(np.diagonal(Q))) == 0: # Q is diagonal
        opt = sum(q for q in Q if q < 0)
        return cvxpy.Problem(cvxpy.Minimize(opt + C)), cvxpy.Variable()


    d = np.array([[1 / np.sqrt(np.linalg.norm(Q[i,:])) for i in range(len(Q))]])
    D = np.diag(d[0])
    Di = np.linalg.inv(D)
    Q = D@Q@D

    if verbose:
        print(Q, C)

    # Y = cvxpy.Variable((len(Q), len(Q)), PSD=True)
    Z = cvxpy.Variable((len(Q), len(Q)), PSD=True)
    constraints = [Z >= 0, Z - d.T @ d <= 0] # [Y >= 0, Y <= 1, Z == Di@Y@Di]

    cvxpy_prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.trace(Q@Z) + C), constraints)

    return cvxpy_prob, D@Z@D


def get_opt_m(model: pyo.ConcreteModel, delta: float = 0.2, solver: str = 'cplex', timelimit: float = 20, verbose: bool = False) -> float:
    """
    Calculates a good penalty coefficient for a QUBO formulation of model.

    Parameters
    ----------
    model: pyo.ConcreteModel
        linearly-constrained quadratic binary problem to be formulated to QUBO

    delta: float
        desired minimal gap between optimal solution and non-valid solutions in QUBO

    solver: str
        solver that gives the best possible solution to the model within the timelimit

    timelimit: float
        maximum time, in seconds, for the solver respectively
    
    Returns
    -------
    M: float
        good penalty coefficient
    """

    # Assert that model is LCBO
    i = 0
    pyo_obj = pyo.Objective()
    for obj in model.component_objects(pyo.Objective):
        assert(pyo.expr.polynomial_degree(obj.expr) <= 2)
        pyo_obj = obj
        i+=1
    assert(i == 1) # should only have one objective function

    for c in model.component_objects(pyo.Constraint):
        if isinstance(c, pyo_const.IndexedConstraint):
            for _, ci in c.items():
                assert(pyo.expr.polynomial_degree(ci.expr) <= 1)
            continue

        assert(pyo.expr.polynomial_degree(c.expr) <= 1)

    pyomo_solver = pyo.SolverFactory(solver)
    if solver.lower() == 'glpk':
        pyomo_solver.options['tmlim'] = timelimit # for glpk time limit
    else:
        pyomo_solver.options['timelimit'] = timelimit # for cplex time limit
    
    if verbose:
        print("Starting CPLEX optimization")
    results = pyomo_solver.solve(model)
    if verbose:
        print("Finished CPLEX optimization")

    f_feas = pyo_obj() # good feasible solution via CPLEX

    cvxpy_prob, Y = pyomo_to_cvxpy(pyo_obj, sense=pyo_obj.sense, verbose=verbose)
    
    if verbose:
        print("Starting SDP relaxation")
    cvxpy_prob.solve(solver="SCS", verbose=False, eps_abs=delta, time_limit_secs=timelimit) # <- leave solver empty to automatically choose best available solver
    if verbose:
        print("Finished SDP relaxation")

    if cvxpy_prob.status == "unbounded":
        raise Exception("relaxed problem is unbounded -- consider normalizing Q")

    if cvxpy_prob.status == "infeasible":
        raise Exception("relaxed problem is infeasible -- consider increasing timelimit")

    if cvxpy_prob.status == "optimal_inaccurate":
        y = Y.value

        if np.all(y >= 0) and np.all(y <= 1): # found solution is feasible but not optimal
            warnings.warn("found solution is feasible but not optimal -- M is likely too small")

    f_unc = cvxpy_prob.value # good lower bound of unconstrained problem via SDP

    if pyo_obj.sense == pyo.minimize:
        M = f_feas - f_unc + delta
    else:
        M = - f_feas - f_unc + delta
        f_unc *= -1

    if verbose:
        print("f_feas =", f_feas)
        print("f_unc =", f_unc)
        print("M =", M)
    return M
