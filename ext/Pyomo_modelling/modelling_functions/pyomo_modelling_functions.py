import pyomo
import pyomo.environ as pyo
import pyomo.util.infeasible as pui
import pyomo.core.base.PyomoModel
from pyomo.core.expr.numeric_expr import decompose_term
from pyomo.core.expr.visitor import identify_variables, ExpressionReplacementVisitor
from pyomo.repn.standard_repn import generate_standard_repn
from typing import Any, List, Tuple, Dict, Union, Optional
import itertools
from collections import defaultdict
import numpy as np
from big_m_heuristic import big_m
from .helpers import get_component_by_name, gcd_decimal, mod_decimal, clear_denominators, set_new_attr_name


__all__ = [
    'solve_instance', 
    'calculate_QUBO_density',
    'check_infeasibilities',
    'reduce_BLP', 
    'BLP_to_QUBO', 
    'validate_penalty_term', 
    'factor_expression', 
    'factored_terms_to_expr',
    'simplify_terms',
    'pyomo_QUBO_to_array', 
    'binary_discretization',
    'MILP_to_QUBO',
    'binary_constraints_to_filter',
    'decompose_linear_binary_constraint',
    'check_trivial_constraint',
    'binary_encoding',
    'first_constraint_reduction_simple',
    'third_constraint_reduction_single_bound',
    'calculate_QUBO_density',
    'substitute_variable_by_expression'
]

def solve_instance(instance, solver='CPLEX', time_limit=0, verbose=False):
    pyomo_solver = pyo.SolverFactory(solver.lower(), tee=False)
    if time_limit != 0:
        if solver.lower() == 'glpk':
            pyomo_solver.options['tmlim'] = time_limit #  glpk time limit
        else:
            pyomo_solver.options['timelimit'] = time_limit # cplex time limit
    results = pyomo_solver.solve(instance, report_timing=verbose)
    computation_time = results.solver.time
    # check if it was feasible:
    assert results.solver.termination_condition != pyo.TerminationCondition.infeasible, 'infeasible'
    assert results.solver.termination_condition != pyo.TerminationCondition.unbounded, 'No solution found'
    assert results.solver.termination_condition != pyo.TerminationCondition.unknown, 'No solution found. Increase time limit'
    objective_function = list(instance.component_objects(pyo.Objective, active=True))[0]
    objective = pyo.value(objective_function)
    return(instance, objective, computation_time)

def check_infeasibilities(solved_instance):
    for constraint, body_value, info in pui.find_infeasible_constraints(solved_instance, tol=1e-6):
        print(constraint.name, 'with body value of', str(body_value), 'too low' if info == 1 else 'too high' if info == 2 else 'undefined' if info == 4 else '')
    
def normalize_constraint_sign(a, x, l, u, normalize_coeffs_by_negation=False):
    """This function normalizes data characterizing a linear binary constraint.
    It makes sure that the upper bound (if given) is positive and, optionally employing negation of variables, that the upper bound is not None and all coefficients are positive.
    Args:
        a (list[Numeric]): list of nontrivial linear coefficients in constraint body
        x (list[VarData]): list of binary variables in constraint body
        l (boundType): lower bound of constraint
        u (boundType): upper bound of constraint
        normalize_coeffs_by_negation (bool, optional): decision to make coefficients positive by negation of variables. Defaults to False.
    Returns:
        (tuple[list[Numeric],  list[VarData], boundType,  boundType, list[bool]]): Normalized constraint data (analogous to input data) with information of which variables are negated to achieve this form.
    """
    assert not (l is None and u is None)
    if (l is None or l < 0) and (u is None or u < 0) or (normalize_coeffs_by_negation and u is None):
        a = list(-np.array(a))
        l, u = None if u is None else -u, None if l is None else -l
    # normalize a if asked for
    if normalize_coeffs_by_negation:
        a_normalized = []
        x_negation: list[bool] = []
        for e in a:
            if e < 0:
                a_normalized += [-e]
                if l is not None:
                    l -= e
                if u is not None:
                    u -= e
                x_negation += [True]
            else:
                a_normalized += [e]
                x_negation += [False]
        return a_normalized, x, l, u, x_negation
    else:
        return a, x, l, u, [False] * len(x)
    
def unnormalize_negated_constraint(a_normalized, x, l, u, x_negation, switch_bounds=False):
    """ 
        Reverses the transformation of constraint data by negation of variables.
    Args:
        a_normalized (list[Numeric]): list of positive nontrivial linear coefficients in constraint body
        x (Any): list of variables in constraint body
        l (boundType): lower bound of constraint
        u (boundType): upper positive bound of constraint
        x_negation (list[bool]): list of bools signifying which variables have been negated to achieve positive coefficients
        switch_bounds (bool, optional): decision to switch lower and upper bound in output. Defaults to False.

    Returns:
        (tuple[ list[Numeric],  Any,  boundType, boundType]): constraint data without negated variables.
    """
    a = [(1 - 2 * x_negation[i]) * a_normalized[i] for i in range(len(a_normalized))]
    if u is not None:
        u -= np.dot(np.array(a_normalized), np.array(x_negation))
    if l is not None:
        l -= np.dot(np.array(a_normalized), np.array(x_negation))
    if switch_bounds:
        a = list(-np.array(a))
        l, u = None if u is None else -u, None if l is None else -l
    return a, x, l, u

def decompose_linear_binary_constraint(constraint, normalize_sign = True):
    '''Decomposes a linear binary constraint into abstract info. l <= ax <= u
    Input:
        - constraint: Pyomo constraint with linear constraint body and exclusively binary variables
        - normalize_sign: True, if constraint signs should be normalized
    Outputs:
        - a: tuple of nontrivial linear coefficients appearing in the constraint body.
        - x: tuple of variables appearing in the constraint body
        - l: float, lower bound of the constraint body
        - u: float, upper bound of the constraint body
    '''
    try:
        assert constraint.is_constructed(), "Constraint {} is not yet constructed".format(constraint.name)
    except AttributeError:
        parent = constraint.parent_component()
        if parent is not None:
            assert parent.is_constructed(), "Constraint {} is not yet constructed".format(constraint.name)
        else:
            raise ValueError("Constraint {} has no valid parent component.".format(constraint.name))
    assert decompose_term(constraint.body)[0], 'Constraint {} not linear.'.format(constraint.name)
    decomposed_body = decompose_term(constraint.body)[1]
    a, x = list(zip(*decomposed_body))
    # remove 0-order part and handle fixed_variables
    intercept = 0
    remove_indices = []
    for i, var in enumerate(x):
        if var is None or a[i] == 0:
            intercept += a[i]
            remove_indices += [i]
        elif var.is_fixed():
            intercept += a[i] * var.value
            remove_indices += [i]
        else:
            pass
    x = tuple(x[i] for i in range(len(x)) if i not in remove_indices)
    a = tuple(a[i] for i in range(len(a)) if i not in remove_indices)
    assert np.all([variable.is_binary() for variable in x]), 'Constraint {} does not have exclusively binary variables.'.format(constraint.name)
    # subtract intercept from bounds
    bounds = [constraint.upper, constraint.lower]
    for i in range(2):
        if bounds[i] is not None:
            bounds[i] -= intercept
            # remove floating point errors
            bounds[i] = round(bounds[i], 10)
    u, l = bounds

    min_a, max_a = sum(i for i in a if i < 0), sum(i for i in a if i > 0)
    assert not (l is None and u is None) and ((l is None or u is None) or l <= u), 'Fix bounds ({},{}) in constraint {}'.format(l, u, constraint.name)
    assert (l is None or l <= max_a) and (u is None or min_a <= u), 'Constraint {} infeasible.'.format(constraint.name)
    # normalize constraint such that upper bound is positive
    if normalize_sign:
        a, x, l, u, _ = normalize_constraint_sign(a, x, l, u)
    return a, x, l, u

def construct_linear_constraint(a, x, l, u):
    '''Constructs a linear constraint with binary variables from the output of decompose_linear_binary_constraint. It serves as the inverse function basically. l <= ax <= u
    Inputs:
        - a: tuple of linear coefficients to appear in the constraint body.
        - x: tuple of variables to appear in the constraint body, multiplied by a
        - l: float, lower bound of the constraint body
        - u: float, upper bound of the constraint body
    Outputs:
        - constraint: Pyomo constraint object
    '''
    assert not (u is None and l is None) and np.count_nonzero(a) == len(a) == len(x)
    expr = pyo.inequality(l, sum(a_i * x_i for a_i, x_i in zip(a,x)), u)
    constraint = pyo.Constraint(expr=expr)
    return constraint

def convert_to_normalized_upper_bound(a, x, l, u):
    """Normalizes data characterizing a linear binary constraint that is bounded from one side to data characterizing an upper bound constraint of the form ax <= u.
    By variable negation all linear coefficients a in the constraint body come out as positive.
    Args:
        a (list[Numeric]): list of linear coefficients in constraint body
        x (Any): list of variables in constraint body
        l (boundType): lower bound of constraint. In the case of an upper bound constraint, l is None
        u (boundType): upper bound of constraint. In the case of an lower bound constraint, u is None
    Returns:
        (tuple[list[Numeric],  Any, Numeric, list[bool]]): a_normalized (list[Numeric]): list of positive linear coefficients in constraint body
                                                           x (Any): list of variables in constraint body
                                                           u (Numeric) positive upper bound of constraint body
                                                           x_negation (list[bool]): list of bools signifying which variables have been negated to achieve positive coefficients
    """
    assert (l is None) != (u is None), "Provide a constraint bounded from one side."
    if u is None:
        a = list(-np.array(a))
        u = -l
    assert u is not None  # unnecessary but for Typing compiler
    # make a positive (u should become positive as well, except the constraint is infeasible)
    a_normalized = []
    x_negation = []
    for i, e in enumerate(a):
        if e < 0:
            a_normalized += [-e]
            u -= e
            x_negation += [True]
        else:
            a_normalized += [e]
            x_negation += [False]
    assert u >= 0, "Constraint infeasible."
    return a_normalized, x, u, x_negation

def update_constraint_info(decomposed_normalized_constraints, indices, values=None):
    '''Update information about normalized (only positive coeffcients and upper bound) based on a selection of variables to be fixed.
    Inputs:
        - decomposed_normalized_constraints: list of tuples (a, x, u, x_negation) with:
            - a: tuple of positive linear coefficients in constraint body.
            - x: tuple of binary coefficients appearing in constraint body.
            - u: upper positive bound.
            - x_negation: tuple of booleans, indicating which variables are negated. This will affect the value of variable fixation.
        - indices: list of indices of variables which shall be removed from the constraint
        - values: list of len(indices) indicating the values that the variables should be fixed to in the non-negated state. Default is None, whoch will fix them all to 0.
    Outputs:
        - decomposed_normalized_constraints: updated list of tuples with some entries removed and upper bound updated.
    '''
    if values is None:
        values = [0] * len(indices)
    for k, j in sorted(enumerate(indices), key=lambda x: x[1], reverse=True):
        for i, (a, x, u, x_negation) in sorted(enumerate(decomposed_normalized_constraints), reverse=True):
            u -=  a[j] * int(values[k] != int(x_negation[j]))
            assert u >= 0, 'upper bound infeasible'
            a = a[:j] + a[j+1:]
            x = x[:j] + x[j+1:]
            x_negation = x_negation[:j] + x_negation[j+1:]
            if len(a) > 0:
                decomposed_normalized_constraints[i] = (a, x, u, x_negation)
            else:
                del decomposed_normalized_constraints[i]
    return decomposed_normalized_constraints

def remove_irrelevant_variables(decomposed_normalized_constraints, adjust_bounds=False):
    '''Update a list of data of normalized constraints by scanning for variables which are too small to make a difference and remove them if possible.
       The method to identify these variables is a heuristic based on calculation the greatest common denominator of the (largest) coefficents of variables, and taking the modulo of the constraint bound with respect to this value.
       If the complement variable set (the smallest variables) can never exceed that modulo, then it won't make a difference in the constraint.
       This is because the values of the largest variables lie on a lattice spanned by the greatest common denominator, and thus the small variables will always fall in between.
       This method is not exhaustive, as cases like 10x + 12y + 3z <= 15 are not picked up, but probably require more expensive routines.
    Inputs:
        - decomposed_normalized_constraints: list of tuples (a, x, u, x_negation) with:
            - a: tuple of positive linear coefficients in constraint body. (In the case of two constraints these should be matching)
            - x: tuple of binary coefficients appearing in constraint body.
            - u: upper positive bound.
            - x_negation: tuple of booleans, indicating which variables are negated. This will affect the value of variable fixation.
    Outputs:
        - decomposed_normalized_constraints: list of tuples (a, x, u, x_negation) with some variables missing
    '''
    a = decomposed_normalized_constraints[0][0]
    if len(decomposed_normalized_constraints) == 2:
        assert np.all(np.array(a) == np.array(decomposed_normalized_constraints[1][0])), 'Something fishy is going on. as do not align.' # make sure both a are the same
    if len(a) > 2:
        a = np.asarray(a)
        # Sort by ascending value and keep original indices
        sorted_indices = np.argsort(a)
        sorted_a = a[sorted_indices]
        # Compute cumulative sums
        csum = np.cumsum(sorted_a)
        bsum_bound = sorted_a[-2] # start with second largest element. Case len(a_new) = 1 is excluded by previous tests
        min_a = sorted_a[0]
        side = 'right'
    else: # do nothing
        bsum_bound = 0
        min_a = 1
    keep_checking = [True] * len(decomposed_normalized_constraints)
    while np.any(keep_checking) and bsum_bound >= min_a:
        k = np.searchsorted(csum, bsum_bound, side=side) # find where the sums of b will be smaller (side='left') or smaller equal (side=('right)) bsum_bound
        b = a[sorted_indices[:k]]
        a_new = a[sorted_indices[k:]] # a without b
        if k == 0: # nothing more to check
            break
        relevance_b = csum[k-1] # = sum(b) # this is a measure for how relevant b is, assuming all bs are positive
        gcd_value = gcd_decimal(*a_new)
        if gcd_value < min_a: # as gcd_value will just decrease with time, it makes not sense checking for any other b
            break
        elif gcd_value < relevance_b: # relevance_bound will be smaller or equal to gcd_value, so already no hope
            bsum_bound = gcd_value
            side = 'right'
            continue
        else:
            bsum_bound = relevance_b
            side = 'left' # when relevance_b is bsum_bound, we want to make b smaller, so check for smaller csum < bsum_bound
        for i, u in enumerate([u for a, x, u, x_negation in decomposed_normalized_constraints]):
            if keep_checking[i]:
                relevance_bound = mod_decimal(u, gcd_value) # the relevance of b has to be larger than this for them to be relevant for the upper bound
                if relevance_b <= relevance_bound:
                    # eliminate b from the constraint
                    decomposed_normalized_constraints[i] = update_constraint_info([decomposed_normalized_constraints[i]], sorted_indices[:k], values=[0] * k)[0] # basically set the variables that are removed to 0
                    if adjust_bounds:
                        a, x, u, x_negation = decomposed_normalized_constraints[i]
                        decomposed_normalized_constraints[i] = (a, x, u - relevance_bound, x_negation) # make u tighter
                    keep_checking[i] = False
    return decomposed_normalized_constraints

def check_trivial_constraint(constraint, verbose=False):
    '''Checks if a linear binary-variable-containing constraint is trivial and determines which variables should be fixed.
    Inputs:
        - constraint: A linear constraint with binary variables of a Pyomo model
        - verbose: bool, for printing info. Default is False.
    Outputs:
        - constraint_trivial: bool indicating if the constraint is trivial and can be removed (if some variables are fixed)
        - fix_variable_tuples: list of pairs of variables and its values to be assigned if the constraint is removed
        - add_constraints = list of tuples that serve as inputs for construct_linear_constraint to construct a linear binary constraint. If the input constraint is removed, these constraints have to be added.
    '''
    a, x, l, u = decompose_linear_binary_constraint(constraint)
    decomposed_normalized_constraints = [] # here I will collect normalized lower and upper bound constraint with form ax <= u each, with positive coefficients
    if l is not None:
        decomposed_normalized_constraints += [(convert_to_normalized_upper_bound(a, x, l, None))]
    if u is not None:
        decomposed_normalized_constraints += [(convert_to_normalized_upper_bound(a, x, None, u))]
    initial_body_length = len(a)
    num_constraints = len(decomposed_normalized_constraints)
    if num_constraints > 1:
        assert np.all([id(x) == id(y) for i in range(1,num_constraints) for x, y in zip(decomposed_normalized_constraints[0][1], decomposed_normalized_constraints[i][1])])
    fix_variable_tuples = []
    keep_checking = True
    # start checking for trivial constraints:
    while len(decomposed_normalized_constraints) > 0 and keep_checking:
        keep_checking = False
        for i, (a, x, u, x_negation) in sorted(enumerate(decomposed_normalized_constraints), reverse=True):
            # upper bound too large
            if sum(a) <= u:
                del decomposed_normalized_constraints[i]
                keep_checking = True
            elif np.any(np.array(a) > u): # upper bound not large enough
                too_large_indices = np.where(np.array(a) > u)[0]
                fix_variable_tuples += [(x[j], int(x_negation[j])) for j in too_large_indices]
                # update constraint coefficients
                try:
                    decomposed_normalized_constraints = update_constraint_info(decomposed_normalized_constraints, too_large_indices, values=[int(x_negation[j]) for j in too_large_indices])
                except AssertionError:
                    raise Exception('Constraint {} infeasible.'.format(constraint.name))
                if len(decomposed_normalized_constraints) == 0:
                    break
                keep_checking = True
            else:
                pass
    # Now, check if variables can be can be removed from constraints (but not fixed)
    bounds_too_loose = []
    if len(decomposed_normalized_constraints) > 0: 
        decomposed_normalized_constraints = remove_irrelevant_variables(decomposed_normalized_constraints)
        for i in range(len(decomposed_normalized_constraints)): # tighten bounds
            a, x, u, x_negation = decomposed_normalized_constraints[i]
            bound_too_loose = mod_decimal(u, gcd_decimal(*a))
            bounds_too_loose += [bound_too_loose]
            if bounds_too_loose:
                decomposed_normalized_constraints[i] = (a, x, u - bound_too_loose, x_negation) # reduce u to lie on the grid spanned by the constraint body
    # prepare output:
    constraint_trivial = len(decomposed_normalized_constraints) == 0
    add_constraints = []
    if num_constraints == len(decomposed_normalized_constraints) and np.all([len(a) >= initial_body_length - len(fix_variable_tuples) for a, x, u, x_negation in decomposed_normalized_constraints]):
        decomposed_normalized_constraints = [dnc for dnc, bound_too_loose in zip(decomposed_normalized_constraints, bounds_too_loose) if bound_too_loose] # in this case case no reduction happened or only due to variable fixations or and we do not need to add these back, except when the bounds were tightened
    if len(decomposed_normalized_constraints) == 2: # if two are left, identify if they can be combined to one again (lower and upper bound)
        if decomposed_normalized_constraints[0][0] == decomposed_normalized_constraints[1][0]: # if a agree, i.e. if they come from the same body
            assert np.all(np.array(decomposed_normalized_constraints[0][3]) != np.array(decomposed_normalized_constraints[1][3])) # just checking
            sort_key = lambda dnc: dnc[2] - np.dot(dnc[0], dnc[3]) # sort by transformed upper bound u - np.dot((a, x_negation))
            decomposed_normalized_constraints = sorted(decomposed_normalized_constraints, key=sort_key) # sort such that upper bound will have larger abs than lower
            a, x, u, x_negation = decomposed_normalized_constraints[1]
            l = sum(a) - decomposed_normalized_constraints[0][2] # this should be the lower bound that comes from negating all variables, as is used in the second constraint
            add_constraints += [(unnormalize_negated_constraint(a, x, l, u, x_negation))]
            decomposed_normalized_constraints = []
    # now add all the remaining constraints to add_constraints seperately
    for a, x, u, x_negation in decomposed_normalized_constraints:
        add_constraints += [(unnormalize_negated_constraint(a, x, None, u, x_negation, switch_bounds=bool(u - np.dot(a, x_negation) <= 0)))]
    # print Info
    if verbose:
        if sum(bounds_too_loose):
            print('Bounds of constraint {} can be tightened by {}'.format(constraint.name, sum(bounds_too_loose)))
        if constraint_trivial:
            if len(fix_variable_tuples) == 0:
                print('Constraint {} is trivial and can be removed'.format(constraint.name))
            elif fix_variable_tuples:
                print('Constraint {} can be removed when some variables are fixed'.format(constraint.name))
        elif fix_variable_tuples:
            if add_constraints:
                print('Constraint {} can be simplified when some variables are fixed'.format(constraint.name))
            else:
                print('Some variables in constraint {} can be fixed'.format(constraint.name))
        elif add_constraints: 
                print('Constraint {} can be simplified'.format(constraint.name))
    return constraint_trivial, fix_variable_tuples, add_constraints

def check_first_constraint_triviality(constraint, verbose=False):
    '''Checks if a linear binary-variable-containing constraint is trivial as it is. This does not suggest any variable fixes
    Inputs:
        - constraint: A linear constraint with binary variables of a Pyomo model
        - verbose: bool, for printing info. Default is False.
    Outputs:
        - lower_bound_trivial: bool indicating if the lower bound constraint is None or trivial and can be removed
        - upper_bound_trivial: bool indicating if the upper bound constraint is None or trivial and can be removed
    '''
    a, x, l, u = decompose_linear_binary_constraint(constraint, normalize_sign=False)
    if l is None:
        lower_bound_trivial = True
    else:
        a_l, x_l, u_l, x_negation = convert_to_normalized_upper_bound(a, x, l, None)
        lower_bound_trivial = sum(a_l) <= u_l
        if verbose and lower_bound_trivial:
            print('Lower bound of Constraint {} is trivial'.format(constraint.name))
    if u is None:
        upper_bound_trivial = True
    else:
        a_u, x_u, u_u, x_negation = convert_to_normalized_upper_bound(a, x, None, u)
        upper_bound_trivial = sum(a_u) <= u_u
        if verbose and upper_bound_trivial:
            print('Upper bound of Constraint {} is trivial'.format(constraint.name))
    return lower_bound_trivial, upper_bound_trivial

def first_constraint_reduction_simple(constraint, verbose=False):
    '''Checks if a linear binary-variable-containing constraint is trivial as it is (without considering variable fixes). If a bound is trivial, the constraint is changed.
    Inputs:
        - constraint: A linear constraint with binary variables of a Pyomo model
        - verbose: bool, for printing info. Default is False.
    Outputs:
        - constraint_trivial: bool indicating if all bounds of the constraint are trivial
    '''
    lower_bound_trivial, upper_bound_trivial = check_first_constraint_triviality(constraint, verbose=verbose)
    constraint_trivial = False
    if lower_bound_trivial and upper_bound_trivial:
        constraint_trivial = True
    elif lower_bound_trivial and constraint.lb is not None:
        constraint.set_value((None, constraint.body, constraint.ub))
    elif upper_bound_trivial and constraint.ub is not None:
        constraint.set_value((constraint.lb, constraint.body, None))
    if verbose and constraint_trivial:
        print('Constraint {} is trivial'.format(constraint.name))
    return constraint_trivial

def third_constraint_reduction_single_bound(constraint, verbose=False):
    '''Checks if a linear binary-variable-containing constraint bounded from one side has some coefficients which are too small to make a difference (third reduction rule). 
    If so, the constraint is changed.
    The bound is also adjusted if possible.
    This, however, may not affect 0-coefficients.
    Inputs:
        - constraint: A linear constraint with binary variables of a Pyomo model, bounded from one side
        - verbose: bool, for printing info. Default is False.
    Outputs:
        - constraint_trivial: bool indicating if all bounds of the constraint are trivial
    '''
    a, x, l, u = decompose_linear_binary_constraint(constraint, normalize_sign=False)
    var_names = {var.name for var in x}
    assert l is None or u is None
    if l is not None:
        a, x, u, x_negation = convert_to_normalized_upper_bound(a, x, l, None)
        switch_normalized_bounds = True
    elif u is not None:
        a, x, u, x_negation = convert_to_normalized_upper_bound(a, x, None, u)
        switch_normalized_bounds = False
    a, x, u, x_negation = remove_irrelevant_variables([(a, x, u, x_negation)], adjust_bounds=True)[0]
    remaining_var_names = {var.name for var in x}
    removed_var_names = var_names - remaining_var_names
    num_eliminated = len(removed_var_names)
    if num_eliminated > 0:
        a, x, l, u = unnormalize_negated_constraint(a, x, None, u, x_negation, switch_bounds=switch_normalized_bounds)
        new_body = sum(a_i * x_i for a_i, x_i in zip(a, x))
        constraint.set_value((l, new_body, u))
        if verbose:
            print('{} variables eliminated from constraint {}'.format(num_eliminated, constraint.name))
    return removed_var_names

    
def fix_vars_by_objective(model):
    '''Marks binary variables to fix which do not appear in any constraints by checking if coefficients in the objective agree in their sign.
        This includes variables which do not appear in any constraint or objective.
    Input:
        model (ConcreteModel): Pyomo concrete model
    Output:
        fix_variable_tuples (list[tuple[VarData, int]]): list of pairs of variables and its values to be assigned
    '''
    # note: The efficiency of this function is yet to be tested
    fix_variable_tuples = []
    for objective in model.component_objects(pyo.Objective, active=True):
        constraints_var_ids = set(id(var) for constraint in list(model.component_data_objects(pyo.Constraint, active=True)) for var in identify_variables(constraint.body, include_fixed=False))
        vars_in_objective = list(identify_variables(objective.expr, include_fixed=False))
        variables = [var for var in model.component_data_objects(pyo.Var, active=True) if not var.is_fixed()]
        assert np.all(var.is_binary() for var in vars_in_objective)
        leftover_ids = set(id(var) for var in variables) - constraints_var_ids
        if leftover_ids:
            leftover_vars = [var for var in variables if id(var) in leftover_ids]
            factored_objective = simplify_terms(factor_expression(objective.expr), reduce_fixed_variables=True, reduce_higher_order_binaries=True)
            factored_objectives_leftover_ids = [(coef, [id(var) for var in vars if id(var) in leftover_ids]) for coef, vars in factored_objective]
            # collect coefficents belonging to a variable (both linear and higher terms)
            var_id_dict = defaultdict(list)
            for coef, ids in factored_objectives_leftover_ids:
                for i in ids:
                    var_id_dict[i].append(coef)
            for var in leftover_vars:
                coefs = var_id_dict.get(id(var), [0])  # include variables which do not appear in objective by 0-coefficient
                same_sign = not min(coefs) < 0 < max(coefs) if len(coefs) > 0 else False
                if same_sign: # if all coefficients have the same sign, we can fix the corresponding variable
                    value = 0 if np.sign(coefs[0]) * objective.sense > 0 else 1 # fix to this value
                    fix_variable_tuples += [(var, value)]
    return fix_variable_tuples

def reduce_BLP(model, verbose=1):
    '''Reduces a Pyomo BLP model (or rather a linearly-constrained binary model) by eliminating trivial constraints and fixing variables
    Inputs:
        - model: linearly constrained Pyomo model
        - verbose: int or bool, for printing info. verbose=2 will print more info than verbose=1. Default is 1
    Outputs:
        - model: Reduced linearly-constrained binary Pyomo model
        - fix_variables: dictionary of all variables to be fixed (var.id:(var,value))
    '''
    trivial_constraint_counter = 0
    variable_fix_counter = 0
    constraints_added_counter = 0
    fix_variables = {}
    check_constraints = True
    while check_constraints:
        check_constraints = False
        constraints = list(model.component_data_objects(pyo.Constraint, active=True))
        for constraint in constraints:
            constraint_trivial, fix_variable_tuples, add_constraints = check_trivial_constraint(constraint, verbose=(verbose > 1))
            if constraint_trivial or add_constraints:
                trivial_constraint_counter += 1
                # deactivate trivial constraints
                constraint.deactivate()
                for i, (a, x, l, u) in enumerate(add_constraints):
                    constraints_added_counter += 1
                    constraint_new = construct_linear_constraint(a, x, l, u)
                    constraint_new_name = set_new_attr_name(model, constraint.name, str(i + 1))
                    setattr(model, constraint_new_name, constraint_new)
                if verbose > 1: print('Constraint {} deactivated'.format(constraint.name))
            if fix_variable_tuples:
                fix_variables = {**fix_variables, **{id(variable):(variable, value) for variable, value in fix_variable_tuples}}
                for variable, value in fix_variable_tuples:
                    if not variable.is_fixed():
                        variable.fix(value)
                        variable_fix_counter += 1
                        check_constraints = True # keep checking until on one iteration no further variable is fixed
                        if verbose > 1: print('Variable {} fixed to {}'.format(variable.name, value))

    # check wether there are some variables in the objective which do not appear in any constraint:
    fix_vars_by_objective_rounds = max(
        [
            objective.expr.polynomial_degree()
            for objective in model.component_objects(pyo.Objective, active=True)
        ]
    )
    for _ in range(fix_vars_by_objective_rounds):
        fix_variable_tuples = fix_vars_by_objective(model)
        if fix_variable_tuples:
            fix_variables = {**fix_variables, **{id(variable):(variable, value) for variable, value in fix_variable_tuples}}
            for variable, value in fix_variable_tuples:
                if not variable.is_fixed():
                    variable.fix(value)
                    variable_fix_counter += 1
                    if verbose > 1: print('Variable {} fixed to {}'.format(variable.name, value))

    # final prints:
    if verbose:
        if trivial_constraint_counter > 0: print('{} trivial constraints removed'.format(trivial_constraint_counter))
        if variable_fix_counter > 0: print('{} variables fixed'.format(variable_fix_counter))
        if constraints_added_counter > 0: print('{} constraints added'.format(constraints_added_counter))
    return model, fix_variables

def BLP_to_QUBO(model, penalties=None, delta=0.2, slack_encoding='sum', relative_precision=0, reduce_model=True, normalize_penalty_terms=False, verbose=0):
    '''Converts a Pyomo Binary linear Programm model to a Pyomo QUBO model
        Inputs:
            model: Pyomo model in BLP-form.
            penalties: penalty coefficient for the penalty terms. Either a universal coefficient, given by a float, or a dictionary with constraint names as keys and penalties as values. Default is None, which will run the big heuristic for finding a penalty.
            delta: float needed to run the big-M heuristic as the standard argument for penalties. If penalties=None is chosen, delta will be added to the penalty. Default is 0.2.
            slack_encoding: sting out of ["sum", "one-hot", "log", "AP"], determining how slack variables, if needed, encode variable ranges. Default is "sum". 
                - sum: sum of binary slack variables. May come with degeneracy in the slack variable values. Scales linearly with the to-be-encoded range.
                - one-hot: one-or-less encoding which constraints enforcing it. No degeneracy here". Scales linearly with the to-be-encoded range.
                - log: standard binary encoding. May come with degeneracy in the slack variable values when the to-be-encoded range does not equal 2^N-1 for some N. Scales logarithmically with the to-be-encoded range.
                - AP: Adapted Arithmetic Progression Encoding, basically taking ascending integers as coeffcients. May come with degeneracy in the slack variable values. Scales with the square root of the to-be-encoded range.
            relative_precision (float): relative precision with which slack terms will approximate the constraint bodies of inequality constraints. Default of precision=0 means no approximation. 
            reduce_model: bool specifying wether to apply the reduce_BLP function to possibly reduce problem complexity. This may deactivate some constraints and fix some variables in the input model. Default is True
            fix_vars_in_input: bool, determining wether variables in trivial constraints shall also be fixed for the input model. This is recommended if the input model is used to define a filter for samplesets of the output QUBO. Default is True.
            - normalize_penalty_terms (bool): If True, the penalty term will be normalized such that the smallest coefficient in the constraint body is scaled to 1. Default is False.
            verbose: int, for printing info. 1 provides moderate detail, while 2 provides high detail. Default is 0.
        Output:
            QUBO_model: Concrete Pyomo model with QUBO as objective function. The constraints of the input model are still there, but set inactive. QUBO_model contains an object penatly_terms, which contains all the information about which penalty terms and coefficients belong to which constraint.
    '''
    if reduce_model: # reduce model complexity by removing trivial constraints and fixing variables
        model, fix_variables = reduce_BLP(model, verbose=verbose)
    QUBO_model = model.clone()
    for objective in QUBO_model.component_objects(pyo.Objective, active=True):
        assert np.all([variable.is_binary() for variable in identify_variables(objective.expr, include_fixed=False)]), 'Objective function must have exclusively binary variables.'
        assert objective.expr.polynomial_degree() <= 2, 'Objective function must not have higher terms than quadratic.'
        if objective.sense < 0: # force minimization
            objective.expr =  - objective.expr
            objective.sense = 1
    constraints = list(QUBO_model.component_data_objects(pyo.Constraint, active=True))
    # prepare penalty dictionary
    if penalties is None:
        penalties = big_m.get_opt_m(model, delta)
    if isinstance(penalties, (float, int)):
        penalties = dict((constraint.name, penalties) for constraint in constraints)
    elif isinstance(penalties, dict):
        dict_to_add = {}
        keys_to_delete = []
        for key, value in penalties.items():
            constraint = get_component_by_name(QUBO_model, key)
            if not constraint.active:
                keys_to_delete += [key]
            else:
                if constraint.is_indexed():
                    for con in constraint.values():
                        if con.active:
                            dict_to_add[con.name] = value
                    keys_to_delete += [key]
        for key in keys_to_delete:
            del penalties[key]
        penalties = {**penalties, **dict_to_add}
    else:
        raise TypeError('penalties argument must be a number or a dictionary')
    if hasattr(QUBO_model, 'penalty_terms'):
        for constraint_name in penalties.keys():
            QUBO_model.penalty_terms[constraint_name] = (penalties[constraint_name],)
    else:
        QUBO_model.penalty_terms = {key:(value,) for key, value in penalties.items()}
    slacks_added = 0
    # turn constraints into penalty terms
    for constraint in constraints:
        if reduce_model:
            constraint_removable = False
        else:
            constraint_removable = first_constraint_reduction_simple(constraint, verbose=(verbose > 1)) # simple version of reduction checking of it is trivial and if possible removing a trivial bound
            # constraint_trivial, fix_variable_tuples, add_constraints = check_trivial_constraint(constraint, verbose=(verbose > 1))
            # if verbose > 1 and (constraint_trivial or fix_variable_tuples or add_constraints):
            #     print('Constraint {} can be removed when fixing variables or simplified. Argument reduce_model=True recommended'.format(constraint.name))
            # constraint_removable = constraint_trivial and len(fix_variable_tuples) == 0 and len(add_constraints) == 0
        if constraint_removable:
            if constraint.name in QUBO_model.penalty_terms:
                QUBO_model.penalty_terms[constraint.name] += (0,)
        else:
            penalty_term, slack = constraint_to_penalty(constraint, slack_encoding=slack_encoding, relative_precision=relative_precision, normalize_penalty_term=normalize_penalty_terms, verbose=(verbose > 1))
            if slack is not None:
                slacks_added += len(slack.index_set())
            if constraint.name in QUBO_model.penalty_terms:
                QUBO_model.penalty_terms[constraint.name] += (penalty_term,)
            else:
                raise Exception('Please include constraint {} in the dictionary in the penalties argument'.format(constraint.name))
        constraint.deactivate()
    if verbose:
        if slacks_added > 0: print('{} slack variables added'.format(slacks_added))
    # add penalty terms to objective
    if len(list(QUBO_model.component_objects(pyo.Objective, active=True))) == 0:
        QUBO_model.obj = pyo.Objective(expr=0)
    for objective in QUBO_model.component_objects(pyo.Objective, active=True):
        for penalty, penalty_term in QUBO_model.penalty_terms.values():
            objective.expr += penalty * penalty_term
    return QUBO_model
    
def constraint_to_penalty(constraint, slack_encoding='sum', relative_precision=0, normalize_penalty_term=False, verbose=False):
    '''Calculates the penalty term to substitute a Pyomo constraint with. If needed, slack variables are added to the corresponding model.
    Inputs:
        - constraint: linear Pyomo constraint of binary variables.
        - slack_encoding: sting out of ["sum", "one-hot", "log", "AP"], determining how slack variables, if needed, encode variable ranges. Default is "sum". 
            - sum: sum of binary slack variables. May come with degeneracy in the slack variable values. Scales linearly with the to-be-encoded range.
            - one-hot: one-or-less encoding which constraints enforcing it. No degeneracy here". Scales linearly with the to-be-encoded range.
            - log: standard binary encoding. May come with degeneracy in the slack variable values when the to-be-encoded range does not equal 2^N-1 for some N. Scales logarithmically with the to-be-encoded range.
            - AP: Adapted Arithmetic Progression Encoding, basically taking ascending integers as coeffcients. May come with degeneracy in the slack variable values. Scales with the square root of the to-be-encoded range.
        - relative_precision (float): relative precision with which the slack term will approximate the constraint body of an inequality constraint. Default of precision=0 means no approximation. 
            The integer range to be encoded by the slack variables will generally be of the magnitude 1 / relative_precision.
            Note that non-zero precision obscures the objective function of the QUBO, as even feasible solutions might induce penalties of order u * relative_precision, where u is the upper bound.
            In order to make sure to improve the count of slack variables with higher slack_precision, it is compared to the case of slack_precision=0.
        - normalize_penalty_term (bool): If True, the penalty term will be normalized such that the smallest coefficient in the constraint body is scaled to 1. Default is False.
        - verbose: bool, for printing info. Default is False.
    Outputs:
        - penalty_term: Pyomo expr, to be added to the objective function with a penalty coefficient.
        - slack: Indexed Pyomo Variables which are the slack variables. If they are not needed, slack will we None.
    '''
    slack = None
    a, x, l, u, x_negation = normalize_constraint_sign(*decompose_linear_binary_constraint(constraint), normalize_coeffs_by_negation=True) # l <= ax <= u with l, u and a positive
    x = [1 - x_i if negated else x_i for x_i, negated in zip(x, x_negation)]
    constraint_body = sum(a_i * x_i for a_i, x_i in zip(a, x))

    # calculate normalization (in the final penalty term the smallest coefficient should be scaled to 1)
    smallest = min(min(a), np.inf if l is None else l, np.inf if u is None else u)

    # equality constraints
    if l is not None and l == u:
        if normalize_penalty_term:
            penalty_term = ((constraint_body - l) / smallest) ** 2
        else:
            penalty_term = (constraint_body - l) ** 2
    else: # inequality constraints
        scaling_base_index = np.argmax(a) # to measure
        scaling_base = a[scaling_base_index]

        # first test the waters with slack_precision=0
        try:
            a0, l0, u0 = clear_denominators(a, l, u) # normalize coefficients to integers so that we do not miss any of the below cases and are able to use slack terms for integer ranges
            upscaling0 = a0[scaling_base_index] / scaling_base
        except OverflowError:
            if relative_precision == 0:
                raise Exception('Coefficients of constraint {} have too many decimal places to be scaled to reasonable integers. Please increase the precision parameter.'.format(constraint.name))
            else:
                a0, l0, u0 = (None,) * 3
                upscaling0 = np.inf        
        if relative_precision == 0:
            a, l, u = a0, l0, u0
            upscaling = upscaling0
        else: # round corresponding to specified precision, but compare to the exact case to see if its better
            # digits = np.round(np.log10(1 / (2 * precision))).astype(int) # simple discrete version is rounding to this many digits
            largest = max(scaling_base, 0 if l is None else l, 0 if u is None else u) # find largest coefficient to calculate absolute precision
            precision = relative_precision * largest # absolute precision
            precision = np.round(precision, -int(np.floor(np.log10(abs(precision)))) + 3) # round precision to 3 significant digits to reduce many-digits expressions
            a = precision * np.round(np.array(a) / precision)
            if l is not None:
                l = precision * np.round(l / precision)
            if u is not None:
                u = precision * np.round(u / precision)
            
            a, l, u = clear_denominators(a, l, u)

            # compare to 0 precision case and take the better performing
            upscaling = a[scaling_base_index] / scaling_base
            if upscaling > upscaling0:
                a, l, u = a0, l0, u0
                upscaling = upscaling0
        if upscaling == 0:
            raise Exception('Approximation of constraint {} trivial. Please consider lowering the precision.'.format(constraint.name))

        # less-than-one-hot constraints \sum_i x_i <= 1 -> \sum_{i<j} x_i x_j
        if u == 1 and (l is None or l <= 0) and np.all(np.array(a) == 1):
            different_pairs = itertools.combinations(x, 2)
            penalty_term = 1 / upscaling**2 * sum(list(x*y for x,y in different_pairs))

        # general inequality constraints
        elif np.all(np.array(int(i) == i for i in a)) and (l is None or int(l) == l) and (u is None or int(u) == u): # try handle inequality constraints with slack variables
            slack_lower = 0 if l is None else l
            slack_upper = sum(a) if u is None else u
            if slack_upper == 0: # trivial constraint
                s = 0
                slack_penalty = 0
                if verbose:
                    print('Constraint {} seems to be trivial. Consider BLP-reduction.'.format(constraint.name))
            else:
                slack_range, slack_def, slack_penalty_def = binary_encoding(slack_encoding, bounds=(slack_lower, slack_upper))
                slack = pyo.Var(slack_range, within=pyo.Binary) # generate slack variables for the encoding
                slack_variable_name = set_new_attr_name(constraint.model(), constraint.name, 'slack_variable')
                setattr(constraint.model(), slack_variable_name, slack)
                s = slack_def(slack)
                slack_penalty = slack_penalty_def(slack)
                if verbose:
                    print('{} slack variables added for constraint {}'.format(len(slack.index_set()), constraint.name))
            if normalize_penalty_term:
                penalty_term = ((constraint_body - s / upscaling) / smallest) ** 2 + slack_penalty
            else:
                penalty_term = (constraint_body - s / upscaling)** 2 + slack_penalty
        else:
            raise Exception('Constraint of unsuitable form') 
    return penalty_term, slack


def binary_encoding(encoding: str, bounds: tuple[int, int], return_penalty: bool=True):
    '''
    Generates an a binary encoding for a given finite range of integers. In the arguments the type of encoding and the range can be specified.
    Inputs:
        - encoding (str): string out of ["sum", "one-hot", "log", "AP"], determining how slack variables, if needed, encode variable ranges. 
            - sum: sum of binary slack variables. May come with degeneracy in the slack variable values. Scales linearly with the to-be-encoded range.
            - one-hot: one-or-less encoding which constraints enforcing it. No degeneracy here". Scales linearly with the to-be-encoded range.
            - log: standard binary encoding. May come with degeneracy in the slack variable values when the to-be-encoded range does not equal 2^N-1 for some N. Scales logarithmically with the to-be-encoded range.
            - AP: Adapted Arithmetic Progression Encoding, basically taking ascending integers as coeffcients. May come with degeneracy in the slack variable values. Scales with the square root of the to-be-encoded range.
            - domain-wall: Domain-Wall encoding, similar to sum but "filling up" from the left side. Penalties are next neighbor couplings.
        - bounds(tuple[int, int]): 2-tuple (lower, upper), specifying lower and upper bounds of the encoded integer range. If upper < lower, the encoding will be in reverse.
        return_penalty (bool): specifying whether the last output should be a penalty term of the constraints enforcing the encoding or a constraint object.
    Returns:
        - binary_range: Iterator defining the number of binary variables needed to generate the encoding.
        - encoding_def: Function mapping a Pyomo Var object with indices in binary_range to a Pyomo expression representing the binary encoding.
        - depending on return_penalty, one of:
            - encoding_penalty_def: Function mapping a Pyomo Var object with indices in binary_range to a Pyomo expression that represents a penalty term for enforcing constraints requird for the definition of the encoding.
            - encoding_constraints_defs: List of functions mapping a Pyomo Var object with indices in binary_range to a Pyomo expression that represents a constraint to be added to the model to enforce the correct encoding definition. None if no constraints are necessary
    '''
    lower, upper = bounds
    reverse = upper < lower # indicating that we want to encode the range in the reverse way
    if reverse:
        lower, upper = upper, lower
    if encoding == 'sum':
        num_bits =  upper - lower
        binary_range = range(1, num_bits + 1)
        encoding_def = lambda var: sum(var[i] for i in binary_range) + lower
        encoding_penalty_def = lambda var: 0 # no extra encoding constraints needed here
        encoding_constraints_defs = [] # no extra encoding constraints needed here
    elif encoding == 'one-hot':
        num_bits =  upper - lower
        binary_range = range(1, num_bits + 1)
        encoding_def = lambda var: sum(i * var[i] for i in binary_range) + lower
        # penalty for less-than-one-hot constraint: \sum_i s_i <= 1 with coefficients weighted by the square of much they change the encoding expression
        ####             adapt that coefficient once known how to choose it             #################
        encoding_penalty_def = lambda var: sum(list(x * y for (n, x), (m, y) in itertools.combinations(var.items(), 2)))
        encoding_constraints_defs = [lambda var: sum(var[i] for i in binary_range) <= 1]
    elif encoding == 'log':
        num_bits = int(np.floor(np.log2(upper - lower))) + 1
        binary_range = range(num_bits)
        encoding_def = lambda var: sum(2**i * var[i] for i in binary_range[:-1]) + (upper - lower - 2**(num_bits - 1) + 1) * var[num_bits - 1] + lower
        encoding_penalty_def = lambda var: 0 # no extra encoding constraints needed here
        encoding_constraints_defs = [] # no extra encoding constraints needed here
    elif encoding == 'AP':
        num_bits = int(np.ceil(1 / 2 + np.sqrt(upper - lower - 3 / 4)))
        binary_range = range(1, num_bits + 1)
        encoding_def = lambda var: sum(i  * var[i] for i in binary_range[:-1]) + (upper - lower - num_bits * (num_bits - 1) / 2) * var[num_bits] + lower
        encoding_penalty_def = lambda var: 0 # no extra encoding constraints needed here
        encoding_constraints_defs = [] # no extra encoding constraints needed here
    elif encoding == 'domain-wall':
        num_bits =  upper - lower
        binary_range = range(1, num_bits + 1)
        encoding_def = lambda var: sum(var[i] for i in binary_range) + lower
        encoding_penalty_def = lambda var: sum(var[i - 1] * (1 - var[i]) for i in binary_range[1:])
        def next_neighbor_constraint_generator(i):
            return lambda var: var[i] <= var[i - 1]
        encoding_constraints_defs = [next_neighbor_constraint_generator(i) for i in binary_range[1:]]
        
    else:
        raise Exception('Specify compatible encoding')
    if reverse:
        unflipped_encoding_def = encoding_def
        encoding_def = lambda var: upper - unflipped_encoding_def(var) + lower
    if return_penalty:
        return binary_range, encoding_def, encoding_penalty_def
    else: 
        return binary_range, encoding_def, encoding_constraints_defs

def fractional_binary_encoding(encoding: str, precision: int):
    '''
    Generates an a binary encoding an interval [0, 1) of a given precision of the range of integers. Next to he precision the type of encoding can be specified in the arguments.
    Inputs:
        - encoding (str): string out of ["sum", "one-hot", "log", "AP"], determining how slack variables, if needed, encode variable ranges. 
            - sum: sum of binary slack variables. May come with degeneracy in the slack variable values. Scales linearly with the to-be-encoded range.
            - one-hot: one-or-less encoding which constraints enforcing it. No degeneracy here". Scales linearly with the to-be-encoded range.
            - log: standard binary encoding. May come with degeneracy in the slack variable values when the to-be-encoded range does not equal 2^N-1 for some N. Scales logarithmically with the to-be-encoded range.
            - AP: Adapted Arithmetic Progression Encoding, basically taking ascending integers as coeffcients. May come with degeneracy in the slack variable values. Scales with the square root of the to-be-encoded range.
        - precision: precision of the fractional binary encoding, For the 'log'-encoding, this will introduce floor(log_2(precision)) + 1 additional variables.
    Returns:
        - encoding_range: Iterator defining the number of binary variables needed to generate the encoding.
        - encoding_def: Function mapping a Pyomo Var object with indices in binary_range to a Pyomo expression representing the binary encoding.
        - encoding_constraints_defs: List of functions mapping a Pyomo Var object with indices in binary_range to a Pyomo expression that represents a constraint to be added to the model to enforce the correct encoding definition. None if no constraints are necessary
    '''
    lower = 0
    if encoding in ['sum', 'one-hot', 'domain-wall']: # non-adaptive encodings
        upper = precision
    elif encoding == 'log':
        upper = int(2**np.ceil(np.log2(precision + 1)) - 1) # extend to a value for which the logarithmic encoding is non-adaptive but non-degenerate
    # AP not yet implemented here
    else:
        raise Exception('Specify compatible slack_encoding')
    encoding_range, encoding_def_unscaled, encoding_constraints_defs = binary_encoding(encoding, bounds=(lower, upper), return_penalty=False)
    encoding_def = lambda var: encoding_def_unscaled(var) / (upper + 1)
    return encoding_range, encoding_def, encoding_constraints_defs


def validate_penalty_term(penalty_term, constraint, tol=1e-06):
    variables = list(identify_variables(constraint.body, include_fixed=False))
    assert np.all([variable.is_binary() for variable in variables]), 'Considered constraints must have exclusively binary variables.'
    slack_variables = [var for var in list(identify_variables(penalty_term, include_fixed=False)) if var.name not in [variable.name for variable in variables]]
    # Generate all possible combinations of binary values
    for values in itertools.product([0, 1], repeat=len(variables)):
        # Assign values to variables
        for variable, val in zip(variables, values):
            variable.fix(val)
        body_value = pyo.value(constraint.body, exception=False)
        constraint_feasible = pui._check_infeasible(constraint, body_value, tol) == 0
        if len(slack_variables) > 0: # case when we have variables in the penalty but not in the constraint
            penalty_term_feasible = False
            for slack_values in itertools.product([0, 1], repeat=len(slack_variables)):
                # Assign values to slack variables
                for slack_variable, slack_val in zip(slack_variables, slack_values):
                    slack_variable.fix(slack_val)
                penalty_term_value = pyo.value(penalty_term)
                if penalty_term_value == 0:
                    penalty_term_feasible = True
                    break
        else:
            penalty_term_value = pyo.value(penalty_term)
            penalty_term_feasible = (penalty_term_value == 0)
        var_values_dict = dict(zip([var.name for var in variables], values))
        assert penalty_term_value >= 0, 'Penalty Term has negative value as {}.'.format(var_values_dict)
        if constraint_feasible != penalty_term_feasible:
            print('Constraint {} {} but penalty term value {} for values {}.'.format(constraint.name, 'un'*(not constraint_feasible) + 'satisfied', 'zero' if penalty_term_feasible else 'positive', var_values_dict))
    # unfix variables again
    for variable in variables + slack_variables:
        variable.unfix()
    print('validation of penalty of {} complete'.format(constraint.name))

def factor_expression(expr):
    """Recursively factor a Pyomo expression into a weighted sum of monomials.
    Inputs:
        expr: Pyomo expression
    Output:
        factored_terms: list of tuples (representing summands of the expression) of the form (coef, vars), where coef are coefficients and vars lists containing the variables present in each summand.
        """
    linearly_decomposed_term = decompose_term(expr)
    if linearly_decomposed_term[0]: # linear term
        return simplify_terms([(coef, [var] if var is not None else []) for coef, var in linearly_decomposed_term[1]], reduce_higher_order_binaries=False)
    expr_type, reduced_expr = expr.__reduce__()[1:]
    if expr_type[0] == pyomo.core.expr.numeric_expr.SumExpression:
        terms = []
        for summand in reduced_expr[0]:
            terms.extend(factor_expression(summand))
        return simplify_terms(terms, reduce_higher_order_binaries=False)
    elif expr_type[0] in [pyomo.core.expr.numeric_expr.ProductExpression, pyomo.core.expr.numeric_expr.NegationExpression]:
        # Distribute multiplication over sums
        sign = -1 if expr_type[0] == pyomo.core.expr.numeric_expr.NegationExpression else +1
        factored_terms = [(sign, [])]  # List of ( coefficient, variable list)
        for factor in reduced_expr[0]:
            new_factors = factor_expression(factor)  # Get factors of this term
            expanded_terms = []
            for coef1, vars1 in factored_terms:
                for coef2, vars2 in new_factors:
                    expanded_terms.append((coef1 *coef2, vars1 + vars2))
            factored_terms = expanded_terms
        return simplify_terms(factored_terms, reduce_higher_order_binaries=False)
    elif expr_type[0] == pyomo.core.expr.numeric_expr.PowExpression:
        for base, power in reduced_expr:
            assert isinstance(power, int), 'Non-integer power encountered in expression {}'.format(str(expr))
            product_expr = 1
            for k in range(power):
                product_expr *= base
            return simplify_terms(factor_expression(product_expr), reduce_higher_order_binaries=False)
    else:
        raise ValueError(f"Unsupported expression type: {expr}")
    
def factored_terms_to_expr(factored_terms):
    expr = 0
    for coef, vars in factored_terms:
        if vars:
            expr += coef * pyo.prod(vars)
        else:
            expr += coef
    return expr

def normalize_vars(vars_list, coef, reduce_fixed_variables=False, reduce_higher_order_binaries=True):
    """
    Normalize the variable list based on the given boolean arguments.
    
    - `reduce_fixed_variables`: If True, fixed variables are removed, and the coefficient is updated.
    - `reduce_higher_order_binaries`: If True, duplicate binary variables are merged (e.g., x² → x for binary x).
    - Variable order is sorted to ensure commutativity.
    Returns:
        - Updated coefficient after handling fixed variables.
        - A tuple of variable IDs for hashing (sorted).
        - A corresponding list of original variables (sorted accordingly).
    """
    processed_vars = []
    original_vars = []  # Stores original variable objects
    seen_binaries = set()

    for v in vars_list:
        if reduce_fixed_variables and v.is_fixed():
            coef *= v.value  # Multiply coefficient by fixed variable value
            continue  # Remove fixed variable from list
        else:
            var_id = id(v)  # Use ID as a hashable key
            if reduce_higher_order_binaries and v.is_binary():
                if var_id in seen_binaries:
                    continue  # Skip duplicate binary variables
                seen_binaries.add(var_id)
            processed_vars.append(var_id)  # Store ID for hashing
            original_vars.append(v)  # Keep variable for output
    # Sort variables by name to ensure commutativity (so [x, y] == [y, x])
    sorted_indices = sorted(range(len(processed_vars)), key=lambda i: str(original_vars[i].name))
    processed_vars = tuple(processed_vars[i] for i in sorted_indices)  # Use only IDs for hashing
    original_vars = [original_vars[i] for i in sorted_indices]  # Keep sorted original variables
    return coef, tuple(processed_vars), original_vars  # Return updated coefficient, hashable key, and sorted original variables

def simplify_terms(factored_terms, reduce_fixed_variables=False, reduce_higher_order_binaries=True):
    """
    Reduce a polynomial-like expression represented as (coefficient, variable list), as the output of factor_expression.
    Supports:
    - `reduce_fixed_variables`: If True, removes fixed variables and updates coefficients.
    - `reduce_higher_order_binaries`: If True, merges duplicate binary variables.
    """
    term_dict = defaultdict(float)  # Maps from hashable keys to summed coefficients
    original_var_map = {}  # Maps from hashable keys to original variable lists

    for coef, vars_list in factored_terms:
        new_coef, norm_vars, original_vars = normalize_vars(vars_list, coef, reduce_fixed_variables, reduce_higher_order_binaries)
        term_dict[norm_vars] += new_coef  # Accumulate coefficient
        original_var_map[norm_vars] = original_vars  # Store original vars

    # Convert back to required output format with original Pyomo variables
    simplified_factored_terms = [(coef, original_var_map[vars]) for vars, coef in term_dict.items() if coef != 0]
    
    return simplified_factored_terms

def pyomo_QUBO_to_array(QUBO_expr):
    '''Converts a Pyomo expression (e.g. the objective) to an QUBO array and a additive constant.
    Input:
        QUBO_expr: Pyomo expression of at most quadratic order.
    Outputs:
        variable_names: sorted list of variable names, determining the QUBO-matrix ordering.
        QUBO_array: upper triangular QUBO matrix
        C: float, constant additive term of the QUBO.
    '''
    if isinstance(QUBO_expr, pyomo.core.base.PyomoModel.ConcreteModel):
        QUBO_expr = list(QUBO_expr.component_objects(pyo.Objective, active=True))[0]
    if isinstance(QUBO_expr, pyomo.core.base.objective.ScalarObjective):
        QUBO_expr = QUBO_expr.expr
    assert np.all([var.is_binary() for var in identify_variables(QUBO_expr, include_fixed=False)]), 'QUBO includes non-binary variables.'
    # extracting coefficients:
    repn = generate_standard_repn(QUBO_expr, quadratic=True) # apparently for more performance PyomoNLP can be used instead
    assert repn.is_constant() or repn.is_linear() or repn.is_quadratic(), 'Provided QUBO expression is of higher order > 2.'
    var_names_and_ids = sorted([(var.name, id(var)) for var in identify_variables(QUBO_expr, include_fixed=False)])
    N = len(var_names_and_ids)
    var_order_dict = dict((var_id, i) for i, (var_name, var_id) in enumerate(var_names_and_ids))
    QUBO_array, C = np.zeros(shape=(N,N)), repn.constant
    # add linear terms:
    for coef, var in zip(repn.linear_coefs, repn.linear_vars):
        i = var_order_dict[id(var)]
        QUBO_array[i, i] += coef
    # add quadratic_terms:
    for coef, (var1, var2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
        i, j = sorted((var_order_dict[id(var1)], var_order_dict[id(var2)])) # sorting indices for upper triangular matrix
        QUBO_array[i,j] += coef
    variable_names = list(zip(*var_names_and_ids))[0] if var_names_and_ids else []
    return variable_names, QUBO_array, C

def binary_discretization(
    model: pyo.ConcreteModel, 
    bounds: Union[
        Tuple[Optional[float], Optional[float]], 
        Dict[str, Tuple[Optional[float], Optional[float]]]
    ] = (None, None), 
    precision: Union[
        Union[int, float],
        Dict[str, Union[int, float]]
    ] = 1, 
    encoding: Union[
        str,
        Dict[str, str]
    ] = 'log', 
    reversed: Union[
        bool,
        Dict[str, bool]
    ] =     False, 
    reduce_model: bool = False,
    verbose: Union[bool, int]=False
) -> pyo.ConcreteModel:
    '''Transforms a model with real or integer variables into a binary program.
    Inputs:
        model: a Pyomo model.
        bounds: tuple or dict of variable names and tuples, representing, lower and upper bounds for the variables, additional to already specified variable bounds within Pyomo.
        precision: float or dictionary of variable names as keys and floats as values, representing step size of the discrete encoding. Default is 1, which means we have integer accuracy.
        encoding: str or dictionary of variable names as keys and strings as values, out of ['log', 'sum', 'one-hot', 'domain-wall'], specifying the mode of encoding using binary variables. Default is 'log', i.e. logarithmic encoding.
        reversed: bool or dictionary of variable names as keys and bools as values. Specifies whether the encoding is reversed (from upper to lower bound). This matters when sub-integer precision > 0 is used, as then the last value (lower for reversed=True) can not be represented.
        reduce_model: bool specifying wether to apply the reduce_BLP function to possibly reduce problem complexity. This may deactivate some constraints and fix some variables in the input model. Default is True
        verbose: bool or int, for printing info. Default is False.
    Output:
        model: Discretized Pyomo model with only binary variables.
            To keep track of the encodings of variables, it features a dictionary attribute "encoding_expressions" with variables names as keys and pyomo expressions as values.
            These expressions are with respect to the binary discretization variables.
    '''
    variables_to_be_discretized =  [var for var in model.component_data_objects(pyo.Var, active=True) if not (var.is_binary() or var.is_fixed())]
    bounds_is_dict = isinstance(bounds, dict)
    if bounds_is_dict:
        bounds_dict = bounds
    precision_is_dict = isinstance(precision, dict)
    if precision_is_dict:
        precision_dict = precision
    encoding_is_dict = isinstance(encoding, dict)
    if encoding_is_dict:
        encoding_dict = encoding
    reversed_is_dict = isinstance(reversed, dict)
    if reversed_is_dict:
        reversed_dict = reversed
    def relation_not_none(a, b, relation=max):
        if a is None and b is None:
            raise ValueError("Please provide non-None variable-specific or general bound.")
        return max(filter(lambda x: x is not None, (a, b)))
    encoding_vars_added = 0
    encoding_expressions = {}  # to keep track of the variable encodings
    # begin iteration over variables
    for variable in variables_to_be_discretized:
        if bounds_is_dict:
            bounds = bounds_dict[variable.name] 
        if reversed_is_dict:
            reversed = reversed_dict[variable.name]
        if precision_is_dict:
            precision = precision_dict[variable.name]
        if encoding_is_dict:
            encoding = encoding_dict[variable.name]
        lower = relation_not_none(bounds[0], variable.lb, relation=max)
        upper = relation_not_none(bounds[1], variable.ub, relation=min)
        if variable.domain == pyo.PositiveReals and lower == 0:
            lower = 1e-6 # to avoid that the lower bound is not representable in the encoding
        if precision > upper - lower:
            if variable.domain == pyo.PositiveReals and lower <= 1e-6:
                fix_value = upper
            else:
                assert lower is not None
                fix_value = lower
                variable.fix(fix_value)
            if verbose > 1:
                print(
                    'Variable {} fixed to {} since precision of {} too large for bounds ({}, {})'.format(
                        variable.name, fix_value, precision, lower, upper
                    )
                )
            continue
        assert 0 < precision <= upper - lower and (not variable.is_integer() or precision == int(precision)), (
            'Precision {} not compatible with {}-variable {} with bounds ({}, {}).'
            'Please specify a different precision for this variable.'
            ).format(
                precision, variable.domain, variable.name, lower, upper
            )
        
        # normalize bounds to multiples of precision
        normalized_lower, normalized_upper = precision * np.ceil(lower / precision), precision * np.floor(upper / precision)
        # scale up to integers
        normalized_lower, normalized_upper, upscaled_precision = clear_denominators(normalized_lower, normalized_upper, precision)
        normalized_lower, normalized_upper = int(normalized_lower), int(normalized_upper)
        upscaling = upscaled_precision / precision  # we later need to divide by this again

        if verbose > 1 and variable.domain == pyo.PositiveReals and normalized_lower == 0 and not reversed:
            print('Option reversed=True for variable {} is recommended. This will make sure that the 0 is excluded.'.format(variable.name))
        if reversed: # reverse encoding
            normalized_lower, normalized_upper = normalized_upper, normalized_lower

        encoding_var_range, encoding_def_upscaled, encoding_constraints_defs = binary_encoding(encoding, bounds=(normalized_lower, normalized_upper), return_penalty=False)

        encoding_def = lambda e: encoding_def_upscaled(e) / upscaling

        # add encoding variables
        encoding_vars = pyo.Var(encoding_var_range, within=pyo.Binary)
        encoding_name = set_new_attr_name(model, variable.name, 'encoding_variable')
        setattr(model, encoding_name, encoding_vars)
        if encoding_constraints_defs:
            encoding_constraints = pyo.Constraint(range(len(encoding_constraints_defs)), rule=lambda model, i: encoding_constraints_defs[i](encoding_vars))
            encoding_constraints_name = set_new_attr_name(model, variable.name, 'encoding_constraints')
            setattr(model, encoding_constraints_name, encoding_constraints)
        e = encoding_def(encoding_vars)
        encoding_vars_added += len(encoding_var_range)
        encoding_expressions[variable.name] = e
        # substitute variable appearences in objective and constraints
        if variable.domain == pyo.PositiveReals and lower == 0:
            fix_value = upper
        else:
            assert lower is not None
            fix_value = lower    
        substitute_variable_by_expression(model, variable, e, fix_value)

    if verbose:
        if encoding_vars_added > 0:
            print('{} binary encoding variables added'.format(
                encoding_vars_added
            ))
        print('{} variables fixed'.format(len(variables_to_be_discretized)))
    setattr(model, "encoding_expressions", encoding_expressions)  # to use this data
    # reduce model complexity by removing trivial constraints and fixing variables
    if reduce_model:
        model, fix_variables = reduce_BLP(model, verbose=verbose)
    return model

def substitute_variable_by_expression(model, variable, expression, fix_value=None):
    sub_map = {id(variable): expression}
    visitor = ExpressionReplacementVisitor(substitute=sub_map)
    # objective:
    for objective in model.component_objects(pyo.Objective, active=True):
        objective.expr = visitor.walk_expression(objective.expr)
    # constraints
    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        constraint.set_value((constraint.lower, visitor.walk_expression(constraint.body), constraint.upper))
    # fix variable
    if fix_value is None:
        if variable.domain == pyo.PositiveReals and variable.lb == 0:
            fix_value = variable.ub
        else:
            assert variable.lb is not None
            fix_value = variable.lb
    variable.fix(fix_value)

def MILP_to_QUBO(model, non_binary_variable_bounds, discretization_precision=1, encoding='log', penalties=None, delta=0.2, slack_encoding='log', reduce_model=True, verbose=False):
    '''Converts a Pyomo Mixed Integer linear Programm model to a Pyomo QUBO model by first discretizing non-binary variables and then substituting constraints for penalty terms.
        Inputs:
                model: Pyomo model in MILP-form.
                non_binary_variable_bounds: tuple, lower and upper bounds for the non-binary variables, additional to already specified variable bounds within Pyomo.
                discretization_precision of the discretization. Default is 1, which means we have integer accuracy.
                encoding: str out of ['log', 'sum'], specifying the mode of encoding using binary variables. Default is 'log', i.e. logarithmic encoding.
                penalties: penalty coefficient for the penalty terms. Either a universal coefficient, given by a float, or a dictionary with constraint names as keys and penalties as values. Default is None, which will run the big heuristic for finding a penalty.
                delta: float needed to run the big-M heuristic as the standard argument for penalties. If penalties=None is chosen, delta will be added to the penalty. Default is 0.2.
                slack_encoding: sting out of ["sum", "one-hot", "log"], determining how slack variables, if needed, encode variable ranges. The options "sum" and "log" may come with degeneracy in the slack variable values. Meanwhile "one-hot" comes with additional penalty terms to enforce one-or-less constraints on the slack variables. Default is "log".
                reduce_model: bool specifying wether to apply the reduce_BLP function to possibly reduce problem complexity. This may deactivate some constraints and fix some variables in the input model. Default is True
                verbose: int, for printing info. True or 1 provides moderate detail, while 2 provides high detail. Default is 0.
        Output:
            QUBO_model: Concrete Pyomo model with QUBO as objective functions. The constraints and variables of the input model (discretized) are still there, but set inactive or fixed. QUBO_model contains an object penatly_terms, which contains all the information about which penalty terms and coefficients belong to which constraint.
    '''
    BLP_model = binary_discretization(model, non_binary_variable_bounds, precision=discretization_precision, encoding=encoding, reduce_model=False, verbose=verbose)
    QUBO_model = BLP_to_QUBO(BLP_model, penalties=penalties, delta=delta, slack_encoding=slack_encoding, reduce_model=reduce_model, verbose=verbose)
    return QUBO_model

def binary_constraints_to_filter(model, variables=None, tol=1e-6, clone_model=True, verbose=False):
    '''Create a boolean filter for the sampleset of a QUBO solution by D-Wave.
    Input:
        - model: Pyomo model with (active) constraints to check the samples on
        - variables: list of str, names of the variables in model that the sample values correspond to. Default is None, which just takes the alphabetic ordering of all nonfixed variables.
        - tol: float, tolerance for comparing values. Default is 1e-6.
        - clone_model: bool, specifying whether the model that is used to generate the filter should be a clone of the input model. Use this option if the variables in the inout model should not be fixed.
        - verbose: bool for printing info
    Output: 
        - filter: Boolean functions to be evaluated on samples. The recommended use is sampleset.filter(filter).
    '''
    a = 0 # measure time
    if clone_model:
        m = model.clone()
    else:
        m = model
    constraints = list(m.component_data_objects(pyo.Constraint, active=True))
    for constraint in constraints:
        assert np.all([var.is_binary() for var in identify_variables(constraint.body, include_fixed=False)]), 'Constraints must have exclusively binary variables.'
        #a +=
    if variables is None: # revert to standard variable order
        variables = [var.name for var in m.component_data_objects(pyo.Var, active=True) if not var.is_fixed()]
    # Cache variable objects
    var_dict = {var_name: get_component_by_name(m, var_name) for var_name in variables}

    # define filter
    def filter(sample):
        feasibles = {}
        for n, var_name in enumerate(variables):
            var = var_dict.get(var_name)
            if var is None:
                pass
            else:
                var.fix(int(sample.sample[n]))
        for constraint in constraints:
            try:
                if pyo.value(constraint.body) is None:
                    if verbose: 
                        print('Not all variables relevant to constraint {} fixed. Skipping constraint'.format(constraint.name))
                    feasible = True
                else:
                    feasible = pui._check_infeasible(constraint, pyo.value(constraint.body), tol) == 0
                    if verbose and not feasible:
                        print('Constraint {} violated. Perhaps increase penalty.'.format(constraint.name))
            except ValueError:
                if verbose: print('Not all variables relevant to constraint {} fixed. Skipping constraint'.format(constraint.name))
                feasible = True
            feasibles[constraint.name] = feasible
        return np.all(list(feasibles.values()))
    return filter

def true_objective_function(model, variables=None, clone_model=True, verbose=False):
    '''Create function mapping samples of a sampleset to the true objective value according to the constraint unapproximated model. This only makes sense for feasible samples, so consider filtering with binary_constraints_to_filter(model) first.
    Input:
        - model: Pyomo model with (active) objective to evaluate samples on
        - variables: list of str, names of the variables in model that the sample values correspond to. Default is None, which just takes the alphabetic ordering of all nonfixed variables.
        - clone_model: bool, specifying whether the model that is used to generate the filter should be a clone of the input model. Use this option if the variables in the inout model should not be fixed.
        - verbose: bool for printing info
    Output: 
        - true_objective: Function mapping samples in sampleset.samples() to their true objective value
    '''
    if clone_model:
        m = model.clone()
    else:
        m = model
    if variables is None: # revert to standard variable order
        variables = [var.name for var in m.component_data_objects(pyo.Var, active=True) if not var.is_fixed()]
    # Cache variable objects
    var_dict = {var_name: get_component_by_name(m, var_name) for var_name in variables}

    #define function
    def true_objective(sample):
        for n, var_name in enumerate(variables):
            var = var_dict.get(var_name)
            if var is None:
                pass
            else:
                var.fix(int(sample[n]))
        objective_function = list(m.component_objects(pyo.Objective, active=True))[0]
        try:
            objective = pyo.value(objective_function)
        except ValueError:
            if verbose: print('Not all variables relevant to objective {} fixed. Returning None'.format(objective_function.name))
            objective = None
        return objective
    return true_objective
    
def optimality_filter(model, optimal_objective, variables=None, epsilon=0, tol=1e-6, clone_model=True, verbose=False):
    '''Create a boolean filter for epsilon-optimality of a sampleset of a QUBO solution by D-Wave.
    Input:
        - model: Pyomo model with (active) constraints to check the samples on
        - variables: list of str, names of the variables in model that the sample values correspond to. Default is None, which just takes the alphabetic ordering of all nonfixed variables.
        - epsilon: float, tolerance for objective value comparison, in addition to tol. Default is 0.
        - tol: float, tolerance for comparing values. Default is 1e-6.
        - clone_model: bool, specifying whether the model that is used to generate the filter should be a clone of the input model. Use this option if the variables in the inout model should not be fixed.
        - verbose: bool for printing info
    Output: 
        - filter: Boolean functions to be evaluated on samples. The recommended use is sampleset.filter(filter).
    '''
    if clone_model:
        m = model.clone()
    else:
        m = model
    constraints = list(m.component_data_objects(pyo.Constraint, active=True))
    for constraint in constraints:
        assert np.all([var.is_binary() for var in identify_variables(constraint.body, include_fixed=False)]), 'Constraints must have exclusively binary variables.'
    if variables is None: # revert to standard variable order
        variables = [var.name for var in m.component_data_objects(pyo.Var, active=True) if not var.is_fixed()]
    # Cache variable objects
    var_dict = {var_name: get_component_by_name(m, var_name) for var_name in variables}

    #define filter
    def filter(sample):
        feasibles = {}
        for n, var_name in enumerate(variables):
            var = var_dict.get(var_name)
            if var is None:
                pass
            else:
                var.fix(sample.sample[n])
        for constraint in constraints:
            try:
                if pyo.value(constraint.body) is None:
                    if verbose: print('Not all variables relevant to constraint {} fixed. Skipping constraint'.format(constraint.name))
                    feasible = True
                else:
                    feasible = pui._check_infeasible(constraint, pyo.value(constraint.body), tol) == 0
                    if verbose and not feasible:
                        print('Constraint {} violated. Perhaps increase penalty.'.format(constraint.name))
            except ValueError:
                if verbose: print('Not all variables relevant to constraint {} fixed. Skipping constraint'.format(constraint.name))
                feasible = True
            feasibles[constraint.name] = feasible

        objective_function = list(m.component_objects(pyo.Objective, active=True))[0]
        try:
            objective = pyo.value(objective_function)
            assert objective >= optimal_objective
            objective_optimal = abs(objective - optimal_objective) <= epsilon + tol
        except ValueError:
            if verbose: print('Not all variables relevant to objective {} fixed. Ignoring objective'.format(objective_function.name))
            objective_optimal = True
        return np.all(list(feasibles.values())) and objective_optimal
    return filter

def calculate_QUBO_density(QUBO, tol=1e-8):
    '''Returns the density of a QUBO in array form, which is the relative number of nondiagonal non-zero components.'''
    values = QUBO[np.triu_indices(len(QUBO),1)]
    density = np.count_nonzero(abs(values) > tol) / len(values) if len(values) else 0
    connections_per_variable = [np.count_nonzero([abs(QUBO[i, j]) > tol for j in range(len(QUBO)) if i < j]) + np.count_nonzero([abs(QUBO[j, i]) > tol for j in range(len(QUBO)) if j < i]) for i in range(len(QUBO))]
    if connections_per_variable:
        mean_connections, max_connections = np.mean(connections_per_variable), np.max(connections_per_variable)
    else:
        mean_connections, max_connections = None, None
    return density, mean_connections, max_connections 





# old functions I have not quite given up on:
def check_trivial_constraint_old(constraint, verbose=False):
    '''Checks if a linear binary-variable-containing constraint is trivial and determines whoch variables should be fixed.
    Inputs:
        - constraint: A linear constraint with binary variables of a Pyomo model
        - verbose: bool, for printing info. Default is False.
    Outputs:
        - constraint_trivial: bool indicating if the constraint is trivial and can be removed (if some variables are fixed)
        - fix_variable_tuples: list of pairs of variables and its values to be assigned if the constraint is removed
        - add_constraints = list of tuples that serve as inputs for construct_linear_constraint to construct a linear binary constraint. If the input constraint is removed, these constraints have to be added.
    '''
    a, x, l, u = decompose_linear_binary_constraint(constraint) # l <= ax <= u
    keep_checking_lower = l is not None # keep checking conditions for lower bound until constraints can't be reduced more
    keep_checking_upper = u is not None # keep checking conditions for upper bound until constraints can't be reduced more
    constraint_trivial =  False
    fix_variable_tuples = []
    add_constraints = []
    add_to_add_constraints = None # Here I keep track of which bound I want to add to the list
    while keep_checking_lower or keep_checking_upper:
        fix_var_len = len(fix_variable_tuples)
        min_a, max_a = sum(e for e in a if e < 0), sum(e for e in a if e > 0)
        # start checking for trivial constraints:
        # lower bound not low enough
        if len(a) == 1 and (l is not None and min_a < l and (u is None or u >= max_a)):
            fix_variable_tuples +=[(x[0], int(a[0] > 0))]
            constraint_trivial = True
            add_to_add_constraints = None
            keep_checking_lower = keep_checking_upper = False # don't need to keep checking then
        # upper bound not large enough
        elif len(a) == 1 and (u is not None and u < max_a and (l is None or l <= min_a)):
            fix_variable_tuples += [(x[0], int(a[0] < 0))]
            constraint_trivial = True
            add_to_add_constraints = None
            keep_checking_lower = keep_checking_upper = False # don't need to keep checking then
        # efective range in between constraint bounds
        elif (l is None or min_a >= l) and (u is None or max_a <= u):
            constraint_trivial = True
            add_to_add_constraints = None
            keep_checking_lower = keep_checking_upper = False # don't need to keep checking then
            u = l = None # constraint can be removed without adding any other
        elif (l is None or min_a >= l) and keep_checking_lower:
            keep_checking_lower = False
            add_to_add_constraints = 'upper'
            l = None # lower bound can be removed
        elif (u is None or max_a <= u) and keep_checking_upper:
            keep_checking_upper = False
            add_to_add_constraints = 'lower'
            u =  None # upper bound can be removed
        else:
            # check if single variables can be fixed in constraints
            for i, b in enumerate(a): # split a in single part b and rest
                a_new = a[:i] + a[i+1:]
                x_new = x[:i] + x[i+1:]
                min_a_new, max_a_new = sum(e for e in a_new if e < 0), sum(e for e in a_new if e > 0)
                # (large lower bound forces b>0 to participate in the sum) or (small upper bound forces b<0 to participate in the sum)
                if (keep_checking_lower and max_a_new < l) or (keep_checking_upper and min_a_new > u): # case b<0 (b>0) excluded by assertion l<=max_a (min_a<=u) in decompose_linear_binary_constraint
                    fix_variable_tuples += [(x[i], 1)]
                    a, x, l, u = a_new, x_new, None if l is None else l - b, None if u is None else u - b # keep checking with this reduced constraint info
                    break # stop here and go from top
                # (small upper bound forces b>0 to be excluded) or (large lower bound forces b<0 to be excluded)
                elif (keep_checking_upper and min_a_new + b > u) or (keep_checking_lower and max_a_new + b < l): # case b<0 (b>0) excluded by assertion min_a<=u (l<=max_a) in decompose_linear_binary_constraint
                    fix_variable_tuples += [(x[i], 0)]
                    a, x = a_new, x_new # keep checking with this reduced constraint info
                    break # stop here and go from top
                else:
                    pass
            if len(fix_variable_tuples) == fix_var_len:
                keep_checking_lower = keep_checking_upper = False # don't get stuck in infinite loop if nothing ever happens
    # check here if trivial constraint with fixed variable is actually feasible. Example pyo.inequality(-2, -10 * model.x[0] - 10 * model.x[1] - 0.5*model.x[3], -1)

    # Once all other triviality options are exhausted, check if variables can be can be removed from constraints (but not fixed)
    # normalize constraint such that upper bound and coefficients a are positive. But we need to keeo track which bound corresponds to which the original constraint bound
    bounds_flipped = (l is None or l < 0) and (u is None or u < 0) # this indicates that in the following normalized form of constraints the upper bound corresponds to the lower
    a, x, l, u, x_negation = normalize_constraint_sign(a, x, l, u, normalize_coeffs_by_negation=True) 
    keep_checking_upper = u is not None
    keep_checking_lower = l is not None
    large_coeff_bound = np.max(np.abs(a))
    while keep_checking_lower or keep_checking_upper:
        large_coeff_indices = np.where(np.abs(a) >= large_coeff_bound)[0]
        # mask for generating the complement
        mask = np.ones(len(a), dtype=bool)
        mask[large_coeff_indices] = False
        # generate a_new and its complement
        a_new = tuple(np.array(a)[large_coeff_indices])
        x_new = tuple(np.array(x)[large_coeff_indices])
        x_negation_new = tuple(np.array(x_negation)[large_coeff_indices])
        bs = tuple(np.array(a)[mask])
        if len(bs) == 0:
            break
        relevance_bs =  sum(bs) # this is a measure for how relevant the bs are, assuming all bs are positive
        gcd_value = gcd_decimal(*a_new)
        relevance_bound_lower = mod_decimal(l, gcd_value) if keep_checking_lower and gcd_value > 0 else 0 # the relevance of bs has to be larger than this for them to be relevant for the lower bound (this bound can be made lower, but would require the solution of a Knappsack problem)
        relevance_bound_upper = mod_decimal(u, gcd_value) if keep_checking_upper and gcd_value > 0 else 0 # the relevance of bs has to be larger than this for them to be relevant for the upper bound (this bound can be made lower, but would require the solution of a Knappsack problem)
        if relevance_bs <= relevance_bound_upper:
            keep_checking_upper = False
            if  relevance_bs <= relevance_bound_lower and relevance_bs < l:
                keep_checking_lower = False
                # add constraint l <= a_new*x_new <= u
                add_constraints += [unnormalize_negated_constraint(a_new, x_new, l, u, x_negation_new)] # 
                add_to_add_constraints = None
                l = u = None
            else:
                # add constraint a_new*x_new <= u
                add_constraints += [unnormalize_negated_constraint(a_new, x_new, None, u, x_negation_new)]
                if l is not None:
                    add_to_add_constraints = 'upper' if bounds_flipped else 'lower'
                elif (not bounds_flipped and add_to_add_constraints == 'upper') or (bounds_flipped and add_to_add_constraints == 'lower'):
                    add_to_add_constraints = None # just aded it
                u = None
        elif relevance_bs <= relevance_bound_lower and relevance_bs < l:
            keep_checking_lower = False
            # add constraint l <= a_new*x_new
            add_constraints += [unnormalize_negated_constraint(a_new, x_new, l, None, x_negation_new)]
            if u is not None:
                add_to_add_constraints = 'lower' if bounds_flipped else 'upper'
            elif (not bounds_flipped and add_to_add_constraints == 'lower') or (bounds_flipped and add_to_add_constraints == 'upper'):
                add_to_add_constraints = None # just added it
            l = None
        else:
            if len(bs) <= 1:
                break
            large_coeff_bound = np.max(np.abs(bs))
    # add other parts of constraint (lower or upper bound) which are not trivial:
    
    if (not bounds_flipped and add_to_add_constraints == 'lower') or (bounds_flipped and add_to_add_constraints == 'upper'):
        add_constraints += [unnormalize_negated_constraint(a, x, l, None, x_negation)]
    elif (not bounds_flipped and add_to_add_constraints == 'upper') or (bounds_flipped and add_to_add_constraints == 'lower'):
        add_constraints += [unnormalize_negated_constraint(a, x, None, u, x_negation)]
    
    # print Info
    if verbose:
        if constraint_trivial:
            if len(fix_variable_tuples) == 0:
                print('Constraint {} is trivial and can be removed'.format(constraint.name))
            elif fix_variable_tuples:
                print('Constraint {} can be removed when some variables are fixed'.format(constraint.name))
        elif fix_variable_tuples:
            if add_constraints:
                print('Constraint {} can be simplified when some variables are fixed'.format(constraint.name))
            else:
                print('Some variables in constraint {} can be fixed'.format(constraint.name))
        elif add_constraints: 
                print('Constraint {} can be simplified'.format(constraint.name))
    return constraint_trivial, fix_variable_tuples, add_constraints