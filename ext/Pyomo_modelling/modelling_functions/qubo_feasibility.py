"""
Fraunhofer CML - qubo feasibility functions
==============================

Version  : 0.1.0

Description
-----------
    This module provides the class `FeasibilityFilter` to evaluate the feasibility and objective of QUBO solutions with respect to a constrained Pyomo model that the QUBO represents.
    In particular the method 'get_filter' generates a boolean function that determines whether all constraints of the model are satisfied by the binary QUBO solution.
    It does this by converting the pyomo model into a block form where the constraints are represented by Ax <= b, where A is an array, x the binary solution vector and b a vector.
    The rows of A abd b correspond to different constraints.
    The actual fesibility check is conducted by matrix operations, checking if Ax <= b is satisfied for a solution vector x.
    Other varitions of the booolean filter function are returned by the methods 'get_dimod_filter' and 'get_verbose_filter'.
    The filter provided by 'get_verbose_filter' returns a list of names of unsatisfied constraints next to the boolean value.
    The filter provided by 'get_dimod_filter' takes a slightly different input adapted to dimod Samplesets.
    With this, it is possible to restrict a sampleset to its feasible samples by the method sampleset.filter(filter), where filter is the output of 'get_dimod_filter'.

    The method 'get_true_objective function' generates a function that computes the objective value of a QUBO solution based on the Pyomo model.
    Hence, if the QUBO objective is merely an approximation of the actual objective, for example when slack_precision > 0 is used for generating the QUBO, then this function is recommended to compute the actual true objective.

Classes:
    FeasibilityFilter: generates tools for QUBO postprocessing.

Usage Example:
    f = FeasibilityFilter(
        BLP
    )
    filter = f.get_filter(variables) # variable names of the variables appealing in the QUBO_solution
    true_objective_function = f.get_true_objective_function(variables)
    solution_feasible = filter(QUBO_solution)
    true_objective = true_objective_function(QUBO_solution)
"""

from typing import Any, Callable, Optional, Tuple, Union, List

import numpy as np
import pyomo.environ as pyo
from pyomo.core.base.constraint import IndexedConstraint
from pyomo.repn.standard_repn import generate_standard_repn


class FeasibilityFilter:
    """Class for QUBO-postprocessing

    Attributes:
        var_names (List[str]): list of names of variables of the input pyomo model that are considered in the QUBO solutions
        A (np.ndarray): block form representation of constraint coefficients
        b (np.ndarray): 1D-block form representation of constraint bounds
        constr_per_index (np.ndarray): 1D-array of names of constraints that the corresponding row of b and A represents. Suffixes 'upper' and 'lower' indicate the bound type.
        c (np.ndarray): 1D-block form representation of objective function coefficients
        constant (float): additive constant of objective function
    """

    var_names: List[str]
    A: np.ndarray
    b: np.ndarray
    constr_per_index: np.ndarray
    c: np.ndarray
    constant: float

    def __init__(self, pyo_model: pyo.ConcreteModel) -> None:
        """Compute block form representation of a pyomo model.

        Args:
            pyo_model (pyo.ConcreteModel): Pyomo model in BLP-form that is represented by a QUBO whose solutions we want to postprocess.
        """
        self.var_names = self.get_vars(pyo_model)
        self.A, self.b, self.constr_per_index = self.get_constraint_rep(
            pyo_model=pyo_model, var_names=self.var_names
        )
        self.c, self.constant = self.get_objective_rep(
            pyo_model=pyo_model, var_names=self.var_names
        )

    @staticmethod
    def get_vars(pyo_model: pyo.ConcreteModel) -> List[str]:
        """
        Args:
            pyo_model (pyo.ConcreteModel): Pyomo model in BLP-form that is represented by QUBO whose solutions we want to postprocess.

        Returns:
            var_names (List[str]): list of names of variables of the input pyomo model that are considered in the QUBO solutions
        """
        var_names = []
        for var in pyo_model.component_data_objects(pyo.Var, active=True):
            if not var.is_fixed():
                var_names.append(var.name)
        return var_names

    @classmethod
    def get_constraint_rep(
        cls, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns block-representation A, b of constraints Ax <= b.

        :param var_names: List[str]
        :param constr_list: List[pyo.Constraint]
        :param pyo_model: pyo.ConcreteModel
        """
        if "var_names" in kwargs:
            var_names = kwargs["var_names"]
        else:
            assert "pyo_model" in kwargs
            var_names = cls.get_vars(kwargs["pyo_model"])

        if "constr_list" in kwargs:
            constr_list = []
            for constr in kwargs["constr_list"]:
                if type(constr) == IndexedConstraint:
                    for i in constr:
                        constr_list.append(constr[i])
                else:
                    constr_list.append(constr)
        else:
            assert "pyo_model" in kwargs
            constr_list = kwargs["pyo_model"].component_data_objects(pyo.Constraint)

        A = np.empty(shape=(0, len(var_names)))
        b = np.empty(shape=(0))
        constr_per_index = np.empty(shape=(0))

        for constr in constr_list:
            repn = generate_standard_repn(constr.body)

            indi = [var_names.index(var.name) for var in repn.linear_vars]
            linear_coeffs = np.array(repn.linear_coefs)
            ub = 0

            if constr.has_ub() and constr.has_lb():
                ub = pyo.value(constr.upper) - repn.constant

                b = np.concatenate((b, [ub]))
                constr_per_index = np.concatenate(
                    (constr_per_index, [constr.name + "_upper"])
                )
                row = np.zeros(shape=(len(var_names)))
                for i, c in zip(indi, linear_coeffs):
                    row[i] = c
                A = np.vstack((A, row))

                ub = -pyo.value(constr.lower) + repn.constant
                linear_coeffs = -linear_coeffs

                b = np.concatenate((b, [ub]))
                constr_per_index = np.concatenate(
                    (constr_per_index, [constr.name + "_lower"])
                )
                row = np.zeros(shape=(len(var_names)))
                for i, c in zip(indi, linear_coeffs):
                    row[i] = c
                A = np.vstack((A, row))

                continue

            elif constr.has_ub():
                ub = pyo.value(constr.upper) - repn.constant

            elif constr.has_lb():
                ub = -pyo.value(constr.lower) + repn.constant
                linear_coeffs = -linear_coeffs

            else:
                raise

            b = np.concatenate((b, [ub]))
            constr_per_index = np.concatenate((constr_per_index, [constr.name + "_upper"]))
            row = np.zeros(shape=(len(var_names)))
            for i, c in zip(indi, linear_coeffs):
                row[i] = c

            A = np.vstack((A, row))

            # Alternative: use sympy
            # bimap, sympy_obj = pyo.expr.sympy_tools.sympyify_expression(constr.body)
            # sympy_poly, _ = sympy.polys.polyutils.dict_from_expr(sympy_obj)

        return A, b, constr_per_index

    @classmethod
    def get_objective_rep(cls, **kwargs: Any) -> Tuple[np.ndarray, float]:
        """
        Returns block-representation c, constant of objective cx + constant

        :param var_names: List[String]
        :param objective: pyo.Objective
        :param pyo_model: pyo.ConcreteModel
        """
        if "var_names" in kwargs:
            var_names = kwargs["var_names"]
        else:
            assert "pyo_model" in kwargs
            var_names = cls.get_vars(kwargs["pyo_model"])

        if "objective" in kwargs:
            pass
        else:
            assert "pyo_model" in kwargs
            objectives = list(
                kwargs["pyo_model"].component_objects(pyo.Objective, active=True)
            )
            if objectives:
                objective = objectives[0]
                expr = objective.expr
            else:  # we may have no objective defined
                expr = 0

        repn = generate_standard_repn(expr)

        constant = repn.constant
        c = np.zeros(shape=(len(var_names)))

        indi = [var_names.index(var.name) for var in repn.linear_vars]
        linear_coeffs = np.array(repn.linear_coefs)

        for i, coeff in zip(indi, linear_coeffs):
            c[i] = coeff

        return c, constant
    

    def get_filter(
        self, variables: Optional[List[str]] = None
    ) -> Callable[[Union[List[int], np.ndarray]], bool]:
        """
        Returns a boolean function that determines whether QUBO solutions are feasible.

        Args:
            variables (Optional[List[str]]): list of names that the variables in the QUBO solution.
                Default is None, which means that self.var_names is taken instead.

        Returns:
            filter (Callable[List[int], bool]): boolean feasibility function
        """
        A = self._reshuffle_columns(self.A, variables)

        def filter(sample: Union[List[int], np.ndarray]) -> bool:
            return np.all(A @ sample <= self.b)

        return filter

    def get_dimod_filter(
        self, variables: Optional[List[str]] = None
    ) -> Callable[Any, bool]:
        """
        Returns a boolean function that determines whether samples of a dimod-sampleset represent feasible solutions.
        With this, it is possible to restrict a sampleset to its feasible samples by the method sampleset.filter(filter).

        Args:
            variables (Optional[List[str]]): list of names that the variables in the QUBO solution.
                Default is None, which means that self.var_names is taken instead.

        Returns:
            filter (Callable[List[int], bool]): boolean feasibility function
        """
        filter = self.get_filter(variables)
        return lambda sample: filter(sample.sample)

    def get_verbose_filter(
        self, variables: Optional[List[str]] = None
    ) -> Callable[[Union[List[int], np.ndarray]], Tuple[bool, List[str]]]:
        A = self._reshuffle_columns(self.A, variables)

        def verbose_filter(
            sample: Union[List[int], np.ndarray],
        ) -> Tuple[bool, List[str]]:
            feasible = A @ sample <= self.b
            infeasibilities = np.where(np.logical_not(feasible))[0]
            return len(infeasibilities) == 0, self.constr_per_index[infeasibilities]

        return verbose_filter

    def get_true_objective_function(
        self, variables: Optional[List[str]] = None
    ) -> Callable[[Union[List[int], np.ndarray]], float]:
        """
        Returns a function that computes the true objective of a QUBO-solution.

        Args:
            variables (Optional[List[str]]): list of names that the variables in the QUBO solution.
                Default is None, which means that self.var_names is taken instead.

        Returns:
            true_objective (Callable[List[int], float]): true objective function
        """
        c = self._reshuffle_columns(self.c, variables)

        def true_objective(sample: Union[List[int], np.ndarray]) -> float:
            true_obj = c @ sample + self.constant
            return true_obj

        return true_objective
    
    ## Helpers:
    def _reshuffle_columns(self, arr: np.ndarray, variables: Optional[List[str]]) -> np.ndarray:
        """Reshuffle array so its last axis corresponds to the entries of a variable vector."""
        if variables is None:
            return arr

        # Determine which requested variables exist in the model
        sample_indices = [
            self.var_names.index(var) if var in self.var_names else None
            for var in variables
        ]

        # Check that all model variables NOT requested have no influence on the array (e.g. for feasibility)
        used_model_indices = [i for i in sample_indices if i is not None]
        remaining_indices = list(set(range(len(self.var_names))) - set(used_model_indices))
        if remaining_indices:
            arr_remaining = np.take(arr, remaining_indices, axis=-1)
            # Reduce over all axes except the last to find non-zero columns
            other_axes = tuple(range(arr_remaining.ndim - 1))
            nonzero_cols = np.where(np.any(arr_remaining != 0, axis=other_axes))[0]
            assert nonzero_cols.size == 0, (
                "Please include all variables that are relevant. Missing: {}".format(
                    [self.var_names[remaining_indices[c]] for c in nonzero_cols]
                )
            )

        # Build result: real slice if variable is in the model, else zeros
        zero_slice = np.zeros(arr.shape[:-1])  # shape of one "column" (scalar for 1D)
        columns = [
            np.take(arr, idx, axis=-1) if idx is not None else zero_slice
            for idx in sample_indices
        ]
        return np.stack(columns, axis=-1)
