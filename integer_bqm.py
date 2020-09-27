from enum import IntEnum
from itertools import product
from collections import OrderedDict
import numpy as np

import dimod
from dimod.sampleset import SampleSet


class VarType(IntEnum):
    BINARY = 0
    UINT = 1
    INT = 2


class IntegerBQM:

    def __init__(self):
        self._bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
        self._vartype_map = {}

        self.uint_precision = 15
        self.int_precision = 16

    @property
    def uint_precision(self):
        return self._uint_precision

    @uint_precision.setter
    def uint_precision(self, value):
        if len(self._vartype_map):
            raise ValueError("Changing UINT precision on non-empty BQM not allowed")
        if not isinstance(value, int) or value <= 0:
            raise ValueError("UINT precision must be a positive integer")
        self._uint_precision = value

    @property
    def int_precision(self):
        return self._int_precision

    @int_precision.setter
    def int_precision(self, value):
        if len(self._vartype_map):
            raise ValueError("Changing int precision on non-empty BQM not allowed")
        if not isinstance(value, int) or value <= 0:
            raise ValueError("int precision must be a positive integer")
        self._int_precision = value

    def _binary_coefficients(self, vartype):
        if vartype == VarType.BINARY:
            return [1]
        elif vartype == VarType.UINT:
            return [2 ** i for i in range(self.uint_precision)]
        elif vartype == VarType.INT:
            return [2 ** i for i in range(self.int_precision - 1)] + [-2 ** (self.int_precision - 1)]
        else:
            raise ValueError(f"Unknown variable type {vartype}")

    def add_variable(self, v, bias, vartype=None):
        """
        Add a variable and/or its bias to the underlying binary quadratic model.
        For each variable, if called the first time, vartype must be specified.

        Args:
            v hashable: The label of the variable
            bias numeric: The associated bias
            vartype VarType: The type of the variable

        Returns:
            The list of corresponding binary variable labels added to the underlying binary quadratic model.
        """
        if self._vartype_map.get(vartype, None):
            if vartype != self._vartype_map.get(vartype, None):
                raise ValueError(f"vartype {vartype} does not match with previously specified one")
        else:
            if vartype is None:
                raise ValueError("Variable not defined previously. The vartype argument must be non-None")

        bc = self._binary_coefficients(vartype)
        self._vartype_map[v] = vartype
        return [self._bqm.add_variable((v, i), bias * c) for i, c in enumerate(bc)]

    def add_interaction(self, u, v, bias):
        if u not in self._vartype_map.keys():
            raise ValueError(f"Variable {u} not defined. Use the add_variable method to define it first.")
        if v not in self._vartype_map.keys():
            raise ValueError(f"Variable {v} not defined. Use the add_variable method to define it first.")

        u_vartype = self._vartype_map[u]
        v_vartype = self._vartype_map[v]

        u_bc = self._binary_coefficients(u_vartype)
        v_bc = self._binary_coefficients(v_vartype)

        for i,j in product(range(len(u_bc)), range(len(v_bc))):
            if (i == j) and (u == v):
                self._bqm.add_variable((u, i), u_bc[i] ** 2)
            else:
                self._bqm.add_interaction((u, i), (v, j), bias * u_bc[i] * v_bc[j])

    def sample(self, sampler, *args, **kwargs):
        sampleset = sampler.sample(self._bqm, *args, **kwargs)

        record = sampleset.record
        variables = sampleset.variables
        info = sampleset.info
        vartype = sampleset.vartype

        reconstructed_samples = np.zeros((record.shape[0], len(self._vartype_map)))
        original_variables = OrderedDict()
        for i, var in enumerate(variables):
            varname = var[0]
            if varname not in original_variables:
                original_variables[varname] = len(original_variables)
            ind = original_variables[varname]
            bc = self._binary_coefficients(self._vartype_map[varname])
            reconstructed_samples[:, ind] += record['sample'][:, i] * bc[var[1]]

        type = (np.record, [('sample', 'i', (len(original_variables),)), ('energy', '<f8'), ('num_occurrences', '<i4')])
        new_record = np.recarray(record.shape, type, names=('sample', 'energy', 'num_occurrences'))
        new_record['sample'] = reconstructed_samples
        new_record['energy'] = record['energy']
        new_record['num_occurrences'] = record['num_occurrences']

        return SampleSet(new_record, original_variables.keys(), info, vartype)
