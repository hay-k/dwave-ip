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


class IntegerQuadraticModel:
    """
    A wrapper class to enable encoding and sampling integer variables with Dwave.
    Uses a dimod.BinaryQuadraticModel under the hood, with dimod.BINARY variable type.
    IntegerQuadraticModel has its own definitions for variable types listed in the VarType enum.
    Three types of variables are supported: BINARY, UINT (unsigned integer) and INT (signed integer).
    To encode integers into a Dwave annealing device, their binary representation is used:
    for UINTs - ordinary binary expansion, and for INTs - two's complement expansion.
    """

    def __init__(self):
        """
        Initialize an empty IntegerQuadraticModel object.

        Member variables uint_precision and int_precision are set to 15 and 16 bits by default.
        They can be overridden, before any variable is added to the model.

        Example:
            >>> iqm = IntegerQuadraticModel()
            >>> iqm.uint_precision = 31
            >>> iqm.int_precision = 32
        """
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
            raise ValueError("Changing UINT precision on a non-empty model is not allowed")
        if not isinstance(value, int) or value <= 0:
            raise ValueError("UINT precision must be a positive integer")
        self._uint_precision = value

    @property
    def int_precision(self):
        return self._int_precision

    @int_precision.setter
    def int_precision(self, value):
        if len(self._vartype_map):
            raise ValueError("Changing int precision on a non-empty model is not allowed")
        if not isinstance(value, int) or value <= 0:
            raise ValueError("INT precision must be a positive integer")
        self._int_precision = value

    def _binary_coefficients(self, vartype):
        """
        Returns list of coefficients in the binary expansion based on the variable type.
        """
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
        Add a variable and/or its bias. If the variable is being added the first time,
        the vartype argument is mandatory. Otherwise, it is optional, however if specified
        it should be the same as when the variable was first added. Multiple calls to this
        function with the same variable label will add up the biases, just like the add_variable
        function in Dwave's BinaryQuadraticModel.

        This function will use the binary expansion corresponding to the variable type, to encode
        corresponding number of binary variables to the underlying binary quadratic model. Each
        binary variable associated with an integer will get a name, which is a tuple constructed with
        v and the index of the binary digit in the expansion. E.g. for a variable v with UINT type
        uint_precision binary variables will be added with names (v, 0), (v, 1), ..., (v, uint_precision - 1)

        Args:
            v hashable: The label of the variable.
            bias numeric: The associated bias.
            vartype VarType: The type of the variable.

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
        """
        Add an interaction term into the model. The variables u and v should both be added to the model
        with the add_variable function before this function is called. If any variable does not have a
        linear term and participates only in interactions, it still needs to be added with add_variable first
        with bias=0.0. Multiple runs of the function with the same variable names will add up the biases just
        like the add_interaction function in Dwave's BinaryQuadraticModel.

        NOTE: unlike the case of binary variables, for integers square terms (variable^2) are also possible in a
        quadratic model. To add such interactions, just use the same variable name for u and v.

        Args:
            u hashable: A variable label.
            v hashable: A variable label.
            bias numeric: The corresponding bias.
        """
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

    def add_offset(self, offset):
        """
        Add offset (constant shift in energy) to the quadratic model.

        Args:
            offset numeric: The offset to add.
        """
        self._bqm.add_offset(offset)

    def sample(self, sampler, *args, **kwargs):
        """
        Sample the integer quadratic model with the given sampler. This will sample the underlying
        binary quadratic model, reconstruct the integer variables from the sampleset, and return a
        new sampleset containing the reconstructed integer variables with their original names.

        NOTE: If you print the SampleSet object returned by this function, you will not see the correct
        integer values. This is because Dwave's printing functionality does not support integers, but
        SampleSet object does indeed contain integers, this is just an issue with printing. If you desparately
        need printing, then you can just print the record inside the SampleSet object
            >>> sampleset = iqm.sample(sampler, num_reads=10)
            >>> print(sampleset.record)

        Args:
            sampler: A Dwave sampler.
            *args: Positional arguments to the sampler's sample() function.
            **kwargs: Keyword arguments to the sampler's sample() function.

        Returns:
            A dimod.SampleSet object containing the samples.
        """
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
