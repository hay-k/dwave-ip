# MIT License
#
# Copyright (c) 2020 Hayk Sargsyan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from dwaveip.integer_quadratic_model import IntegerQuadraticModel, VarType
from itertools import product


class IntegerLinearProgramming:
    """
    Solve integer linear programming problems with equality constraints using D-Wave.
    """
    def __init__(self, c, a, b, vartypes, iqm_params=None, oweight=None, cweight=None):
        """
        Initialize an ILP problem and encode into a IntegerQuadraticModel.
        It is assumed that the problem is given as follows: maximize c^Tx given that ax=b, where
        c and x are vectors of size n, a is an nxm matrix, b is a vector of size m. n is the number
        of variables, m is the number of equality constraints.

        Args:
            c numpy array: The coefficients in the minimization objective.
            a numpy array: The matrix of coefficients of the equality constraints.
            b numpy array: The RHS of the equality constraints.
            vartypes list: Entry at i'th index specifies the VarType of i'th variable x. TODO: allow single value also
            iqm_params dict: The parameters for the underlying IntegerQuadraticModel.
            oweight numeric: The weight of the objective function in the QUBO.
            cweight numeric: The weight of the part of the QUBO due to the constraints.
        """
        if not oweight or not cweight:
            # TODO: determine automatically (for inspiration see arXiv:1302.5843). For now set to 1 and 100.
            oweight = 1
            cweight = 100

        iqm = IntegerQuadraticModel(iqm_params)
        for i in range(len(c)):
            iqm.add_variable(f"x_{i}", -oweight * c[i], vartypes[i])

        aTa = a.T @ a
        aTb = a.T @ b
        bTa = b.T @ a

        for i in range(len(c)):
            iqm.add_variable(f"x_{i}", -cweight * (aTb + bTa)[i])
        for i,j in product(range(len(c)), repeat=2):
            iqm.add_interaction(f"x_{i}", f"x_{j}", cweight * aTa[i, j])

        iqm.add_offset(cweight * b.T @ b)
        self._iqm = iqm

    def sample(self, sampler, *args, **kwargs):
        """
        Sample the underlying IntegerQuadraticModel.

        Args:
            sampler: A D-Wave sampler.
            *args: Positional arguments to the sampler's sample() function.
            **kwargs: Keyword arguments to the sampler's sample() function.

        Returns:
            A dimod.SampleSet object containing the samples.
        """
        return self._iqm.sample(sampler, *args, **kwargs)
