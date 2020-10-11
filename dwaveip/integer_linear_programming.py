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
    Solve integer linear programming problems with equality constraints using D'Wave.
    """
    def __init__(self, c, a, b, vartypes):
        """
        Initialize an ILP problem and encode into a IntegerQuadraticModel.
        It is assumed that the problem is given as follows: minimize c^Tx given that ax=b, where
        c and x are vectors of size n, a is an nxm matrix, b is a vector of size m. n is the number
        of variables, m is the number of equality constraints.

        Args:
            c numpy array: The coefficients in the minimization objective.
            a numpy array: The matrix of coefficients of the equality constraints.
            b numpy array: The RHS of the equality constraints.
            vartypes list: Entry at i'th index specifies the VarType of i'th variable x
        """
        iqm = IntegerQuadraticModel()
        for i in range(len(c)):
            iqm.add_variable(f"x_{i}", c[i], vartypes[i])

        iqm.add_offset(b.T @ b)

        aTa = a.T @ a
        aTb = a.T @ b
        bTa = b.T @ a

        for i in range(len(c)):
            iqm.add_variable(f"x_{i}", -(aTb + bTa)[i])
        for i,j in product(range(len(c)), repeat=2):
            iqm.add_interaction(f"x_{i}", f"x_{j}", aTa[i, j])

        self._iqm = iqm

    def sample(self, sampler, *args, **kwargs):
        """
        Sample the underlying IntegerQuadraticModel.

        Args:
            sampler: A D'Wave sampelr.
            *args: Positional arguments to the sampler's sample() function.
            **kwargs: Keyword arguments to the sampler's sample() function.

        Returns:
            A dimod.SampleSet object containing the samples.
        """
        return self._iqm.sample(sampler, *args, **kwargs)
