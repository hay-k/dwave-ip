# dwave-ip
A wrapper package over D-Wave Ocean providing functionality to encode and sample integer variables.

D-Wave currently supports three types of variables (SPIN, BINARY and DISCRETE) and they [seem to have plans](https://github.com/dwavesystems/dimod/issues/664) to support integer variables at the fundamental level as well, however the functionality is not available yet. dwave-ip implements a simple wrapper over D-Wave's API to allow easy and straightforward handling of integer variables.

The basic idea is to have a `dimod.BinaryQuadraticModel` object under the hood and use binary representation for each integer variable to encode them as a collection of binary variables. dwave-ip introduces the class `IntegerQuadraticModel`, which is very similar to D-Wave's `dimod.BinaryQuadraticModel`, however allows encoding of integer variables. dwave-ip defines its own variable type, namely BINARY, UINT (unsigned int) and INT (signed int). In addition to a typical functionality provided by `dimod.BinaryQuadraticModel`, `IntegerQuadraticModel` also provides a method `sample()`, which takes care of reconstructing the integers before returning the sampleset. More detailed documentation can be found in the docstrings.

## Examples
Examples on how to use the package are provided in [examples.ipynb](examples.ipynb)

## Installation
This package can be installed using `pip` as follows

`pip install git+https://github.com/hay-k/dwave-ip`
