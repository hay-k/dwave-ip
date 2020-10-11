import unittest
from dwaveip import IntegerQuadraticModel, VarType


class TestIntegerQuadraticModel(unittest.TestCase):

    def test_init(self):
        iqm = IntegerQuadraticModel()
        self.assertEqual(4, iqm.uint_precision)
        self.assertEqual(5, iqm.int_precision)

    def test_add_variable(self):
        iqm = IntegerQuadraticModel()
        iqm.uint_precision = 5
        iqm.int_precision = 6

        iqm.add_variable('x', 1, VarType.BINARY)
        self.assertEqual(1, iqm._bqm.num_variables)

        iqm.add_variable('y', 1, VarType.UINT)
        self.assertEqual(6, iqm._bqm.num_variables)

        iqm.add_variable('z', 1, VarType.INT)
        self.assertEqual(12, iqm._bqm.num_variables)

        self.assertRaises(ValueError, iqm.add_variable, 'x', 1, VarType.INT)
        self.assertRaises(ValueError, iqm.add_variable, 'a', 1)

        iqm.add_variable('x', 1)
        x = iqm.add_variable('x', 0.2, VarType.BINARY)
        self.assertEqual(2.2, iqm._bqm.get_linear(x[0]))


if __name__ == '__main__':
    unittest.main()
