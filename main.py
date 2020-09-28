from core.integer_bqm import IntegerBQM, VarType
import dimod


if __name__ == '__main__':
    ibqm = IntegerBQM()
    ibqm.uint_precision = 4
    ibqm.add_variable('x', -4, VarType.UINT)
    ibqm.add_variable('y', 6, VarType.INT)
    ibqm.add_interaction('x', 'x', 1)
    ibqm.add_interaction('y', 'y', 1)

    sampler = dimod.SimulatedAnnealingSampler()
    print(ibqm.sample(sampler, num_reads=10).record)

