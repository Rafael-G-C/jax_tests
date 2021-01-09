import numpy as np


def reference_h2o_derivative(order):
    natoms = 3
    a = np.loadtxt(f"tests/{'g' * order}_h2o.txt").reshape((natoms, 3) * order)
    return a


"""
coordinates = 0
charges = [0,0,0]
layer = 3
a = test_dervs(coordinates,charges,layer)
print(a)
"""
