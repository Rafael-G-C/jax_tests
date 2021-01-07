import os
import os.path as osp
import numpy as np


def test_dervs(coordinates, charges, order):
    files = ["g_h2o.txt", "gg_h2o.txt", "ggg_h2o.txt", "gggg_h2o.txt"]
    natoms = len(charges)
    a = np.loadtxt("tests/" + files[order - 1])
    a = a.reshape((natoms, 3) * order)
    return a


"""
coordinates = 0
charges = [0,0,0]
layer = 3
a = test_dervs(coordinates,charges,layer)
print(a)
"""
