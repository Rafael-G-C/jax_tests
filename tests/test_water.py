import jax_test3 as jt3
import pytest as pt

import water as w
import numpy as np

np.set_printoptions(linewidth=120)


@pt.mark.parametrize(
    "order",
    [(1), (2), (3), (4)],
)
def test_dervs(order):
    coordinates = np.array(
        [
            [-0.10142170869456, -0.07509332372525, 0.0],
            [-0.05369405529541, 1.75540922326286, 0.0],
            [1.66333017194447, -0.56362368204899, 0.0],
        ]
    )
    charges = np.array([8.0, 1.0, 1.0])
    ref_grad = w.reference_h2o_derivative(order)
    jax_grad = jt3.E_NN_derivatives(coordinates, charges, order)
    assert jax_grad == pt.approx(ref_grad, rel=1e-5)
