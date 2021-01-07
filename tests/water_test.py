import jax_test3 as jt3
import pytest as pt

import water as w
import numpy as np

np.set_printoptions(linewidth=120)


@pt.mark.parametrize(
    "coordinates, charges, layer",
    [
        (
            np.array(
                [
                    [-0.10142170869456, -0.07509332372525, 0.0],
                    [-0.05369405529541, 1.75540922326286, 0.0],
                    [1.66333017194447, -0.56362368204899, 0.0],
                ]
            ),
            np.array([8.0, 1.0, 1.0]),
            0,
        ),
        (
            np.array(
                [
                    [-0.10142170869456, -0.07509332372525, 0.0],
                    [-0.05369405529541, 1.75540922326286, 0.0],
                    [1.66333017194447, -0.56362368204899, 0.0],
                ]
            ),
            np.array([8.0, 1.0, 1.0]),
            1,
        ),
        (
            np.array(
                [
                    [-0.10142170869456, -0.07509332372525, 0.0],
                    [-0.05369405529541, 1.75540922326286, 0.0],
                    [1.66333017194447, -0.56362368204899, 0.0],
                ]
            ),
            np.array([8.0, 1.0, 1.0]),
            2,
        ),
        (
            np.array(
                [
                    [-0.10142170869456, -0.07509332372525, 0.0],
                    [-0.05369405529541, 1.75540922326286, 0.0],
                    [1.66333017194447, -0.56362368204899, 0.0],
                ]
            ),
            np.array([8.0, 1.0, 1.0]),
            3,
        ),
    ],
)
def test_dervs(coordinates, charges, layer):
    ref_grad = w.test_dervs(coordinates, charges, layer)
    jax_grad = jt3.get_derv(coordinates, charges, layer)
    assert jax_grad == pt.approx(ref_grad, abs=1e-4)
