import jax_test3 as jt3
import pytest as pt

import analytical
import numpy as np

np.set_printoptions(linewidth=120)


@pt.mark.parametrize(
    "coordinates, charges, layer",
    [
        (np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]), np.array([1.0, 2.0]), 0),
        (
            np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 2.0]]),
            np.array([2.0, 1.0, 3.0]),
            0,
        ),
        (
            np.array(
                [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 2.0], [5.0, 4.0, 3.0]]
            ),
            np.array([3.0, 4.0, 1.0, 7.0]),
            0,
        ),
        (
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 1.0, 3.0],
                    [3.0, 1.0, 2.0],
                    [5.0, 4.0, 3.0],
                    [4.0, 3.0, 5.0],
                ]
            ),
            np.array([7.0, 6.0, 1.0, 5.0, 5.0]),
            0,
        ),
    ],
)
def test_grad(coordinates, charges, layer):
    ref_grad = analytical.gradient(coordinates, charges)
    jax_grad = jt3.get_derv(coordinates, charges, layer)
    assert jax_grad == pt.approx(ref_grad)


@pt.mark.parametrize(
    "coordinates, charges, layer",
    [
        (np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]), np.array([1.0, 2.0]), 1),
        (
            np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 2.0]]),
            np.array([2.0, 1.0, 3.0]),
            1,
        ),
        (
            np.array(
                [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 2.0], [5.0, 4.0, 3.0]]
            ),
            np.array([3.0, 4.0, 1.0, 7.0]),
            1,
        ),
        (
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 1.0, 3.0],
                    [3.0, 1.0, 2.0],
                    [5.0, 4.0, 3.0],
                    [4.0, 3.0, 5.0],
                ]
            ),
            np.array([7.0, 6.0, 1.0, 5.0, 5.0]),
            1,
        ),
    ],
)
def test_hessian(coordinates, charges, layer):
    ref_hess = analytical.hessian_redundant(coordinates, charges)
    print(f"ref_hess\n{ref_hess}")
    jax_hess = jt3.get_derv(coordinates, charges, layer)
    print(f"jax_hess\n{jax_hess}")
    assert jax_hess == pt.approx(ref_hess)
