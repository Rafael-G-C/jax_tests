import jax_test3 as jt3
import pytest as pt

import analytical
import numpy as np

np.set_printoptions(linewidth=120)


@pt.mark.parametrize(
    "coordinates, charges, order",
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
def test_grad(coordinates, charges, order):
    ref_grad = analytical.gradient(coordinates, charges)
    jax_grad = jt3.E_NN_derivatives(coordinates, charges, order)
    assert jax_grad == pt.approx(ref_grad)


@pt.mark.parametrize(
    "coordinates, charges, order",
    [
        (np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]), np.array([1.0, 2.0]), 2),
        (
            np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 2.0]]),
            np.array([2.0, 1.0, 3.0]),
            2,
        ),
        (
            np.array(
                [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 2.0], [5.0, 4.0, 3.0]]
            ),
            np.array([3.0, 4.0, 1.0, 7.0]),
            2,
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
            2,
        ),
    ],
)
def test_hessian(coordinates, charges, order):
    ref_hess = analytical.hessian_redundant(coordinates, charges)
    print(f"ref_hess\n{ref_hess}")
    jax_hess = jt3.E_NN_derivatives(coordinates, charges, order)
    print(f"jax_hess\n{jax_hess}")
    assert jax_hess == pt.approx(ref_hess)
