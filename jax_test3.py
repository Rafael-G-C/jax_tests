from itertools import product
from jax.config import config
import numpy as np

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, jacfwd, jacrev


def _derv_sequence(orders):
    sequence = []
    for variable, variable_order in enumerate(orders):
        if variable_order > 0:
            sequence += variable_order * [variable]
    return sequence


def test_derv_sequence():
    assert _derv_sequence((3, 2, 1, 0)) == [0, 0, 0, 1, 1, 2]
    assert _derv_sequence((0, 1, 2, 3)) == [1, 2, 2, 3, 3, 3]
    assert _derv_sequence((0, 1, 0, 1)) == [1, 3]


def derv(fun, variables, orders) -> float:
    """
    fun: function to differentiate which expects a certain number of variables
    variables: list of variables at which to differentiate the function
    orders: [1, 0, 2, 0] means differentate with respect to variable 1 once,
                         and differentiate with respect to variable 3 twice.
    """
    sequence = _derv_sequence(orders)
    functions = [fun]
    for i, order in enumerate(sequence):
        functions.append(grad(functions[i], (order)))
    return functions[-1](*variables)


def E_NN(N, Zs, *Rs):
    if len(Rs) != N * 3:
        raise RuntimeError(f"Must have {3*N} coordinates, got {len(Rs)}")

    if len(Zs) != N:
        raise RuntimeError(f"Must have {N} charges, got {len(Zs)}")

    e_nn = 0
    for A in range(N):
        for B in range(A):
            d_AB = jnp.sqrt(
                (Rs[3 * A] - Rs[3 * B]) ** 2
                + (Rs[3 * A + 1] - Rs[3 * B + 1]) ** 2
                + (Rs[3 * A + 2] - Rs[3 * B + 2]) ** 2
            )
            e_nn += Zs[A] * Zs[B] / d_AB

    return e_nn


def distribute(indices, num_variables):
    l = [0 for _ in range(num_variables)]
    for index in indices:
        l[index] += 1
    return l


# this will generate the derivatives layer
def E_NN_derivatives(coordinates, charges, order):
    """
    1 --> gradient; 2 --> hessian; 3 --> 3rd derivatives; etc...
    """
    dervs = []
    natoms = len(charges)
    flat_coords = coordinates.reshape(3 * natoms)
    num_variables = 3 * natoms

    combo = product(range(num_variables), repeat=order)
    derivative_orders = map(lambda x: distribute(x, num_variables), combo)

    for d_order in derivative_orders:
        dervs.append(
            derv(E_NN, [charges.size, charges, *flat_coords], 2 * [0] + d_order)
        )

    dervs = np.array(dervs).reshape((natoms, 3) * order)

    return dervs


"""
coordinates = np.array(
    [[-0.10142170869456, -0.07509332372525,  0.              ],
     [-0.05369405529541,  1.75540922326286,  0.              ],
     [ 1.66333017194447, -0.56362368204899,  0.              ]]
    )

charges = np.array([8., 1., 1.])

grad_E = E_NN_derivatives(coordinates,charges,2)
print(grad_E)
"""
