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


def _derv_plus(current, stop, state, dervs, charges, flat_coords):
    if current != stop:
        current += 1
        for i in range(2, len(state)):
            state[i] += 1  # change the item state

            derv_plus(current, stop, state, dervs, charges, flat_coords)

            state[i] -= 1
    else:
        for i in range(2, len(state)):
            state[i] += 1  # change the item state

            eval_E = derv(
                E_NN,
                [charges.size, charges, *flat_coords],
                state,
            )
            dervs.append(eval_E)

            state[i] -= 1


# this will generate the derivatives layer
def get_derv(coordinates, charges, layer):
    """
    0 = gradient, 1 = hessian, 2 = 3rd derivatives, etc...
    """
    natoms = len(charges)
    state = np.zeros(2 + 3 * natoms, dtype=int)
    dervs = []
    flat_coords = coordinates.reshape(3 * natoms)

    derv_plus(0, layer, state, dervs, charges, flat_coords)

    dervs = np.array(dervs)
    dervs = dervs.reshape((natoms, 3) * (layer + 1))

    return dervs


"""
coordinates = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
charges = np.array([1.0, 1.0])

grad_E = get_dervs(coordinates,charges,0)
print(grad_E)
"""
