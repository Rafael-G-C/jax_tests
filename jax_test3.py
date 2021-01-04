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


def get_grad(coordinates, charges):
    gradient = []
    natoms = len(charges)
    state = np.zeros(
        2 + 3 * natoms, dtype=int
    )  # makes an array that derv can work with
    flat_coords = coordinates.reshape(3 * natoms)
    for item in range(2, len(state)):
        state[item] = 1  # change the array to the wanted derivation for the hessian
        eval_E = derv(
            E_NN,
            [charges.size, charges, *(coordinates.reshape(3 * natoms).tolist())],
            state,
        )
        gradient.append(eval_E)
        state[item] = 0  # reset array

    gradient = np.array(gradient)
    gradient = gradient.reshape((natoms, 3))
    return gradient


# this will generate the full hessian
def get_hessian(coordinates, charges):
    natoms = len(charges)
    state = np.zeros(2 + 3 * natoms, dtype=int)
    hessian = np.zeros(
        (natoms, 3) * 2
    )  # reshaping the hessian gave errors so here's created a priori
    # these will keep track whether we are on the x y or z coordinate
    i_xyz = 0
    a_xyz = 0
    for i in range(2, len(state)):
        state[i] += 1  # change the item state
        for a in range(2, len(state)):
            state[a] += 1
            eval_E = derv(
                E_NN,
                [charges.size, charges, *(coordinates.reshape(3 * natoms).tolist())],
                state,
            )
            # print(f"mapping to {(i-2)//3},{i_xyz},{(a-2)//3},{a_xyz}") METADATA
            hessian[(i - 2) // 3, i_xyz, (a - 2) // 3, a_xyz] = eval_E
            state[a] -= 1  # reset
            a_xyz += 1  # change x y z
            a_xyz *= a_xyz != 3  # reset if more than 2 (z)

        state[i] -= 1
        i_xyz += 1
        i_xyz *= i_xyz != 3

    return hessian


"""
coordinates = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
charges = np.array([1.0, 1.0])

grad_E = get_hessian(coordinates,charges)
print(grad_E)
"""
