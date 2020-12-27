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

def some_data(charges):
    natoms = len(charges)
    state = np.zeros(2+3*natoms, dtype=int)
    return natoms, state

def E_NN(N, Zs, *Rs):  
    if len(Rs) != N * 3:
        raise RuntimeError(f"Must have {3*N} coordinates, got {len(Rs)}") 

    if len(Zs) != N:
        raise RuntimeError(f"Must have {N} charges, got {len(Zs)}")
    

    e_nn = 0
    for A in range(N):
        for B in range(A):
            d_AB = jnp.sqrt((Rs[3*A] - Rs[3*B])**2 + (Rs[3*A+1] - Rs[3*B+1])**2 + (Rs[3*A+2] - Rs[3*B+2])**2)
            e_nn += Zs[A] * Zs[B] / d_AB
    
    return e_nn


def get_grad(coordinates,charges):
    gradient = []
    natoms, state = some_data(charges)
    for item in range(2,len(state)):
        state[item] = 1
        eval_E = derv(E_NN, [charges.size, charges, *(coordinates.reshape(3*natoms).tolist())], state)
        gradient.append(eval_E)
        state[item] = 0

    gradient = np.array(gradient)
    gradient = gradient.reshape((natoms,3))
    return gradient


def get_hessian(coordinates,charges):
    natoms, state = some_data(charges)
    hessian = np.zeros((natoms, 3) * 2)
    print(hessian.shape)
    i_xyz = 0
    a_xyz = 0
    for i in range(2,len(state)):
        state[i] = 1
        for a in range(2,len(state)):
            state[a] = 1     
            eval_E = derv(E_NN, [charges.size, charges, *(coordinates.reshape(3*natoms).tolist())], state)
            print(f"mapping to {(i-2)//3},{i_xyz},{(a-2)//3},{a_xyz}")
            hessian[(i-2)//3,i_xyz,(a-2)//3,a_xyz] = eval_E
            state[a] = 0
            a_xyz += 1
            a_xyz *= a_xyz != 3
                
        state[i] = 0
        i_xyz += 1
        i_xyz *= i_xyz != 3 

    return hessian

coordinates = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
charges = np.array([1.0, 1.0])
"""
#E_list = E_calc(coordinates,charges)
#print(E_list) 

g_E_calc = grad(E_calc, 0)
test = g_E_calc(coordinates,charges)
print(test)
gg_E_calc = grad(g_E_calc,0)
testo = gg_E_calc(coordinates,charges)
print(testo)
"""
grad_E = get_hessian(coordinates,charges)
print(grad_E)

