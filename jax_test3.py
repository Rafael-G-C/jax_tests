from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, jacfwd, jacrev


def E_calc(coordinates, charges):
    energy = 0.0
    natoms = charges.size
    for A in range(natoms):
        for B in range(A):
            d = jnp.linalg.norm(coordinates[A, :] - coordinates[B, :])
            energy += charges[A] * charges[B] / d
    return energy


def get_grad(coordinates, charges):
    g_E_calc = grad(E_calc)
    eval_E = g_E_calc(coordinates, charges)
    return eval_E


def get_hessian(coordinates, charges):
    gg_E_calc = jacfwd(jacrev(E_calc))
    eval_E = gg_E_calc(coordinates, charges)
    return eval_E


"""
coordinates = jnp.array([[2.0,3.0,2.0],[1.0,2.0,3.0]])
charges = jnp.array([1.0,1.0])

#E_list = E_calc(coordinates,charges)
#print(E_list) 

g_E_calc = grad(E_calc, 0)
test = g_E_calc(coordinates,charges)
print(test)
gg_E_calc = grad(g_E_calc,0)
testo = gg_E_calc(coordinates,charges)
print(testo)

#grad_E = get_hessian(coordinates,charges)
#print(grad_E)
"""
