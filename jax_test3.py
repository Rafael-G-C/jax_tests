from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, jacfwd, jacrev


def E_calc(x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b):
    d = jnp.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    energy = (c_a * c_b) / d
    return energy


def get_grad(coordinates,charges):
    number_coordinates = len(coordinates)
    eval_E = []
    for atom_a in range(number_coordinates):
        for atom_b in range(number_coordinates):
            if atom_a < atom_b:
                for i in range(6): 
                    g_E_calc = grad(E_calc, i)
                    eval_E.append(g_E_calc(coordinates[atom_a,0],coordinates[atom_a,1],coordinates[atom_a,2],coordinates[atom_b,0],coordinates[atom_b,1],coordinates[atom_b,2],charges[atom_a],charges[atom_b]))
    return jnp.array(eval_E)


def get_hessian(coordinates,charges):
    eval_E = []
    number_coordinates = len(coordinates)
    for atom_a in range(number_coordinates):
        for atom_b in range(number_coordinates):
            if atom_a < atom_b:

                for i in range(6):
                    g_E_calc = grad(E_calc, i)
                    for a in range(6):
                        if i <= a:
                            gg_E_calc = grad(g_E_calc, a)
                            eval_E.append(gg_E_calc(coordinates[atom_a,0],coordinates[atom_a,1],coordinates[atom_a,2],coordinates[atom_b,0],coordinates[atom_b,1],coordinates[atom_b,2],charges[atom_a],charges[atom_b]))
                            # print(f"{i},{a} {eval_E}")
    return jnp.array(eval_E)


coordinates = jnp.array([[1.0,2.0,3.0],[2.0,3.0,2.0],[1.0,2.0,4.0]])
charges = [1.0,1.0,1.0]
#E_list = E_calc(coordinates,charges)
#print(E_list) 
#var = 3


#gg_E_calc = grad(E_calc, 0)
#testo = gg_E_calc(1.0,2.0,3.0,1.0,3.0,2.0,1.0,1.0)
#print(testo)

grad_E = get_hessian(jnp.array([[1.0,2.0,3.0],[2.0,1.0,3.0],[3.0,1.0,2.0]]),[1.0,1.0,1.0])
print(grad_E)
#print(eval_E)
