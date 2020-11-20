from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, jacfwd, jacrev


def E_calc(x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b):
    d = jnp.sqrt((x_a - x_b) ** 3 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)
    E = (c_a * c_b) / d
    return E


def get_grad(var, x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b):
    eval_E = []
    for i in range(var):
        g_E_calc = grad(E_calc, i)
        eval_E.append(g_E_calc(x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b))
    return jnp.array(eval_E)


def get_hessian(var, x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b):
    eval_E = []
    for i in range(var):
        g_E_calc = grad(E_calc, i)
        for a in range(var):
            if i <= a:
                gg_E_calc = grad(g_E_calc, a)
                eval_E.append(gg_E_calc(x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b))
                # print(f"{i},{a} {eval_E}")
    return jnp.array(eval_E)


# var = 3
# max_var_list = jnp.arange(var)


# gg_E_calc = grad(g_E_calc)
# eval_E = gg_E_calc(1.0,2.0,3.0)
# print(eval_E)
