import numpy as np


def gradient(x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b):
    """Computes the gradient."""

    cxc = c_a * c_b
    xyz_squared = ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2) ** (3 / 2)

    d_x = -(x_a - x_b) * cxc / xyz_squared
    d_y = -(y_a - y_b) * cxc / xyz_squared
    d_z = -(z_a - z_b) * cxc / xyz_squared

    return np.array([d_x, d_y, d_z])


def hessian(x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b):
    """Computes the Hessian."""

    cxc = c_a * c_b
    xyz_squared = (x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2

    d_xx = (
        3 * cxc * (x_a - x_b) ** 2 / xyz_squared ** (5 / 2)
    ) - cxc / xyz_squared ** (3 / 2)
    d_xy = (3 * cxc * (x_a - x_b) * (y_a - y_b)) / xyz_squared ** (5 / 2)
    d_xz = (3 * cxc * (x_a - x_b) * (z_a - z_b)) / xyz_squared ** (5 / 2)
    d_yy = (
        3 * cxc * (y_a - y_b) ** 2 / xyz_squared ** (5 / 2)
    ) - cxc / xyz_squared ** (3 / 2)
    d_yz = (3 * cxc * (y_a - y_b) * (z_a - z_b)) / xyz_squared ** (5 / 2)
    d_zz = (
        3 * cxc * (z_a - z_b) ** 2 / xyz_squared ** (5 / 2)
    ) - cxc / xyz_squared ** (3 / 2)

    return np.array([d_xx, d_xy, d_xz, d_yy, d_yz, d_zz])
