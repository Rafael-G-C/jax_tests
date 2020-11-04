import numpy as np


def gradient(x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b):
    """Computes the gradient."""

    cxc = c_a * c_b
    xyz_squared = ((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2) ** (3 / 2)

    d_ax = -(x_a - x_b) * cxc / xyz_squared
    d_ay = -(y_a - y_b) * cxc / xyz_squared
    d_az = -(z_a - z_b) * cxc / xyz_squared

    d_bx = -d_ax
    d_by = -d_ay
    d_bz = -d_az
    return np.array([d_ax, d_ay, d_az, d_bx, d_by, d_bz])


def hessian(x_a, y_a, z_a, x_b, y_b, z_b, c_a, c_b):
    """Computes the Hessian."""

    cxc = c_a * c_b
    xyz_squared = (x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2

    d_axax = (
        3 * cxc * (x_a - x_b) ** 2 / xyz_squared ** (5 / 2)
    ) - cxc / xyz_squared ** (3 / 2)

    d_axbx = cxc / xyz_squared ** (3 / 2) - 3 * cxc * (
        x_a - x_b
    ) ** 2 / xyz_squared ** (5 / 2)

    d_axay = (3 * cxc * (x_a - x_b) * (y_a - y_b)) / xyz_squared ** (5 / 2)
    d_axby = -d_axay

    d_axaz = (3 * cxc * (x_a - x_b) * (z_a - z_b)) / xyz_squared ** (5 / 2)
    d_axbz = -d_axaz

    d_ayay = (
        3 * cxc * (y_a - y_b) ** 2 / xyz_squared ** (5 / 2)
    ) - cxc / xyz_squared ** (3 / 2)
    d_aybx = -d_axay
    d_ayby = cxc / xyz_squared ** (3 / 2) - (
        3 * cxc * (y_a - y_b) ** 2 / xyz_squared ** (5 / 2)
    )

    d_ayaz = (3 * cxc * (y_a - y_b) * (z_a - z_b)) / xyz_squared ** (5 / 2)
    d_aybz = -d_ayaz
    d_azaz = (
        3 * cxc * (z_a - z_b) ** 2 / xyz_squared ** (5 / 2)
    ) - cxc / xyz_squared ** (3 / 2)
    d_azbx = -d_axaz
    d_azby = -d_ayaz
    d_azbz = cxc / xyz_squared ** (3 / 2) - (
        3 * cxc * (z_a - z_b) ** 2 / xyz_squared ** (5 / 2)
    )
    d_bxbx = d_axax
    d_bxby = d_axay
    d_bxbz = d_axaz
    d_byby = d_ayay
    d_bybz = d_ayaz
    d_bzbz = d_azaz
    return np.array(
        [
            d_axax,
            d_axay,
            d_axaz,
            d_axbx,
            d_axby,
            d_axbz,
            d_ayay,
            d_ayaz,
            d_aybx,
            d_ayby,
            d_aybz,
            d_azaz,
            d_azbx,
            d_azby,
            d_azbz,
            d_bxbx,
            d_bxby,
            d_bxbz,
            d_byby,
            d_bybz,
            d_bzbz,
        ]
    )
