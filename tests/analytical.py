import numpy as np


def gradient(coordinates,charges):
    """Computes the gradient."""
    number_coordinates = len(coordinates)
    eval_E = []
    for atom_a in range(number_coordinates):
        for atom_b in range(number_coordinates):
            if atom_a < atom_b:
                cxc = charges[atom_a] * charges[atom_b]
                xyz_squared = ((coordinates[atom_a,0] - coordinates[atom_b,0]) ** 2 + (coordinates[atom_a,1] - coordinates[atom_b,1]) ** 2 + (coordinates[atom_a,2] - coordinates[atom_b,2]) ** 2) ** (3 / 2)

                d_ax = -(coordinates[atom_a,0] - coordinates[atom_b,0]) * cxc / xyz_squared
                eval_E.append(d_ax)
                d_ay = -(coordinates[atom_a,1] - coordinates[atom_b,1]) * cxc / xyz_squared
                eval_E.append(d_ay)
                d_az = -(coordinates[atom_a,2] - coordinates[atom_b,2]) * cxc / xyz_squared
                eval_E.append(d_az)

                d_bx = -d_ax
                eval_E.append(d_bx)
                d_by = -d_ay
                eval_E.append(d_by)
                d_bz = -d_az
                eval_E.append(d_bz)

    return np.array(eval_E)


def hessian(coordinates,charges):
    """Computes the Hessian."""
    number_coordinates = len(coordinates)
    eval_E = []
    for atom_a in range(number_coordinates):
        for atom_b in range(number_coordinates):
            if atom_a < atom_b:
                cxc = charges[atom_a] * charges[atom_b]
                x_a = coordinates[atom_a,0]
                y_a = coordinates[atom_a,1]
                z_a = coordinates[atom_a,2]
                x_b = coordinates[atom_b,0]
                y_b = coordinates[atom_b,1]
                z_b = coordinates[atom_b,2]

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

                eval_E.append(d_axax)
                eval_E.append(d_axay)
                eval_E.append(d_axaz)
                eval_E.append(d_axbx)
                eval_E.append(d_axby)
                eval_E.append(d_axbz)
                eval_E.append(d_ayay)
                eval_E.append(d_ayaz)
                eval_E.append(d_aybx)
                eval_E.append(d_ayby)
                eval_E.append(d_aybz)
                eval_E.append(d_azaz)
                eval_E.append(d_azbx)
                eval_E.append(d_azby)
                eval_E.append(d_azbz)
                eval_E.append(d_bxbx)
                eval_E.append(d_bxby)
                eval_E.append(d_bxbz)
                eval_E.append(d_byby)
                eval_E.append(d_bybz)
                eval_E.append(d_bzbz)

    return np.array(eval_E)
a_gradient = hessian(np.array([[1.0,2.0,3.0],[2.0,1.0,3.0],[3.0,1.0,2.0]]),[1.0,1.0,1.0])
print(a_gradient)