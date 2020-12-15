import numpy as np


def gradient(coordinates, charges):
    """Computes the gradient."""
    g = np.zeros_like(coordinates)

    for A in range(charges.size):
        for B in range(charges.size):
            if A < B:
                cxc = charges[A] * charges[B]
                xyz_32 = np.linalg.norm(coordinates[A, :] - coordinates[B, :]) ** 3
                g[A, :] -= (coordinates[A, :] - coordinates[B, :]) * cxc / xyz_32
                g[B, :] -= g[A, :]

    return g


def hessian_redundant(coordinates, charges):
    """Computes the Hessian in redundant format

    Compute [3N*3N] second derivatives, rather than justs [3N*(3N+1)/2].
    """
    natoms = charges.size
    gg = np.zeros((natoms, 3) * 2)
    for A in range(charges.size):
        for B in range(charges.size):
            if A < B:
                cxc = charges[A] * charges[B]
                x_a = coordinates[A, 0]
                y_a = coordinates[A, 1]
                z_a = coordinates[A, 2]
                x_b = coordinates[B, 0]
                y_b = coordinates[B, 1]
                z_b = coordinates[B, 2]

                # distance between A and B
                dist = np.linalg.norm(coordinates[A, :] - coordinates[B, :])
                # third power of distance between A and B
                dist_32 = np.linalg.norm(coordinates[A, :] - coordinates[B, :]) ** 3
                # fifth power of distance between A and B
                dist_52 = np.linalg.norm(coordinates[A, :] - coordinates[B, :]) ** 5

                # AxAx
                gg[A, 0, A, 0] += (3 * cxc * (x_a - x_b) ** 2 / dist_52) - cxc / dist_32
                # AxAy and AyAx
                gg[A, 0, A, 1] += (3 * cxc * (x_a - x_b) * (y_a - y_b)) / dist_52
                gg[A, 1, A, 0] = gg[A, 0, A, 1]
                # AxAz and AzAx
                gg[A, 0, A, 2] += (3 * cxc * (x_a - x_b) * (z_a - z_b)) / dist_52
                gg[A, 2, A, 0] = gg[A, 0, A, 2]
                # AyAy
                gg[A, 1, A, 1] += (3 * cxc * (y_a - y_b) ** 2 / dist_52) - cxc / dist_32
                # AyAz and AzAy
                gg[A, 1, A, 2] += (3 * cxc * (y_a - y_b) * (z_a - z_b)) / dist_52
                gg[A, 2, A, 1] = gg[A, 1, A, 2]
                # AzAz
                gg[A, 2, A, 2] += (3 * cxc * (z_a - z_b) ** 2 / dist_52) - cxc / dist_32

                # AxBx and BxAx
                gg[A, 0, B, 0] += cxc / dist_32 - 3 * cxc * (x_a - x_b) ** 2 / dist_52
                gg[B, 0, A, 0] = gg[A, 0, B, 0]
                # AxBy, AyBx, ByAx, BxAy
                gg[A, 0, B, 1] = -gg[A, 0, A, 1]
                gg[A, 1, B, 0] = -gg[A, 0, A, 1]
                gg[B, 1, A, 0] = -gg[A, 0, A, 1]
                gg[B, 0, A, 1] = -gg[A, 0, A, 1]
                # AxBz, AzBx, BzAx, BxAz
                gg[A, 0, B, 2] = -gg[A, 0, A, 2]
                gg[A, 2, B, 0] = -gg[A, 0, A, 2]
                gg[B, 2, A, 0] = -gg[A, 0, A, 2]
                gg[B, 0, A, 2] = -gg[A, 0, A, 2]
                # AyBy and ByAy
                gg[A, 1, B, 1] += cxc / dist_32 - (3 * cxc * (y_a - y_b) ** 2 / dist_52)
                gg[B, 1, A, 1] = gg[A, 1, B, 1]
                # AyBz, AzBy, ByAz, BzAy
                gg[A, 1, B, 2] = -gg[A, 1, A, 2]
                gg[A, 2, B, 1] = -gg[A, 1, A, 2]
                gg[B, 1, A, 2] = -gg[A, 1, A, 2]
                gg[B, 2, A, 1] = -gg[A, 1, A, 2]
                # AzBz and BzAz
                gg[A, 2, B, 2] += cxc / dist_32 - (3 * cxc * (z_a - z_b) ** 2 / dist_52)
                gg[B, 2, A, 2] = gg[A, 2, B, 2]

                # BxBx
                gg[B, 0, B, 0] = gg[A, 0, A, 0]
                # BxBy and ByBx
                gg[B, 0, B, 1] = gg[A, 0, A, 1]
                gg[B, 1, B, 0] = gg[A, 0, A, 1]
                # BxBz and BzBx
                gg[B, 0, B, 2] = gg[A, 0, A, 2]
                gg[B, 2, B, 0] = gg[A, 0, A, 2]
                # ByBy
                gg[B, 1, B, 1] = gg[A, 1, A, 1]
                # ByBz and BzBy
                gg[B, 1, B, 2] = gg[A, 1, A, 2]
                gg[B, 2, B, 1] = gg[A, 1, A, 2]
                # BzBz
                gg[B, 2, B, 2] = gg[A, 2, A, 2]

    return gg


def hessian(coordinates, charges):
    """Computes the Hessian."""
    natoms = charges.size
    eval_E = []
    for A in range(charges.size):
        for B in range(charges.size):
            if A < B:
                cxc = charges[A] * charges[B]
                x_a = coordinates[A, 0]
                y_a = coordinates[A, 1]
                z_a = coordinates[A, 2]
                x_b = coordinates[B, 0]
                y_b = coordinates[B, 1]
                z_b = coordinates[B, 2]

                # distance between A and B
                dist = np.linalg.norm(coordinates[A, :] - coordinates[B, :])
                # third power of distance between A and B
                dist_32 = np.linalg.norm(coordinates[A, :] - coordinates[B, :]) ** 3
                # fifth power of distance between A and B
                dist_52 = np.linalg.norm(coordinates[A, :] - coordinates[B, :]) ** 5

                d_axax = (3 * cxc * (x_a - x_b) ** 2 / dist_52) - cxc / dist_32

                d_axbx = cxc / dist_32 - 3 * cxc * (x_a - x_b) ** 2 / dist_52

                d_axay = (3 * cxc * (x_a - x_b) * (y_a - y_b)) / dist_52
                d_axby = -d_axay

                d_axaz = (3 * cxc * (x_a - x_b) * (z_a - z_b)) / dist_52
                d_axbz = -d_axaz

                d_ayay = (3 * cxc * (y_a - y_b) ** 2 / dist_52) - cxc / dist_32
                d_aybx = -d_axay
                d_ayby = cxc / dist_32 - (3 * cxc * (y_a - y_b) ** 2 / dist_52)

                d_ayaz = (3 * cxc * (y_a - y_b) * (z_a - z_b)) / dist_52
                d_aybz = -d_ayaz
                d_azaz = (3 * cxc * (z_a - z_b) ** 2 / dist_52) - cxc / dist_32
                d_azbx = -d_axaz
                d_azby = -d_ayaz
                d_azbz = cxc / dist_32 - (3 * cxc * (z_a - z_b) ** 2 / dist_52)
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
