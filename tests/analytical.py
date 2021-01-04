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
                gg[A, 0, B, 1] += -(3 * cxc * (x_a - x_b) * (y_a - y_b)) / dist_52
                gg[A, 1, B, 0] = gg[A, 0, B, 1]
                gg[B, 1, A, 0] = gg[A, 0, B, 1]
                gg[B, 0, A, 1] = gg[A, 0, B, 1]
                # AxBz, AzBx, BzAx, BxAz
                gg[A, 0, B, 2] += -(3 * cxc * (x_a - x_b) * (z_a - z_b)) / dist_52
                gg[A, 2, B, 0] = gg[A, 0, B, 2]
                gg[B, 2, A, 0] = gg[A, 0, B, 2]
                gg[B, 0, A, 2] = gg[A, 0, B, 2]
                # AyBy and ByAy
                gg[A, 1, B, 1] += cxc / dist_32 - (3 * cxc * (y_a - y_b) ** 2 / dist_52)
                gg[B, 1, A, 1] = gg[A, 1, B, 1]
                # AyBz, AzBy, ByAz, BzAy
                gg[A, 1, B, 2] += -(3 * cxc * (y_a - y_b) * (z_a - z_b)) / dist_52
                gg[A, 2, B, 1] = gg[A, 1, B, 2]
                gg[B, 1, A, 2] = gg[A, 1, B, 2]
                gg[B, 2, A, 1] = gg[A, 1, B, 2]
                # AzBz and BzAz
                gg[A, 2, B, 2] += cxc / dist_32 - (3 * cxc * (z_a - z_b) ** 2 / dist_52)
                gg[B, 2, A, 2] = gg[A, 2, B, 2]

                # BxBx
                gg[B, 0, B, 0] += (3 * cxc * (x_a - x_b) ** 2 / dist_52) - cxc / dist_32
                # BxBy and ByBx
                gg[B, 0, B, 1] += (3 * cxc * (x_a - x_b) * (y_a - y_b)) / dist_52
                gg[B, 1, B, 0] = gg[B, 0, B, 1]
                # BxBz and BzBx
                gg[B, 0, B, 2] += (3 * cxc * (x_a - x_b) * (z_a - z_b)) / dist_52
                gg[B, 2, B, 0] = gg[B, 0, B, 2]
                # ByBy
                gg[B, 1, B, 1] += (3 * cxc * (y_a - y_b) ** 2 / dist_52) - cxc / dist_32
                # ByBz and BzBy
                gg[B, 1, B, 2] += (3 * cxc * (y_a - y_b) * (z_a - z_b)) / dist_52
                gg[B, 2, B, 1] = gg[B, 1, B, 2]
                # BzBz
                gg[B, 2, B, 2] += (3 * cxc * (z_a - z_b) ** 2 / dist_52) - cxc / dist_32

    return gg


"""
coordinates = np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 1.0, 3.0],
                    [3.0, 1.0, 2.0],
                    [5.0, 4.0, 3.0],
                    [4.0, 3.0, 5.0],
                ]
            )
charges = np.array([7.0, 6.0, 1.0, 5.0, 5.0])

a_hessian = hessian_redundant(coordinates,charges)
#print(a_hessian)
"""
