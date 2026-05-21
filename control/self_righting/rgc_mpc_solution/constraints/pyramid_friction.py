import numpy as np
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.utils.contact_surfaces import contact_surfaces


def pyramid_friction_matrix(n, t1, t2, mu):
    Cf = np.vstack([-mu * n + t1, -mu * n + t2, mu * n + t2, mu * n + t1, n])

    return Cf


def pyramid_friction(contacts, mu):

    n_contacts = contacts.shape[0]

    if n_contacts < 3:
        raise ValueError("At least 3 contacts are required.")

    Cf_list = []

    for i in range(n_contacts):

        p0 = contacts[i]
        p1 = contacts[(i + 1) % n_contacts]
        p2 = contacts[(i + 2) % n_contacts]

        n, t1, t2 = contact_surfaces(p0, p1, p2)

        Cf_i = pyramid_friction_matrix(n, t1, t2, mu)

        Cf_list.append(Cf_i)

    return block_diag(*Cf_list)
