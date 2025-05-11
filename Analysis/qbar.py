import numpy as np
import MDAnalysis as mda
from scipy.special import sph_harm

def neighbor_list(positions, box, cutoff):
    n_atoms = positions.shape[0]
    dist_vec = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n, n, 3)
    dist_vec -= box * np.round(dist_vec / box)
    dist = np.sqrt(np.sum(dist_vec**2, axis=-1))
    neighbor_mask = (dist < cutoff)
    np.fill_diagonal(neighbor_mask, False)  # exclude self
    n_list = [np.where(neighbor_mask[i])[0].tolist() for i in range(n_atoms)]
    return n_list

def compute_qlm_i_vectorized(i, positions, box, neighbor_list, l):
    m_values = np.arange(-l, l+1)
    neighbors = neighbor_list[i]
    Nn = len(neighbors)
    if Nn == 0:
        return np.zeros(2*l+1, dtype=complex)
    rij = positions[neighbors] - positions[i]
    rij -= box * np.round(rij / box)
    r = np.linalg.norm(rij, axis=1)
    theta = np.arccos(rij[:, 2] / r)
    phi = np.arctan2(rij[:, 1], rij[:, 0])
    qlm_sum = np.array([np.sum(sph_harm(m, l, phi, theta)) for m in m_values])
    qlm_i = qlm_sum / Nn
    return qlm_i

def compute_qbar_lm_i(i, qlm_array, neighbor_list):
    neighbors = neighbor_list[i]
    Nn = len(neighbors)
    if Nn == 0:
        return qlm_array[i]
    neighbor_sum = np.sum(qlm_array[neighbors], axis=0)
    qbar_lm_i = (qlm_array[i] + neighbor_sum) / (Nn + 1)
    return qbar_lm_i

def compute_qbar_l(positions, box, cutoff, l=6):
    """Compute q̄_l(i) for all atoms."""
    n_atoms = positions.shape[0]

    # Step 1: Build neighbor list
    nlist = neighbor_list(positions, box, cutoff)

    # Step 2: Compute qlm for all atoms
    qlm_array = np.array([
        compute_qlm_i_vectorized(i, positions, box, nlist, l)
        for i in range(n_atoms)
    ])

    # Step 3: Compute q̄lm for all atoms
    qbar_lm_array = np.array([
        compute_qbar_lm_i(i, qlm_array, nlist)
        for i in range(n_atoms)
    ])

    # Step 4: Compute rotationally invariant scalar q̄_l
    qbar_l = np.sqrt(4*np.pi/(2*l+1) * np.sum(np.abs(qbar_lm_array)**2, axis=1))

    return qbar_l


def time_qbar_l(u, t_array, atom_name, n_bins, cutoff, l):
    atoms = u.select_atoms(atom_name)
    pos = atoms.positions
    N = pos.shape[0]

    qbar_l_list = []

    for i in t_array:
        
        u.trajectory[np.int32(i)]
        box = u.dimensions[:3]
        pos = atoms.positions
        
        qbar_l = compute_qbar_l(pos, box, cutoff, l)
        qbar_l_list += [qbar_l]

    return qbar_l_list
