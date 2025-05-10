import numpy as np

def compute_rdf(pos_A, pos_B, box, max_radius, n_bins):
    """
    Compute RDF between particles A and B in a periodic box.
    """
    N = pos_A.shape[0]
    volume = np.prod(box)
    rho = pos_B.shape[0] / volume
    dr = max_radius / n_bins
    bins = np.linspace(0, max_radius, n_bins + 1)

    # Minimum image convention for periodic distances
    delta = pos_A[:, None, :] - pos_B[None, :, :]
    delta -= box * np.round(delta / box)
    dist = np.linalg.norm(delta, axis=2).ravel()

    # Remove zero distances (self-interactions)
    dist = dist[dist != 0.0]

    # Histogram
    counts, edges = np.histogram(dist, bins=bins)
    shell_volumes = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)

    rdf = counts / (rho * N * shell_volumes)
    cumulative = np.cumsum(counts)

    return edges[:-1], rdf, cumulative
