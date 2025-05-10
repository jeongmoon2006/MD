# analyze_rdf.py

import MDAnalysis as mda
from rdf import compute_rdf, rdf_average

# Example usage:
u = mda.Universe("topology.pdb", "trajectory.xtc")
fs_atom_name = "name O"  # Example: oxygen atoms
sn_atom_name = "name H"  # Example: hydrogen atoms
time_indices = range(0, 100, 10)  # Every 10th frame
n_bins = 100
max_radius = 10

bins, rdf_avg, nr_avg = rdf_average(u, time_indices, fs_atom_name, sn_atom_name, n_bins, max_radius)

# Save or plot as needed
import matplotlib.pyplot as plt
plt.plot(bins, rdf_avg)
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.show()
