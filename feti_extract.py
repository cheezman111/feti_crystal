# This script takes data from the 2013 VASP FeTi data set, after being filtered
# by grep. File: fet_filtered.dat. It produces graphdot graphs for each
# snapshot, adds them to a list and stores them in a file called
# 'feti_graphs.dat'. It was originally filtered from the OUTCAR file.
from ase import Atoms
import graphdot
import pickle

# import data from feti_filtered.dat
positions = []
forces = []

with open('feti_filtered.dat') as feti:
    for line in feti:
        parts = line.split()
        if len(parts) == 3:
            positions.append([])
            forces.append([])
        if len(parts) == 6:
            parts_floats = [float(x) for x in parts]
            positions[-1].append(parts_floats[:3])
            forces[-1].append(parts_floats[3:])

# produce graphs
graphs = []
symbols = 'Fe'*64 + 'Ti'*64
for position_step in positions[:60]:
    atoms = Atoms(symbols, position_step, pbc=True)
    graphs.append(graphdot.Graph.from_ase(atoms))

# save graph list into a file
output = open("feti_graphs.dat", "wb")
pickle.dump(graphs, output)
output.close()
