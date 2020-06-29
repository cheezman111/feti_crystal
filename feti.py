from ase                        import Atoms
from ase.visualize              import view
import matplotlib.pyplot            as plt
import graphdot.kernel.molecular    as gkern
import graphdot
import seaborn                      as sns
import numpy                        as np
import time
import pandas                       as pd
import scipy
import sklearn.linear_model
import random
import pickle


#-----------------------------------------------------------------------
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
for position_step in positions[:2]:
    time1 = time.time()
    atoms = Atoms(symbols, position_step, pbc=True)
    print(str(time.time()-time1))
    time2 = time.time()
    graphs.append(graphdot.Graph.from_ase(atoms))
    print(str(time.time()-time2))
# save graph list into a file
output = open("feti_graphs.dat", "wb")
pickle.dump(graphs[0], output)
output.close()
#-----------------------------------------------------------------------



# load graphs from file
input_file = open("feti_graphs.dat", "rb")
graph = pickle.load(input_file)
input_file.close()

graphs =[graph]

# set hyperparmeters:
nu_h     = 0.3
zeta_h   = 1
lambda_h = 0.1
s_h      = 1
q_h      = 0.05

# use kernel from atomization paper
tang_kernel = gkern.Tang2019MolecularKernel(
                    stopping_probability=q_h,
                    starting_probability=lambda x, y: s_h,
                    element_prior=nu_h,
                    edge_length_scale=lambda_h)

start = time.time()
k = tang_kernel(graphs)
d = np.diag(k)**-0.5
k_sim = np.diag(d).dot(k).dot(np.diag(d))
ax = sns.heatmap(k_sim, square=True,cmap="YlGnBu",linewidths=0)
end = time.time()
print('Time: ' + str(end-start))
plt.show()
