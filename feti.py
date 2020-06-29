from ase                        import Atoms
from ase.visualize              import view
import matplotlib.pyplot            as plt
import graphdot.kernel.molecular    as gkern
import graphdot
import seaborn                      as sns
import numpy                        as np
import time
import pandas as pd
import scipy
import sklearn.linear_model
import random
import pickle


# load graphs from file, which should contain a list of graphdot graphs.
with open("feti_graphs.dat", "rb") as input_file:
    graphs = pickle.load(input_file)


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
k = tang_kernel(graphs[:30])
d = np.diag(k)**-0.5
k_sim = np.diag(d).dot(k).dot(np.diag(d))
ax = sns.heatmap(k_sim, square=True,cmap="YlGnBu",linewidths=0)
end = time.time()
print('Time: ' + str(end-start))
plt.show()


