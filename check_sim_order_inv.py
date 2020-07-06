# This script will check if the choice of hyperparameters affects the order of 
# similarity calculated from the graph kernel method. 

import numpy as np
import seaborn as sns
import pandas as pd
import graphdot
import pickle
import graphdot.kernel.molecular as gkern
import random


# import data from feti_filtered.dat
with open("feti_graphs.dat", "rb") as input_file:
    graphs = pickle.load(input_file)


# returns a normalized k matrix given a set of hyperparameters.
def hyper_to_k(nu, lambda_h, n):
    zeta = 1
    s = 1
    # try higher q for shorter mean path lengths
    q = .2
    # use kernel from atomization paper
    tang_kernel = gkern.Tang2019MolecularKernel(
                        stopping_probability=q,
                        starting_probability=lambda x, y: s,
                        element_prior=nu,
                        edge_length_scale=lambda_h)
    k = tang_kernel(graphs[:n], lmin=1)
    return k


def k_to_k_sim(k):
    d = np.diag(k)**-0.5
    k_sim = np.diag(d).dot(k).dot(np.diag(d))
    return k_sim


def k_sim_to_mean(k_sim):
    flat = k_sim.flatten()
    return np.mean(flat)


# creates a list of matrix indices with corresponding similarities, ordered
# by similarity
def order_matrix(k_sim):
    n = k_sim.shape[0]
    indices_and_sims = []
    for x in range(n):
        for y in range(n):
            if x < y:
                indices = (x,y)
                sims = k_sim[x][y]
                indices_and_sims.append((indices, sims))
    indices_and_sims.sort(key=lambda x: x[1])
    return indices_and_sims


if __name__ == '__main__':
    n = 50
    k_sim1 = k_to_k_sim(hyper_to_k(.3, .1, n))
    l1 = order_matrix(k_sim1)
    k_sim2 = k_to_k_sim(hyper_to_k(.8,  1, n))
    l2 = order_matrix(k_sim2)

    indices1, sims1 = zip(*l1)
    indices2, sims2 = zip(*l2)

    print(indices1[:10])
    print(indices2[:10])
