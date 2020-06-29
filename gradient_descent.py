# Gradient Descent to find set of hyperparameters which 
# maximize the dissimilarity of the normalized k matrix

import numpy as np
import seaborn as sns
import pandas as pd
import graphdot
import pickle
import graphdot.kernel.molecular as gkern

# import data from feti_filtered.dat
with open("feti_graphs.dat", "rb") as input_file:
    graphs = pickle.load(input_file)

# returns a normalized k matrix given a set of hyperparameters.
def hyper_to_k_sim(nu, lambda_h):
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
    k = tang_kernel(graphs[:50])
    d = np.diag(k)**-0.5
    k_sim = np.diag(d).dot(k).dot(np.diag(d))
    
    return k_sim


def get_partials(hypers, mean0, beta):
    partial_derivatives = []       
    for i, hyper_value in enumerate(hypers):
        delta_hyper = hyper_value/beta
        
        hypers1 = hypers.copy()
        hypers1[i] += delta_hyper
        k_sim1 = hyper_to_k_sim(*hypers1)
        mean1  = k_sim_to_mean(k_sim1)

        change_in_mean = mean1-mean0

        partial_derivatives.append(change_in_mean/delta_hyper)
    return partial_derivatives


def k_sim_to_variance(k_sim):
    flat = k_sim.flatten()
    return np.var(flat)

def k_sim_to_mean(k_sim):
    flat = k_sim.flatten()
    return np.mean(flat)


# establish starting point (legend: [nu, lambda])
hypers = [.3, .1]

alpha = .1     # learning rate
beta = 100     # size of hyperparameter differential, delta_h=h/beta

for iteration in range(10):
    print('Hypers:   ', hypers)

    # calculate 
    k_sim0 = hyper_to_k_sim(*hypers)
    mean0  = k_sim_to_mean(k_sim0)
    print('Mean:     ', mean0)
    
    # produce partial derivatives:
    partial_derivatives = get_partials(hypers, mean0, beta)
    print('Partials: ', partial_derivatives)
    
    # update hypers
    hypers = list(map( (lambda hyp,part:hyp-alpha*part), hypers, 
                        partial_derivatives))



