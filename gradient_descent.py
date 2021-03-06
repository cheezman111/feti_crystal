# Gradient Descent to find set of hyperparameters which 
# maximize the dissimilarity of the normalized k matrix

import numpy as np
import seaborn as sns
import pandas as pd
import graphdot
import pickle
import graphdot.kernel.molecular as gkern
import random
from matplotlib.pyplot import plot, draw, show


# import data from feti_filtered.dat
with open("feti_graphs.dat", "rb") as input_file:
    graphs = pickle.load(input_file)

# import second data from feti_filtered.dat for randomization
with open("feti_graphs.dat", "rb") as input_file:
    graphs_random = pickle.load(input_file)
random.shuffle(graphs_random)


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


def check_bounds(hypers, bounds, beta):
    # checks if hypers are within the ranges specified by bounds.
    # if not, it changes the hyperprameter to be just before the boundary.
    # it returns the potntially modified list of hyperparameters, leaving
    # the original unchanged.
    new_hyper = []
    for hyper, bound in zip(hypers, bounds):
        if hyper < bound[0]:
            hyper = bound[0] + np.abs(hyper/beta)
        if hyper > bound[1]:
            hyper = bound[1] - np.abs(hyper/beta)
        new_hyper.append(hyper)
    return new_hyper


if __name__ == '__main__':
    # establish starting point 
    # legend: [nu, lambda]
    hypers =  [0.24940064654666944, 0.0030006889798517445]

    # establish bounds for hypers
    bounds = [(0,1), (0,float('inf'))]

    alpha = .1    # learning rate
    beta = 100     # size of hyperparameter differential, delta_h=h/beta

    means_history = []
    for iteration in range(50):
        print('Iteration: ', iteration)
        print('Hypers:   ', hypers)

        # calculate 
        k_sim0 = hyper_to_k_sim(*hypers)
        mean0  = k_sim_to_mean(k_sim0)
        means_history.append(mean0)
        print('Mean:     ', mean0)

        # produce partial derivatives:
        partial_derivatives = get_partials(hypers, mean0, beta)
        print('Partials: ', partial_derivatives)

        # update hypers
        hypers = list(map( (lambda hyp,part:hyp-alpha*part), hypers,
                        partial_derivatives))
        hypers = check_bounds(hypers, bounds, beta)
        print()


