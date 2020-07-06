# Gradient Descent to find set of hyperparameters which 
# maximize the dissimilarity of the normalized k matrix

import numpy as np
import seaborn as sns
import pandas as pd
import pickle
import random

def get_partials(hypers, f_at_0, beta, target_f):
    partial_derivatives = []
    for i, hyper_value in enumerate(hypers):
        delta_hyper = hyper_value/beta

        hypers1 = hypers.copy()
        hypers1[i] += delta_hyper
        f_at_1  = target_f(*hypers1)

        delta_f = f_at_1 - f_at_0

        partial_derivatives.append(delta_f/delta_hyper)
    return partial_derivatives


def check_bounds(hypers, bounds, beta):
    # checks if hypers are within the ranges specified by bounds.
    # if not, it changes the hyperprameter to be just before the boundary.
    # it returns the potntially modified list of hyperparameters, leaving
    # the original unchanged.
    new_hyper = []
    for hyper, bound in zip(hypers, bounds):
        if hyper < bound[0]:
            hyper = bound[0] + np.abs(bound[0]/beta)
        if hyper > bound[1]:
            hyper = bound[1] - np.abs(bound[1]/beta)
        new_hyper.append(hyper)
    return new_hyper


if __name__ == '__main__':

    target_f = lambda x, y : x**2 + y**2

    # establish starting point 
    # legend: [nu, lambda]
    hypers =  [3, 1]

    # establish bounds for hypers
    bounds = [(-10,10), (-10,10)]

    alpha = .1    # learning rate
    beta = 100     # size of hyperparameter differential, delta_h=h/beta

    for iteration in range(50):
        print('Iteration: ', iteration)
        print('Hypers:   ', hypers)

        # calculate 
        f_at_0  = target_f(*hypers)
        print('F(hypers):     ', f_at_0)

        # produce partial derivatives:
        partial_derivatives = get_partials(hypers, f_at_0, beta, target_f)
        print('Partials: ', partial_derivatives)

        # update hypers
        hypers = list(map( (lambda hyp,part:hyp-alpha*part), hypers,
                        partial_derivatives))
        hypers = check_bounds(hypers, bounds, beta)
        print()


