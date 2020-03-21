# -*- coding:utf-8 -*-

"""Particle Swarm Optimization for Cancer Evolution

Usage:
    main.py (--infile <infile>) [--particles <particles>] [--iterations <iterations>] [--alpha=<alpha>] [--beta=<beta>] [--gamma=<gamma>] [--k=<k>] [--w=<w>] [--c1=<c1>] [--c2=<c2>] [--maxdel=<max_deletions>] [--mutfile <mutfile>] [--multiple <runptcl>...]
    main.py -h | --help
    main.py -v | --version

Options:
    -h --help                               Shows this screen.
    -v --version                            Shows version.
    -i infile --infile infile               Matrix input file.
    -m mutfile --mutfile mutfile            Path of the mutation names. If this parameter is not used, then the mutations will be named progressively from 1 to mutations.
    -p particles --particles particles      Number of particles to use for PSO [default: 5].
    -t iterations --iterations iterations   Number of iterations [default: 3].
    --alpha=<alpha>                         False negative rate [default: 0.15].
    --beta=<beta>                           False positive rate [default: 0.00001].
    --gamma=<gamma>                         Loss rate for each mutation (single float for every mutations or file with different rates) [default: 0.5].
    --w=<w>                                 Inertia factor [default: 0.5].
    --c1=<c1>                               Learning factor for particle best [default: 0.25].
    --c2=<c2>                               Learning factor for swarm best [default: 0.75].
    --k=<k>                                 K value of Dollo(k) model used as phylogeny tree [default: 3].
    --maxdel=<max_deletions>                Maximum number of total deletions allowed [default: 10].
"""

import io
import sys
import os
import copy

import numpy as np
from docopt import docopt

import matplotlib.pyplot as plt

from Data import Data
import pso
from datetime import datetime


def main(argv):
    arguments = docopt(__doc__, version = "PSO-Cancer-Evolution 2.0")

    particles = int(arguments['--particles'])
    iterations = int(arguments['--iterations'])
    alpha = float(arguments['--alpha'])
    beta = float(arguments['--beta'])
    k = int(arguments['--k'])
    w = float(arguments['--w'])
    c1 = float(arguments['--c1'])
    c2 = float(arguments['--c2'])
    max_deletions = int(arguments['--maxdel'])
    runs = list(map(int, arguments['<runptcl>']))

    matrix = read_matrix(arguments['--infile'])

    mutation_number = matrix.shape[1]
    cells = matrix.shape[0]

    mutation_names = read_mutation_names(arguments['--mutfile'], mutation_number)
    gamma = read_gamma(arguments['--gamma'], mutation_number)


    matrix = matrix.tolist()


    base_dir = "results" + datetime.now().strftime("%Y%m%d%H%M%S")

    if runs:
        runs_data = []
        for r, ptcl in enumerate(runs):
            print ("=== Run number %d ===" % r)
            run_dir = base_dir + "/p%d_i%d" % (ptcl, iterations)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            data, helper = pso.init(ptcl, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions)
            data.summary(helper, run_dir)
            runs_data.append(data)
        Data.runs_summary(runs, runs_data, base_dir)

    else:
        run_dir = base_dir + "/p%d_i%d" % (particles, iterations)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        data, helper = pso.init(particles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions)
        data.summary(helper, run_dir)




def read_matrix(path):
    with open(path, 'r') as f:
        # assuring that we at least have 2D array to work with
        return np.atleast_2d(np.loadtxt(io.StringIO(f.read())))


def read_mutation_names(path, mutation_number):
    if path:
        with open(path, 'r') as f:
            mutation_names = [l.strip() for l in f.readlines()]
            if len(mutation_names) != mutation_number:
                raise Exception("Mutation names number in file does not match mutation number in data!", len(mutation_names), mutations)
    else:
        mutation_names = [i + 1 for i in range(mutation_number)]
    return mutation_names


def read_gamma(path, mutation_number):
    gamma = path
    try:
        gamma = float(gamma)
        gamma = [gamma]*mutation_number
    except ValueError:
        with open(gamma) as f:
            tmp = [float(l.strip()) for l in f.readlines()]
            if len(tmp) != mutatio_number:
                raise Exception("gammas number does not match mutation names number!", len(mutation_names), mutation_number)
        gamma = tmp
    return gamma




if __name__ == "__main__":
    main(sys.argv[1:])
