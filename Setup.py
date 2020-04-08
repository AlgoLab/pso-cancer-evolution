import io
import sys
import numpy as np
from Data import Data
import multiprocessing as mp

def setup_arguments(arguments):

    particles = int(arguments['--particles'])
    iterations = int(arguments['--iterations'])
    alpha = float(arguments['--alpha'])
    beta = float(arguments['--beta'])
    k = int(arguments['--k'])
    w = float(arguments['--w'])
    c1 = float(arguments['--c1'])
    c2 = float(arguments['--c2'])
    max_deletions = int(arguments['--maxdel'])
    max_time = int(arguments['--maxtime'])
    if arguments['<runptcl>'] != []:
        multiple_runs = [int(i) for i in arguments['<runptcl>']]
    else:
        multiple_runs = None


    #checking for errors
    if particles < 0:
        raise Exception("ERROR! Particles < 0")
    if iterations < 0:
        raise Exception("ERROR! Iterations < 0")
    if k < 0:
        raise Exception("ERROR! k < 0")
    if w < 0:
        raise Exception("ERROR! w < 0")
    if c1 < 0:
        raise Exception("ERROR! c1 < 0")
    if c2 < 0:
        raise Exception("ERROR! c2 < 0")
    if w == 0 and c1 == 0 and c2 == 0:
        raise Exception("ERROR! w,c1,c2 are all = 0")
    if max_deletions < 0:
        raise Exception("ERROR! maxdel < 0")
    if max_time < 30:
        raise Exception("ERROR! minimum time limit is 30 seconds")


    with open(arguments['--infile'], 'r') as f:
        matrix =  np.atleast_2d(np.loadtxt(io.StringIO(f.read()))) # assuring that we at least have 2D array to work with
    mutation_number = matrix.shape[1]
    cells = matrix.shape[0]
    matrix = matrix.tolist()

    # default number of particles = number of cores of cpu
    if particles == 0:
        particles = mp.cpu_count()

    mutation_names = read_mutation_names(arguments['--mutfile'], mutation_number)
    gamma = read_gamma(arguments['--gamma'], mutation_number)

    return particles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, max_time, multiple_runs



# reading file with mutation names, if given in input
# generating them otherwise
def read_mutation_names(path, mutation_number):
    if path:
        with open(path, 'r') as f:
            mutation_names = [l.strip() for l in f.readlines()]
            if len(mutation_names) != mutation_number:
                raise Exception("Mutation names number in file does not match mutation number in data!", len(mutation_names), mutations)
    else:
        mutation_names = [i + 1 for i in range(mutation_number)]
    return mutation_names



# reading file with gamma values or float value, if given in input
def read_gamma(path, mutation_number):
    gamma = path
    try:
        gamma = float(gamma)
        gamma = [gamma]*mutation_number
    except ValueError:
        with open(gamma) as f:
            tmp = [float(l.strip()) for l in f.readlines()]
            if len(tmp) != mutation_number:
                raise Exception("gammas number does not match mutation names number!", len(mutation_names), mutation_number)
        gamma = tmp
    return gamma
