import io
import sys
import numpy as np
from Data import Data
import multiprocessing as mp

def setup_arguments(arguments):
    """Check arguments for errors and returns them ready for execution"""

    # get the arguments
    particles = int(arguments['--particles'])
    iterations = int(arguments['--iterations'])
    alpha = float(arguments['--alpha'])
    beta = float(arguments['--beta'])
    k = int(arguments['--k'])
    max_deletions = int(arguments['--maxdel'])
    tolerance = float(arguments['--tolerance'])
    max_time = int(arguments['--maxtime'])
    if arguments['<runptcl>'] != []:
        multiple_runs = [int(i) for i in arguments['<runptcl>']]
    else:
        multiple_runs = None

    # check for errors
    if particles < 0:
        raise Exception("Error! Particles < 0")
    if iterations < 0:
        raise Exception("Error! Iterations < 0")
    if k < 0:
        raise Exception("Error! K < 0")
    if max_deletions < 0:
        raise Exception("Error! Maxdel < 0")
    if tolerance < 0 or tolerance > 1:
        raise Exception("Error! Tolerance is not between 0 and 1")
    if max_time < 30:
        raise Exception("Error! Minimum time limit is 30 seconds")

    with open(arguments['--infile'], 'r') as f:
        matrix =  np.atleast_2d(np.loadtxt(io.StringIO(f.read()))) # assuring that we at least have 2D array to work with
    mutation_number = matrix.shape[1]
    cells = matrix.shape[0]
    matrix = matrix.tolist()

    # default number of particles = number of CPU cores
    if particles == 0:
        particles = mp.cpu_count()

    mutation_names = _read_mutation_names(arguments['--mutfile'], mutation_number)
    gamma = _read_gamma(arguments['--gamma'], mutation_number)

    return particles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, max_deletions, tolerance, max_time, multiple_runs



# reading file with mutation names, if given in input
# generating them otherwise (1,2,...n)
def _read_mutation_names(path, mutation_number):
    if path:
        with open(path, 'r') as f:
            mutation_names = [l.strip() for l in f.readlines()]
            if len(mutation_names) != mutation_number:
                raise Exception("Mutation names number in file does not match mutation number in data!", len(mutation_names), mutations)
    else:
        mutation_names = [i + 1 for i in range(mutation_number)]
    return mutation_names



# reading file with gamma values or float value, if given in input
def _read_gamma(path, mutation_number):
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
