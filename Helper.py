import io
import sys
import numpy
from Data import Data
import multiprocessing as mp


class Helper(object):


    def __init__(self, arguments):

        (filename, nparticles, cores, iterations, matrix, truematrix, mutation_number, mutation_names, cells, alpha, beta,
            gamma, k, max_deletions, tolerance, max_time, multiple_runs, silent, output, automatic_stop) = setup_arguments(arguments)

        # psosc arguments
        self.filename = filename
        self.nparticles = nparticles
        self.cores = cores
        self.iterations = iterations
        self.matrix = matrix
        self.truematrix = truematrix
        self.mutation_number = mutation_number
        self.mutation_names = mutation_names
        self.cells = cells
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.max_deletions = max_deletions
        self.tolerance = tolerance
        self.max_time = max_time
        self.multiple_runs = multiple_runs
        self.silent = silent
        self.output = output
        self.automatic_stop = automatic_stop


def setup_arguments(arguments):
    """Check arguments for errors and returns them ready for execution"""

    # print(arguments)

    # get the arguments
    particles = int(arguments['--particles'])
    cores = int(arguments['--cores'])
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
    silent = arguments['--silent']
    output = arguments['--output']

    # check for errors
    if particles < 0:
        raise Exception("Error! Particles < 0")
    if cores < 0:
        raise Exception("Error! Cores < 0")
    if (multiple_runs and any(cores > particles for particles in multiple_runs)) or (not(multiple_runs) and cores > particles):
        raise Exception("Error! Cores cannot be more than particles")
    if cores > mp.cpu_count():
        raise Exception("Error! Cores in input are more than this computer's cores")
    if iterations < 0:
        raise Exception("Error! Iterations < 0")
    if k < 0:
        raise Exception("Error! K < 0")
    if max_deletions < 0:
        raise Exception("Error! Maxdel < 0")
    if tolerance < 0 or tolerance > 1:
        raise Exception("Error! Tolerance is not between 0 and 1")
    if max_time < 20:
        raise Exception("Error! Minimum time limit is 20 seconds")
    if output not in ["image", "plot", "text_file", "all"]:
        raise Exception("Error! Output must be either one of these: (image | plot | text_file | all)")

    # read matrix
    filename = arguments['--infile']
    with open(filename, 'r') as f:
        matrix =  numpy.atleast_2d(numpy.loadtxt(io.StringIO(f.read())))
    mutation_number = matrix.shape[1]
    cells = matrix.shape[0]
    matrix = [list(map(int, x)) for x in matrix.tolist()] #convert matrix to int

    # read truematrix if given in input
    if arguments['--truematrix'] == "0":
        truematrix = 0
    else:
        with open(arguments['--truematrix'], 'r') as f:
            truematrix =  numpy.atleast_2d(numpy.loadtxt(io.StringIO(f.read())))
        truematrix = [list(map(int, x)) for x in truematrix.tolist()] #convert matrix to int

    # default number of cores = half of this computer's cpu
    if cores == 0:
        cores = int(mp.cpu_count() / 2)

    # default number of particles = number of CPU cores used
    if particles == 0:
        particles = cores

    mutation_names = _read_mutation_names(arguments['--mutfile'], mutation_number)
    gamma = _read_gamma(arguments['--gamma'], mutation_number)

    max_time -= 0.5

    automatic_stop = False
    if iterations == 0:
        automatic_stop = True
        iterations = 100000

    return (filename, particles, cores, iterations, matrix, truematrix, mutation_number, mutation_names,
        cells, alpha, beta, gamma, k, max_deletions, tolerance, max_time, multiple_runs, silent, output, automatic_stop)


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
