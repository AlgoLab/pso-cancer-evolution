import io
import numpy
import multiprocessing as mp


class Helper(object):


    def __init__(self, arguments):

        (filename, nparticles, cores, iterations, matrix, truematrix, mutation_number, mutation_names, cells, alpha, beta,
            gamma, k, max_deletions, tolerance, max_time, multiple_runs, quiet, output, automatic_stop) = setup_arguments(arguments)

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
        self.quiet = quiet
        self.output = output
        self.automatic_stop = automatic_stop


def setup_arguments(arguments):
    """Check arguments for errors and returns them ready for execution"""

    # print(arguments)

    # get the arguments
    particles = int(arguments['-p'])
    cores = int(arguments['-c'])
    iterations = (arguments['-t'])
    automatic_stop = False
    if iterations == None:
        automatic_stop = True
        iterations = 100000
    else:
        iterations = int(iterations)
    beta = float(arguments['-b'])
    k = int(arguments['-k'])
    max_deletions = float(arguments['-d'])
    if max_deletions != float("+inf"):
        max_deletions = int(max_deletions)
    tolerance = float(arguments['-T'])
    max_time = int(arguments['-m'])
    if arguments['-M'] != None:
        multiple_runs = [int(i) for i in arguments['-M'].split(',')]
    else:
        multiple_runs = None
    quiet = arguments['--quiet']
    output = arguments['--output']

    # check for errors
    if particles < 2:
        raise Exception("Error! Particles < 2")
    if cores < 1:
        raise Exception("Error! Cores < 1")
    if (multiple_runs and any(cores > particles for particles in multiple_runs)) or (not(multiple_runs) and cores > particles):
        raise Exception("Error! Cores cannot be more than particles")
    if cores > mp.cpu_count():
        raise Exception("Error! Cores in input are more than this computer's cores")
    if iterations < 1:
        raise Exception("Error! Iterations < 1")
    if k < 0:
        raise Exception("Error! K < 0")
    if max_deletions < 0:
        raise Exception("Error! Maxdel < 0")
    if tolerance < 0 or tolerance > 1:
        raise Exception("Error! Tolerance is not between 0 and 1")
    if max_time < 10:
        raise Exception("Error! Minimum time limit is 10 seconds")
    if output not in ["image", "plot", "text_file", "all"]:
        raise Exception("Error! Output must be either one of these: (image | plot | text_file | all)")

    # read matrix
    filename = arguments['-i']
    with open(filename, 'r') as f:
        matrix =  numpy.atleast_2d(numpy.loadtxt(io.StringIO(f.read())))
    mutation_number = matrix.shape[1]
    cells = matrix.shape[0]
    matrix = [list(map(int, x)) for x in matrix.tolist()] #convert matrix to int

    alpha = _read_gamma(arguments['-a'], mutation_number, "alpha")
    gamma = _read_gamma(arguments['-g'], mutation_number, "gamma")
    mutation_names = _read_mutation_names(arguments['-e'], mutation_number)

    # read truematrix if given in input
    if arguments['-I'] == None:
        truematrix = None
    else:
        with open(arguments['-I'], 'r') as f:
            truematrix =  numpy.atleast_2d(numpy.loadtxt(io.StringIO(f.read())))
        truematrix = [list(map(int, x)) for x in truematrix.tolist()] #convert matrix to int

    max_time -= 0.5

    return (filename, particles, cores, iterations, matrix, truematrix, mutation_number, mutation_names,
        cells, alpha, beta, gamma, k, max_deletions, tolerance, max_time, multiple_runs, quiet, output, automatic_stop)


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


# reading file with multiple or single value, if given in input (for alpha and gamma)
def _read_gamma(path, mutation_number, type):
    values = path
    try:
        values = float(values)
        values = [values]*mutation_number
    except ValueError:
        with open(values) as f:
            tmp = [float(l.strip()) for l in f.readlines()]
            if len(tmp) != mutation_number:
                raise Exception(type + " rates number does not match mutation names number!", len(mutation_names), mutation_number)
        values = tmp
    return values
