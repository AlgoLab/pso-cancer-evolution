import io
import numpy
import multiprocessing as mp


class Helper(object):


    def __init__(self, arguments):

        (filename, n_particles, cores, iterations, matrix, truematrix, mutation_number, mutation_names, cells, alpha, beta,
            gamma, k, max_deletions, tolerance, max_time, multiple_runs, quiet, output, automatic_stop) = setup_arguments(arguments)

        self.filename = filename
        self.n_particles = n_particles
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

    # particles
    n_particles = [int(i) for i in arguments['-p'].split(',')]
    multiple_runs = len(n_particles) > 1
    if len(n_particles) == 1:
        n_particles = n_particles[0]
        if n_particles < 2:
            raise Exception("Error! n_particles < 2")

    # cores
    cores = int(arguments['-c'])
    if cores < 1:
        raise Exception("Error! Cores < 1")
    if cores > mp.cpu_count():
        raise Exception("Error! Cores in input are more than this computer's cores")
    if (multiple_runs and any(cores > p for p in n_particles)) or (not(multiple_runs) and cores > n_particles):
        raise Exception("Error! Cores cannot be more than particles")

    # iterations
    iterations = (arguments['-t'])
    automatic_stop = False
    if iterations == None:
        automatic_stop = True
        iterations = 100000
    else:
        iterations = int(iterations)
    if iterations < 1:
        raise Exception("Error! Iterations < 1")

    # matrix
    filename = arguments['-i']
    with open(filename, 'r') as f:
        matrix =  numpy.atleast_2d(numpy.loadtxt(io.StringIO(f.read())))
    mutation_number = matrix.shape[1]
    cells = matrix.shape[0]
    matrix = [list(map(int, x)) for x in matrix.tolist()] #convert matrix to int

    # truematrix
    if arguments['-I'] == None:
        truematrix = None
    else:
        with open(arguments['-I'], 'r') as f:
            truematrix =  numpy.atleast_2d(numpy.loadtxt(io.StringIO(f.read())))
        truematrix = [list(map(int, x)) for x in truematrix.tolist()] #convert matrix to int

    # mutation names
    mutation_names = _read_mutation_names(arguments['-e'], mutation_number)

    # alpha, beta, gamma
    alpha = _read_gamma(arguments['-a'], mutation_number, "alpha")
    beta = float(arguments['-b'])
    gamma = _read_gamma(arguments['-g'], mutation_number, "gamma")

    # k
    k = int(arguments['-k'])
    if k < 0:
        raise Exception("Error! K < 0")

    # max deletions
    max_deletions = float(arguments['-d'])
    if max_deletions != float("+inf"):
        max_deletions = int(max_deletions)
    if max_deletions < 0:
        raise Exception("Error! Maxdel < 0")

    # tolerance
    tolerance = float(arguments['-T'])
    if tolerance < 0 or tolerance > 1:
        raise Exception("Error! Tolerance is not between 0 and 1")

    # maximum time
    max_time = arguments['-m']
    if max_time == None:
        max_time = float("+inf")
    else:
        max_time = int(max_time)
    if max_time < 5:
        raise Exception("Error! Minimum time limit is 5 seconds")
    max_time *= 0.999

    # execution options
    quiet = arguments['--quiet']
    output = arguments['--output']
    if output not in ["image", "plot", "text_file", "all"]:
        raise Exception("Error! Output must be either one of these: (image | plot | text_file | all)")

    return (filename, n_particles, cores, iterations, matrix, truematrix, mutation_number, mutation_names,
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
