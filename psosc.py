"""
Particle Swarm Optimization Single Cell inference

Usage:
    psosc.py (--infile <infile>) [--particles <particles>] [--iterations <iterations>] [--alpha=<alpha>] [--beta=<beta>] [--gamma=<gamma>] [--k=<k>] [--maxdel=<max_deletions>] [--mutfile <mutfile>] [--tolerance=<tolerance>] [--maxtime=<maxtime>] [--multiple <runptcl>...]
    psosc.py -h | --help
    psosc.py -v | --version

Options:
    -h --help                                   Shows this screen.
    -v --version                                Shows version.
    -i infile | --infile infile                 Matrix input file.
    -m mutfile | --mutfile mutfile              Path of the mutation names. If not used, then the mutations will be named progressively from 1 to mutations.
    -p particles | --particles particles        Number of particles to use for PSO. If not used or zero, it will be number of CPU cores [default: 0]
    -t iterations | --iterations iterations     Number of iterations. If not used or zero, PSO will stop when stuck on a best fitness value (or after maxtime of total execution) [default: 0].
    --alpha=<alpha>                             False negative rate [default: 0.15].
    --beta=<beta>                               False positive rate [default: 0.00001].
    --gamma=<gamma>                             Loss rate for each mutation (single float for every mutations or file with different rates) [default: 1].
    --k=<k>                                     K value of Dollo(k) model used as phylogeny tree [default: 3].
    --maxdel=<max_deletions>                    Maximum number of total deletions allowed [default: 5].
    --tolerance=<tolerance>                     Minimum relative improvement (between 0 and 1) in the last 200 iterations in order to keep going, if iterations are zero [default: 0.005].
    --maxtime=<maxtime>                         Maximum time (in seconds) of total PSOSC execution [default: 300].

"""

import Setup
from Helper import Helper
from Node import Node
from Operation import Operation as Op
from Particle import Particle
from Tree import Tree
from Data import Data
import os
import sys
import time
from docopt import docopt
from datetime import datetime
import multiprocessing


# global scope
helper = None
data = None


def main(argv):
    arguments = docopt(__doc__, version = "PSOSC-Cancer-Evolution 2.0")
    (particles, iterations, matrix, mutation_number, mutation_names, cells,
        alpha, beta, gamma, k, max_deletions, tolerance, max_time, multiple_runs) = Setup.setup_arguments(arguments)

    base_dir = "results" + datetime.now().strftime("%Y%m%d%H%M%S")

    if multiple_runs is None:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        data, helper = pso(particles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, max_deletions, tolerance, max_time)
        data.summary(helper, base_dir)
    else:
        runs_data = []
        for r, ptcl in enumerate(multiple_runs):
            print ("\n\n===== Run number %d =====" % r)
            run_dir = base_dir + "/particles%d_run%d" % (ptcl, r)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            data, helper = pso(ptcl, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, max_deletions, tolerance, max_time)
            data.summary(helper, run_dir)
            runs_data.append(data)
        Data.runs_summary(multiple_runs, runs_data, base_dir)


def pso(nparticles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, max_deletions, tolerance, max_time):
    global helper
    global data
    helper = Helper(matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, max_deletions, tolerance, max_time)
    data = Data(nparticles)

    print("\n • PARTICLES START-UP")
    particles = pso_initialization(nparticles)

    print("\n • PSO RUNNING...")
    print("\t  Time\t\t Best likelihood so far")
    pso_execution(particles, iterations)

    print("\n • FINAL RESULTS")
    print("\t- time to complete pso with %d particles: %s seconds" % (data.nofparticles, str(round(data.pso_end - data.pso_start, 2))))
    print("\t- best likelihood: %s\n" % str(round(helper.best_particle.best.likelihood, 2)))

    return data, helper


def pso_initialization(nparticles):
    """Creates the particles"""
    global helper
    global data
    data.initialization_start = time.time()

    particles = [Particle(helper.cells, helper.mutation_number, helper.mutation_names, n) for n in range(nparticles)]
    helper.best_particle = particles[0]
    for p in particles:
        p.current_tree.likelihood = Tree.greedy_loglikelihood(p.current_tree, helper.matrix, helper.cells, helper.mutation_number, helper.alpha, helper.beta)
        p.best.likelihood = p.current_tree.likelihood
        if (p.current_tree.likelihood > helper.best_particle.best.likelihood):
            helper.best_particle = p
    data.starting_likelihood = helper.best_particle.best.likelihood

    data.initialization_end = time.time()
    return particles


def pso_execution(particles, iterations):
    """Runs the particles simultaneously"""
    global helper
    global data
    data.pso_start = time.time()

    # creating shared memory between processes
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    ns = manager.Namespace()

    # coping data into shared memory
    ns.best_swarm = helper.best_particle.best.copy()
    ns.best_iteration_likelihoods = []
    ns.particle_iteration_times = data.particle_iteration_times
    ns.stop = False
    ns.automatic_stop = iterations == 0
    ns.operations = [2,3]

    # run particle processes
    for p in particles:
        p.particle_start(iterations, helper, ns, lock)
    for p in particles:
        p.particle_join()

    # copying back data from shared memory
    data.best_iteration_likelihoods = ns.best_iteration_likelihoods
    data.particle_iteration_times = ns.particle_iteration_times
    helper.best_particle.best = ns.best_swarm.copy()

    data.pso_end = time.time()


if __name__ == "__main__":
    main(sys.argv[1:])
