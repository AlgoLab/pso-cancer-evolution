
"""Particle Swarm Optimization Single Cell inference

Usage:
    prova.py (--infile <infile>) [--particles <particles>] [--iterations <iterations>] [--alpha=<alpha>] [--beta=<beta>] [--gamma=<gamma>] [--k=<k>] [--w=<w>] [--c1=<c1>] [--c2=<c2>] [--maxdel=<max_deletions>] [--mutfile <mutfile>] [--multiple <runptcl>...] [--maxtime=<maxtime>]
    prova.py -h | --help
    prova.py -v | --version

Options:
    -h --help                               Shows this screen.
    -v --version                            Shows version.
    -i infile --infile infile               Matrix input file.
    -m mutfile --mutfile mutfile            Path of the mutation names. If not used, then the mutations will be named progressively from 1 to mutations.
    -p particles --particles particles      Number of particles to use for PSO. If not used or zero, it will be estimated based on the number of particles and cells [default: 0]
    -t iterations --iterations iterations   Number of iterations. If not used or zero, PSO will stop when stuck on a best fitness value (or after around 3 minutes of total execution) [default: 0].
    --alpha=<alpha>                         False negative rate [default: 0.15].
    --beta=<beta>                           False positive rate [default: 0.00001].
    --gamma=<gamma>                         Loss rate for each mutation (single float for every mutations or file with different rates) [default: 0.5].
    --w=<w>                                 Inertia factor [default: 1].
    --c1=<c1>                               Learning factor for particle best [default: 1].
    --c2=<c2>                               Learning factor for swarm best [default: 1].
    --k=<k>                                 K value of Dollo(k) model used as phylogeny tree [default: 3].
    --maxdel=<max_deletions>                Maximum number of total deletions allowed [default: 10].
    --maxtime=<maxtime>                     Maximum time (in seconds) of total PSOSC execution [default: 300].
"""

import os
import sys
import time
from docopt import docopt
from datetime import datetime
import multiprocessing as mp

import Setup
from Helper import Helper
from Node import Node
from Operation import Operation as Op
from Particle import Particle
from Tree import Tree
from Data import Data

# global scope for multiprocessing
helper = None
data = None



def main(argv):
    arguments = docopt(__doc__, version = "PSOSC-Cancer-Evolution 2.0")
    (particles, iterations, matrix, mutation_number, mutation_names, cells,
        alpha, beta, gamma, k, w, c1, c2, max_deletions, max_time, multiple_runs) = Setup.setup_arguments(arguments)

    base_dir = "results" + datetime.now().strftime("%Y%m%d%H%M%S")

    if multiple_runs is None:
        run_dir = base_dir + "/particles%d_run%d" % (particles, iterations)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        data, helper = init(particles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, max_time)
        data.summary(helper, run_dir)
    else:
        runs_data = []
        for r, ptcl in enumerate(multiple_runs):
            print ("\n===== Run number %d =====" % r)
            run_dir = base_dir + "/particles%d_run%d" % (ptcl, r)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            data, helper = init(ptcl, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, max_time)
            data.summary(helper, run_dir)
            runs_data.append(data)
        Data.runs_summary(multiple_runs, runs_data, base_dir)



def init(nparticles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, max_time):
    global helper
    global data
    helper = Helper(matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, max_time)
    data = Data(nparticles, iterations)
    pso(nparticles, iterations)
    data.helper = helper
    return data, helper



def pso(nparticles, iterations):
    global helper
    global data

    print("\n • PARTICLES START-UP  ")
    particles = pso_start_up(nparticles)

    print("\n • PSO RUNNING...")
    print("\tIteration\tBest likelihood so far")
    pso_parallel_execution(particles, iterations)

    print("\n • ADDING BACKMUTATIONS...")
    pso_adding_backmutations()

    print("\n • FINAL RESULTS:        ")
    print("\t- time to complete pso with %d particles: %s seconds" % (data.nofparticles, str(round(data.pso_passed_seconds(), 2))))
    print("\t- best likelihood: %s\n" % str(round(helper.best_particle.best.likelihood, 2)))



def pso_start_up(nparticles):
    global helper
    global data

    data.initialization_start = time.time()

    # Random position, each tree is a binary tree at the beginning
    particles = [Particle(helper.cells, helper.mutation_number, helper.mutation_names, n) for n in range(nparticles)]
    helper.best_particle = particles[0]
    for p in particles:
        p.current_tree.likelihood = Tree.greedy_loglikelihood(helper, p.current_tree)
        if (p.current_tree.likelihood > helper.best_particle.best.likelihood):
            helper.best_particle = p
    data.starting_likelihood = helper.best_particle.best.likelihood

    data.initialization_end = time.time()

    return particles



def pso_parallel_execution(particles, iterations):
    global helper
    global data

    data.pso_start = time.time()

    # creating shared memory between processes
    manager = mp.Manager()
    ns = manager.Namespace()

    # coping data into shared memory
    ns.best_swarm = helper.best_particle.best.copy()
    ns.best_iteration_likelihoods = data.best_iteration_likelihoods
    ns.iteration_times = data.iteration_times
    ns.particle_iteration_times = data.particle_iteration_times
    ns.stop = False
    ns.automatic_stop = False
    if iterations == 0:
        iterations = 10000
        ns.automatic_stop = True

    # creating lock
    lock = manager.Lock()

    # starting particle processes
    for p in particles:
        p.particle_start(iterations, helper, ns, lock)
    for p in particles:
        p.particle_join()

    # copying back data from shared memory
    data.best_iteration_likelihoods = ns.best_iteration_likelihoods
    data.iteration_times = ns.iteration_times
    data.particle_iteration_times = ns.particle_iteration_times
    helper.best_particle.best = ns.best_swarm.copy()



def pso_adding_backmutations():
    global helper
    global data

    helper.best_particle.current_tree = helper.best_particle.best

    iterations_performed = min([len(u) for u in data.particle_iteration_times])
    it = iterations_performed + 1
    end = (iterations_performed + int(iterations_performed / 3)) if (iterations_performed < 300) else (iterations_performed + 100)

    while it < end:
        start_it = time.time()

        data = helper.best_particle.add_back_mutations(it, helper, data)

        lh = helper.best_particle.best.likelihood
        data.best_iteration_likelihoods.append(lh)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

        if it % 10 == 0:
            print("\t   %d\t\t     %s" % (it, str(round(lh, 2))))
        it += 1

    data.set_iterations(end)

    # print("number of iterations per particle: " + str([len(u) for u in data.particle_iteration_times]))

    data.pso_end = time.time()



if __name__ == "__main__":
    main(sys.argv[1:])
