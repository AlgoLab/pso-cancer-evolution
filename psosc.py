"""
Particle Swarm Optimization Single Cell inference

Usage:
    psosc.py (--infile <infile>) [--particles <particles>] [--cores <cores>] [--iterations <iterations>] [--alpha=<alpha>] [--beta=<beta>] [--gamma=<gamma>] [--k=<k>] [--maxdel=<max_deletions>] [--mutfile <mutfile>] [--tolerance=<tolerance>] [--maxtime=<maxtime>] [--multiple <runptcl>...] [--truematrix=<truematrix>] [--silent] [--output=<output>]
    psosc.py -h | --help
    psosc.py -v | --version

Options:
    -h --help                                   Shows this screen.
    -v --version                                Shows version.
    -i infile | --infile infile                 Matrix input file.
    -m mutfile | --mutfile mutfile              Path of the mutation names. If not used, then the mutations will be named progressively from 1 to mutations.
    -p particles | --particles particles        Number of particles to use for PSO. If not used or zero, it'll be number of CPU cores [default: 0]
    -c cores | --cores cores                    Number of CPU cores used for the execution. If not used or zero, it'll be half of this computer's CPU cores [default: 0]
    -t iterations | --iterations iterations     Number of iterations. If not used or zero, PSO will stop when stuck on a best fitness value (or after maxtime of total execution) [default: 0].
    --alpha=<alpha>                             False negative rate [default: 0.15].
    --beta=<beta>                               False positive rate [default: 0.00001].
    --gamma=<gamma>                             Loss rate for each mutation (single float for every mutations or file with different rates) [default: 1].
    --k=<k>                                     K value of Dollo(k) model used as phylogeny tree [default: 3].
    --maxdel=<max_deletions>                    Maximum number of total deletions allowed [default: 5].
    --tolerance=<tolerance>                     Minimum relative improvement (between 0 and 1) in the last 500 iterations in order to keep going, if iterations are zero [default: 0.005].
    --maxtime=<maxtime>                         Maximum time (in seconds) of total PSOSC execution [default: 1200].
    --truematrix=<truematrix>                   Actual correct matrix, for algorithm testing [default: 0].
    --silent                                    Doesn't print anything
    --output=<output>                           Limit the output (files created) to: (image | plot | text_file | all) [default: all]

"""


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


def main(argv):
    arguments = docopt(__doc__, version = "PSOSC-Cancer-Evolution 2.0")
    helper = Helper(arguments)

    base_dir = "results" + datetime.now().strftime("%Y%m%d%H%M%S")
    if helper.multiple_runs is None:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        pso(helper).summary(helper, base_dir)
    else:
        runs_data = []
        for r, particles in enumerate(helper.multiple_runs):
            print ("\n\n===== Run number %d =====" % r)
            run_dir = base_dir + "/particles%d_run%d" % (particles, r)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            data = pso(helper, particles)
            data.summary(helper, run_dir)
            runs_data.append(data)
            helper.avg_dist = 0
        Data.runs_summary(helper.multiple_runs, runs_data, base_dir)


def pso(helper, nparticles=None):

    if not helper.silent:
        print("\n • PARTICLES START-UP")

    Tree.set_probabilities(helper.alpha, helper.beta)

    if nparticles != None:
        helper.nparticles = nparticles
    data = Data(helper.filename, helper.nparticles, helper.output)
    data.pso_start = time.time()

    # create particles
    particles = [Particle(helper.cells, helper.mutation_number, helper.mutation_names, n, helper.silent) for n in range(helper.nparticles)]
    best = particles[0].current_tree
    best.likelihood = float("-inf")
    for p in particles:
        p.current_tree.likelihood = Tree.greedy_loglikelihood(p.current_tree, helper.matrix, helper.cells, helper.mutation_number)
        p.best.likelihood = p.current_tree.likelihood
        if (p.current_tree.likelihood > best.likelihood):
            best = p.current_tree
            data.starting_likelihood = best.likelihood
    if helper.truematrix != 0:
        data.starting_likelihood_true = Tree.greedy_loglikelihood(best, helper.truematrix, helper.cells, helper.mutation_number)

    # creating shared memory between processes
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    ns = manager.Namespace()

    # coping data into shared memory
    ns.best_swarm = best.copy()
    ns.best_iteration_likelihoods = []
    ns.iterations_performed = data.iterations_performed
    ns.stop = False
    ns.operations = [2,3]
    ns.attach = True
    ns.avg_dist = 0

    # selecting particles to assign to processes
    assigned_particles = []
    for i in range(helper.cores):
        assigned_particles.append([])
    for i in range(helper.nparticles):
        assigned_particles[i%helper.cores].append(particles[i])

    if not helper.silent:
        print("\n • PSO RUNNING...")
        print("\t  Time\t\t Best likelihood so far")

    # creating and starting processes
    processes = []
    for i in range(helper.cores):
        processes.append(multiprocessing.Process(target = start_threads, args = (assigned_particles[i], helper, ns, lock)))
        processes[i].start()
    for proc in processes:
        proc.join()

    # copying back data from shared memory
    data.best_iteration_likelihoods = ns.best_iteration_likelihoods
    data.iterations_performed = ns.iterations_performed
    data.best = ns.best_swarm.copy()

    data.pso_end = time.time()

    if not helper.silent:
        print("\n • FINAL RESULTS")
        print("\t- time to complete pso with %d particles: %s seconds" % (data.nofparticles, str(round(data.get_total_time(), 2))))
        print("\t- best likelihood: %s\n" % str(round(data.best.likelihood, 2)))

    return data


def start_threads(assigned_particles, helper, ns, lock):
    for p in assigned_particles:
        p.particle_start(helper, ns, lock)
    for p in assigned_particles:
        p.particle_join()


if __name__ == "__main__":
    main(sys.argv[1:])
