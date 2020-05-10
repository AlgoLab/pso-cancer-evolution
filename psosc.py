"""
Particle Swarm Optimization Single Cell inference

Usage:
    psosc.py (-i infile) (-p particles) (-c cores) (-k k) (-a alpha) (-b beta)
        [-g gamma] [-t iterations] [-d max_deletions] [-e mutfile] [-T tolerance] [-m maxtime] [-I truematrix] [--quiet] [--output output]
    psosc.py -h | --help
    psosc.py -v | --version

Options:
    -i infile                       Matrix input file.
    -p particles                    Number of particles to use for PSO (single or multiple values, separated by commas, for a multiple run).
    -c cores                        Number of CPU cores used for the execution.
    -k k                            K value of Dollo(k) model used as phylogeny tree.
    -a alpha                        False negative rate in input file or path of the file containing different FN rates for each mutations.
    -b beta                         False positive rate.
    -g gamma                        Loss rate in input file or path of the file containing different GAMMA rates for each mutations [default: 1].
    -t iterations                   Number of iterations (-m argument will be ignored; not used by default).
    -d max_deletions                Maximum number of total deletions allowed [default: +inf].
    -e mutfile                      Path of the mutation names. If not used, mutations will be named progressively from 1 to mutations.
    -T tolerance                    Tolerance, minimum relative improvement (between 0 and 1) in the last iterations in order to keep going, if iterations are not used [default: 0.005].
    -m maxtime                      Maximum time (in seconds) of total PSOSC execution (not used by default).
    -I truematrix                   Actual correct matrix, for algorithm testing (not used by default).
    --quiet                         Doesn't print anything (not used by default).
    --output output                 Limit the output (files created) to: (image | plot | text_file | all) [default: all].

"""


from Helper import Helper
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
    base_dir = "results" + datetime.now().strftime("%Y%m%d%H%M%S")
    helper = Helper(arguments)

    if helper.multiple_runs:
        runs_data = []
        for r, n_particles in enumerate(helper.n_particles):
            print ("\n\n======= Run number %d =======" % (r+1))
            run_dir = base_dir + "/particles%d_run%d" % (n_particles, (r+1))
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            data = pso(helper, n_particles)
            data.summary(helper, run_dir)
            runs_data.append(data)
        Data.runs_summary(helper.n_particles, runs_data, base_dir)
    else:
        data = pso(helper)
        data.summary(helper, base_dir)


def pso(helper, n_particles=None):
    if not helper.quiet:
        print("\n • PARTICLES START-UP")

    Tree.set_probabilities(helper.alpha, helper.beta)

    if n_particles == None:
         n_particles = helper.n_particles
     # print(n_particles)
    data = Data(helper.filename, n_particles, helper.output)
    data.pso_start = time.time()

    # create particles
    max_stall_iterations = 500-int(0.75*n_particles)
    if helper.max_deletions == 0:
        max_stall_iterations = int(max_stall_iterations * 1.1)
    particles = [Particle(helper.cells, helper.mutation_number, helper.mutation_names, n, max_stall_iterations, helper.quiet) for n in range(n_particles)]
    best = particles[0].current_tree
    best.likelihood = float("-inf")
    for p in particles:
        p.current_tree.likelihood = Tree.greedy_loglikelihood(p.current_tree, helper.matrix, helper.cells, helper.mutation_number)
        p.best.likelihood = p.current_tree.likelihood
        if (p.current_tree.likelihood > best.likelihood):
            best = p.current_tree
    data.starting_likelihood = best.likelihood
    if helper.truematrix != None:
        data.starting_likelihood_true = Tree.greedy_loglikelihood(best, helper.truematrix, helper.cells, helper.mutation_number)

    # creating shared memory between processes
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    ns = manager.Namespace()

    # coping data into shared memory
    ns.best_swarm = best.copy()
    ns.swarm_best_likelihoods = []
    ns.particles_best_likelihoods = [[] for p in particles]
    ns.iterations_performed = data.iterations_performed
    ns.stop = False
    ns.operations = [2,3]
    ns.attach = True
    ns.max_dist = 0.1

    # selecting particles to assign to processes
    assigned_particles = [[] for i in range(helper.cores)]
    for i in range(n_particles):
        assigned_particles[i%helper.cores].append(particles[i])

    if not helper.quiet:
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
    data.swarm_best_likelihoods = ns.swarm_best_likelihoods
    data.particles_best_likelihoods = ns.particles_best_likelihoods
    data.iterations_performed = ns.iterations_performed
    data.best = ns.best_swarm.copy()

    data.pso_end = time.time()

    if not helper.quiet:
        print("\n • FINAL RESULTS")
        print("\t- time to complete pso with %d particles: %s seconds" % (data.n_particles, str(round(data.get_total_time(), 2))))
        print("\t- best likelihood: %s\n" % str(round(data.best.likelihood, 2)))

    return data


def start_threads(assigned_particles, helper, ns, lock):
    for p in assigned_particles:
        p.particle_start(helper, ns, lock)
    for p in assigned_particles:
        p.particle_join()


if __name__ == "__main__":
    main(sys.argv[1:])
