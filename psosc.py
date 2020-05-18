"""
Particle Swarm Optimization Single Cell inference

Usage:
    psosc.py (-i infile) (-c cores) (-k k) (-a alpha) (-b beta)
         [-p particles] [-g gamma] [-t iterations] [-d max_deletions] [-e mutfile] [-T tolerance] [-m maxtime] [-I truematrix] [--quiet] [--output output]
    psosc.py --help
    psosc.py --version

Options:
    -i infile               Matrix input file
    -c cores                Number of CPU cores
    -k k                    K value of Dollo(k) model used as phylogeny tree
    -a alpha                False negative rate in input file or path of the file containing different FN rates for each mutations
    -b beta                 False positive rate

    -p particles            Number of particles (single or multiple values, separated by commas, for a multiple run); by default it is calculated proportionally to the size of the matrix
    -g gamma                Loss rate in input file or path of the file containing different GAMMA rates for each mutations [default: 1]
    -t iterations           Number of iterations (-m argument will be ignored; not used by default)
    -d max_deletions        Maximum number of total deletions allowed [default: +inf]
    -e mutfile              Path of the mutation names. If not used, mutations will be named progressively from 1 to mutations (not used by default)
    -T tolerance            Tolerance, minimum relative improvement (between 0 and 1) in the last iterations in order to keep going, if iterations are not used [default: 0.005]
    -m maxtime              Maximum time (in seconds) of total PSOSC execution (not used by default)
    -I truematrix           Actual correct matrix, for algorithm testing (not used by default)
    --quiet                 Doesn't print anything (not used by default)
    --output output         Limit the output (files created) to: (image | plots | text_file | all) [default: all]

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
import threading
import psutil


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
    # assigning process to cores
    selected_cores = get_least_used_cores(helper.cores)
    assign_to_cores(os.getpid(), selected_cores)

    if n_particles == None:
         n_particles = helper.n_particles

    if not helper.quiet:
        print("\n • %d PARTICLES START-UP" % (n_particles))

    Tree.set_probabilities(helper.alpha, helper.beta)

    data = Data(helper.filename, n_particles, helper.output)
    data.pso_start = time.time()

    # creating shared memory between processes
    manager = multiprocessing.Manager()
    assign_to_cores(manager._process.ident, selected_cores)
    lock = manager.Lock()
    ns = manager.Namespace()

    # selecting particles to assign to processes
    assigned_numbers = [[] for i in range(helper.cores)]
    for i in range(n_particles):
        assigned_numbers[i%(helper.cores)].append(i)

    # coping data into shared memory
    ns.best_swarm = None
    ns.swarm_best_likelihoods = []
    ns.particle_best_likelihoods = [[] for x in range(helper.n_particles)]
    ns.iterations_performed = data.iterations_performed
    ns.stop = False
    ns.operations = [2,3]
    ns.attach = True

    if not helper.quiet:
        print("\n • PSO RUNNING...")
        print("\t  Time\t\t Best likelihood so far")

    # creating and starting processes
    processes = []
    for i in range(helper.cores):
        processes.append(multiprocessing.Process(target = start_threads, args = (selected_cores, assigned_numbers[i], data, helper, ns, lock)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # copying back data from shared memory
    data.swarm_best_likelihoods = ns.swarm_best_likelihoods
    data.particle_best_likelihoods = ns.particle_best_likelihoods
    data.iterations_performed = ns.iterations_performed
    data.best = ns.best_swarm.copy()
    data.pso_end = time.time()

    if not helper.quiet:
        print("\n • FINAL RESULTS")
        print("\t- time to complete pso with %d particles: %s seconds" % (data.n_particles, str(round(data.get_total_time(), 2))))
        print("\t- best likelihood: %s\n" % str(round(data.best.likelihood, 2)))

    return data


def start_threads(selected_cores, assigned_numbers, data, helper, ns, lock):
    assign_to_cores(os.getpid(), selected_cores)
    particles = []
    for i in assigned_numbers:
        p = Particle(helper.cells, helper.mutation_number, helper.mutation_names, i)
        p.current_tree.likelihood = Tree.greedy_loglikelihood(p.current_tree, helper.matrix, helper.cells, helper.mutation_number)
        if ns.best_swarm is None:
            ns.best_swarm = p.current_tree.copy()
        p.thread = threading.Thread(target = p.run_iterations, args = (helper, ns, lock))
        particles.append(p)
    for p in particles:
        p.thread.start()
    for p in particles:
        p.thread.join()


def get_least_used_cores(n_cores):
    cpu_usage = psutil.cpu_percent(percpu=True)
    cores = []
    for i in range(n_cores):
        c = cpu_usage.index(min(cpu_usage))
        cores.append(c)
        cpu_usage[c] = float("+inf")
    return cores


def assign_to_cores(pid, cores):
    proc = psutil.Process(pid)
    proc.cpu_affinity(cores)


if __name__ == "__main__":
    main(sys.argv[1:])
