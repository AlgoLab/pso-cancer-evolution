
"""Particle Swarm Optimization Single Cell inference

Usage:
    prova.py (--infile <infile>) [--particles <particles>] [--iterations <iterations>] [--alpha=<alpha>] [--beta=<beta>] [--gamma=<gamma>] [--k=<k>] [--w=<w>] [--c1=<c1>] [--c2=<c2>] [--maxdel=<max_deletions>] [--mutfile <mutfile>] [--multiple <runptcl>...] [--parallel=<parallel>]
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
    --parallel=<parallel>                   Multi-core execution [default: True].
"""

import multiprocessing as mp
import random
import os
import sys
import time
import copy
import numpy as np
from docopt import docopt
from datetime import datetime

import Setup
from Helper import Helper
from Node import Node
from Operation import Operation as Op
from Particle import Particle
from Tree import Tree
from datetime import datetime
from Data import Data

# global scope for multiprocessing
particles = []
helper = None
data = None


def main(argv):
    arguments = docopt(__doc__, version = "PSOSC-Cancer-Evolution 2.0")
    (particles, iterations, matrix, mutation_number, mutation_names, cells,
        alpha, beta, gamma, k, w, c1, c2, max_deletions, parallel, multiple_runs) = Setup.setup_arguments(arguments)

    base_dir = "results" + datetime.now().strftime("%Y%m%d%H%M%S")

    if multiple_runs is None:
        run_dir = base_dir + "/p%d_i%d" % (particles, iterations)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        data, helper = init(particles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, parallel)
        data.summary(helper, run_dir)
    else:
        runs_data = []
        for r, ptcl in enumerate(multiple_runs):
            print ("\n=== Run number %d ===" % r)
            run_dir = base_dir + "/p%d_i%d" % (ptcl, iterations)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            data, helper = init(ptcl, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, parallel)
            data.summary(helper, run_dir)
            runs_data.append(data)
        Data.runs_summary(multiple_runs, runs_data, base_dir)




def init(nparticles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, parallel):
    global helper
    global particles
    global data

    # calculating number of particles based on number of cells and mutations
    if nparticles == 0:
        x = 3 * mutation_number + cells
        proc = mp.cpu_count()
        nparticles = - int(x / 20) + 14 + proc
        if nparticles < proc:
            nparticles = proc

    helper = Helper(matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions, parallel)
    data = Data(nparticles, iterations)
    pso(nparticles, iterations, matrix)
    data.helper = helper
    return data, helper



def cb_init_particle(result):
    i, particle = result
    particles[i] = particle
    if (particle.current_tree.likelihood > helper.best_particle.best.likelihood):
        helper.best_particle = particle

def init_particle(i, particle, helper):
    particle.current_tree.likelihood = Tree.greedy_loglikelihood(helper, particle.current_tree)
    return i, particle



def add_back_mutations(it, p, helper):
    start_time = time.time()
    op = 0
    tree_copy = p.best.copy()
    Op.tree_operation(helper, tree_copy, op)
    return it, p, tree_copy, start_time

def cb_backmutations(r):
    i, p, tree_copy, start_time = r
    particles[p.number] = p

    # updating log likelihood and bests
    lh = Tree.greedy_loglikelihood(helper, tree_copy)
    tree_copy.likelihood = lh
    p.current_tree = tree_copy

    best_particle_lh = p.best.likelihood
    best_swarm_lh = helper.best_particle.best.likelihood

    if lh > best_particle_lh:
        data.iteration_new_particle_best[i][p.number] = lh
        p.best = tree_copy

    if lh > best_swarm_lh:
        data.iteration_new_best[i][p.number] = lh
        helper.best_particle = p

    data.particle_iteration_times[p.number].append(data._passed_seconds(start_time, time.time()))
    for part in particles:
        data.particle_iteration_times[part.number].append(0)




def run_particle(iterations, p, helper, ns, lock):

    for it in range(iterations):
        start_it = time.time()
        cb_particle_iteration(particle_iteration(it, p, helper, ns.best_swarm.copy()), ns, lock)

        if p.number == 0:

            # ---- critical section
            lock.acquire()
            ns.best_iteration_likelihoods = append_to_shared_array(ns.best_iteration_likelihoods, ns.best_swarm.likelihood)
            ns.iteration_times = append_to_shared_array(ns.iteration_times, time.time() - start_it)
            # ---- end of critical section
            lock.release()

            if it % 10 == 0:
                print("\t\t    %d\t\t     %s" % (it, str(round(ns.best_swarm.likelihood, 2))))



def append_to_shared_array(arr, v):
    arr.append(v)
    return arr

def update_shared_matrix(matrix, i, j, new_value):
    matrix[i][j] = new_value
    return matrix



def cb_particle_iteration(r, ns, lock):

    i, p, tree_copy, start_time = r
    particles[p.number] = p

    # updating log likelihood and bests
    lh = Tree.greedy_loglikelihood(helper, tree_copy)

    tree_copy.likelihood = lh
    p.current_tree = tree_copy

    # update particle best
    best_particle_lh = p.best.likelihood
    if lh > best_particle_lh:
        ns.iteration_new_particle_best = update_shared_matrix(ns.iteration_new_particle_best, i, p.number, lh)
        p.best = tree_copy

    # ---- critical section
    lock.acquire()

    # update swarm best
    best_swarm_lh = ns.best_swarm.likelihood
    if lh > best_swarm_lh:
        ns.iteration_new_best = update_shared_matrix(ns.iteration_new_best, i, p.number, lh)
        ns.best_swarm = tree_copy

    # update particle iteration times
    tmp = ns.particle_iteration_times
    tmp[p.number].append(time.time() - start_time)
    ns.particle_iteration_times = tmp

    # ---- end of critical section
    lock.release()


def particle_iteration(it, p, helper, best_swarm):

    start_time = time.time()
    tree_copy = p.current_tree.copy()

    # movement to particle best
    particle_distance = tree_copy.phylogeny.distance(helper, p.best.phylogeny)
    if particle_distance != 0:
        clade_to_be_attached = p.best.phylogeny.get_clade_by_distance(helper, particle_distance, it, helper.c1)
        clade_to_be_attached = clade_to_be_attached.copy().detach()
        clade_destination = random.choice(tree_copy.phylogeny.get_clades())
        clade_destination.attach_clade(helper, tree_copy, clade_to_be_attached)

    # movement to swarm best
    swarm_distance = tree_copy.phylogeny.distance(helper, best_swarm.phylogeny)
    if swarm_distance != 0:
        clade_to_be_attached = best_swarm.phylogeny.get_clade_by_distance(helper, swarm_distance, it, helper.c2)
        clade_to_be_attached = clade_to_be_attached.copy().detach()
        clade_destination = random.choice(tree_copy.phylogeny.get_clades())
        clade_destination.attach_clade(helper, tree_copy, clade_to_be_attached)

    # self movement
    ops = [2,3]
    n_op = int(helper.w * random.random() * 2)
    if n_op < 1:
        n_op = 1
    elif n_op > 3:
        n_op = 3
    for i in range(n_op):
        Op.tree_operation(helper, tree_copy, random.choice(ops))

    return it, p, tree_copy, start_time



def pso(nparticles, iterations, matrix):
    global particles
    global helper
    global data

    print("\n1) PARTICLES INIZIALIZATION...  ")

    # Random position, each tree is a binary tree at the beginning
    particles = [Particle(helper.cells, helper.mutation_number, helper.mutation_names, n) for n in range(nparticles)]

    helper.best_particle = particles[0]
    pool = mp.Pool(mp.cpu_count())

    data.initialization_start = time.time()

    # parallelizing tree initialization
    for i, p in enumerate(particles):
        pool.apply_async(init_particle, args=(i, p, helper), callback=cb_init_particle)

    pool.close()
    pool.join()

    data.starting_likelihood = helper.best_particle.best.likelihood
    data.initialization_end = time.time()

    data.pso_start = time.time()

    if helper.parallel:
        parallel_run(helper, data, particles, iterations)
    else:
        single_core_run(helper, data, particles, iterations)

    data.pso_end = time.time()

    print("\n4) FINAL RESULTS:        ")
    print(" - time to complete pso with %d particles: %s seconds" % (data.nofparticles, str(round(data.pso_passed_seconds(), 2))))
    print(" - best likelihood: %s\n" % str(round(helper.best_particle.best.likelihood, 2)))



def parallel_run(helper, data, particles, iterations):

    # creating shared memory between processes
    manager = mp.Manager()
    ns = manager.Namespace()

    # coping data into shared memory
    ns.best_swarm = helper.best_particle.best.copy()
    ns.best_iteration_likelihoods = data.best_iteration_likelihoods
    ns.iteration_times = data.iteration_times
    ns.iteration_new_particle_best = data.iteration_new_particle_best
    ns.iteration_new_best = data.iteration_new_best
    ns.particle_iteration_times = data.particle_iteration_times

    # creating lock
    lock = manager.Lock()

    # printing info for displaying the following results
    print("\n2) PSO RUNNING (multi-core execution)...")
    print("\t\tIteration:\tBest likelihood so far:")

    # starting particle processes
    processes = []
    for p in particles:
        processes.append(mp.Process(target = run_particle, args = (iterations, p, helper, ns, lock)))
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    # copying back data from shared memory
    data.best_iteration_likelihoods = ns.best_iteration_likelihoods
    data.iteration_times = ns.iteration_times
    data.iteration_new_particle_best = ns.iteration_new_particle_best
    data.iteration_new_best = ns.iteration_new_best
    data.particle_iteration_times = ns.particle_iteration_times
    helper.best_particle.best = ns.best_swarm.copy()

if __name__ == "__main__":
    main(sys.argv[1:])
