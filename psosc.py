
"""Particle Swarm Optimization Single Cell inference

Usage:
    pso.py (--infile <infile>) [--particles <particles>] [--iterations <iterations>] [--alpha=<alpha>] [--beta=<beta>] [--gamma=<gamma>] [--k=<k>] [--w=<w>] [--c1=<c1>] [--c2=<c2>] [--maxdel=<max_deletions>] [--mutfile <mutfile>] [--multiple <runptcl>...] [--parallel=<parallel>]
    pso.py -h | --help
    pso.py -v | --version

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



def cb_particle_iteration(r):
    i, p, tree_copy, start_time = r
    # doing this for when we parallelize everything
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



def particle_iteration(it, p, helper):

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
    swarm_distance = tree_copy.phylogeny.distance(helper, helper.best_particle.best.phylogeny)
    if swarm_distance != 0:
        clade_to_be_attached = helper.best_particle.best.phylogeny.get_clade_by_distance(helper, swarm_distance, it, helper.c2)
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



def particle_iteration_2(it, p, helper):
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
    swarm_distance = tree_copy.phylogeny.distance(helper, helper.best_particle.best.phylogeny)
    if swarm_distance != 0:
        clade_to_be_attached = helper.best_particle.best.phylogeny.get_clade_by_distance(helper, swarm_distance, it, helper.c2)
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
        temp = tree_copy.copy()
        Op.tree_operation(helper, temp, random.choice(ops))
        temp.likelihood = Tree.greedy_loglikelihood(helper, temp)
        if temp.likelihood > tree_copy.likelihood or random.random() < helper.t:
            tree_copy = temp

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



def single_core_run(helper, data, particles, iterations):
    old_lh = helper.best_particle.best.likelihood
    same_lh = 0
    start_time = data.pso_start

    print("\n2) PSO RUNNING (single-core execution)...        ")
    print("\t\tIteration:\tBest likelihood so far:")
    print("\t\t    /\t\t     %s" % str(round(old_lh, 2)))

    automatic_stop = False
    if iterations == 0:
        iterations = 500
        automatic_stop = True

    for it in range(iterations):

        start_it = time.time()

        for p in particles:
            cb_particle_iteration(particle_iteration(it, p, helper))

        lh = helper.best_particle.best.likelihood
        if lh > old_lh:
            print("\t\t    %d\t\t     %s" % (it, str(round(lh, 2))))
            same_lh = 0
        else:
            print("\t\t    %d\t\t        \"" % it)
            same_lh += 1
        old_lh = lh

        data.best_iteration_likelihoods.append(lh)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

        # it stops if any of the following conditions are met:
        #   - 35 iterations without improvements
        #   - 3 minutes of total execution time
        now_time = time.time()
        if automatic_stop and (same_lh == 35 or (now_time - start_time) > 180):
            # data.set_iterations(it)
            break

    # Adding backmutations
    print("\n3) ADDING BACKMUTATIONS...")
    it += 1
    end = it + data.bm_iterations
    while it < end:
        start_it = time.time()
        cb_backmutations(add_back_mutations(it, helper.best_particle, helper))
        lh = helper.best_particle.best.likelihood
        data.best_iteration_likelihoods.append(lh)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

        if lh > old_lh:
            print("\t\t    %d\t\t     %s" % (it, str(round(lh, 2))))
            same_lh = 0
            same_lh_time = time.time()
        else:
            print("\t\t    %d\t\t        \"" % it)
            same_lh += 1
        old_lh = lh

        it += 1
    data.set_iterations(it)



def parallel_run(helper, data, particles, iterations):

    old_lh = helper.best_particle.best.likelihood
    same_lh = 0
    start_time = data.pso_start

    automatic_stop = False
    if iterations == 0:
        iterations = 500
        automatic_stop = True

    print("\n2) PSO RUNNING (multi-core execution)...")
    print("\t\tIteration:\tBest likelihood so far:")
    print("\t\t    /\t\t     %s" % str(round(old_lh, 2)))

    for it in range(iterations):

        start_it = time.time()
        pool = mp.Pool(mp.cpu_count())
        for p in particles:
            pool.apply_async(particle_iteration, args=(it, p, helper), callback=cb_particle_iteration)
        pool.close()
        pool.join()

        lh = helper.best_particle.best.likelihood
        if lh > old_lh:
            print("\t\t    %d\t\t     %s" % (it, str(round(lh, 2))))
            same_lh = 0
        else:
            print("\t\t    %d\t\t        \"" % it)
            same_lh += 1
        old_lh = lh

        data.best_iteration_likelihoods.append(helper.best_particle.best.likelihood)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

        # it stops if any of the following conditions are met:
        #   - 35 iterations without improvements
        #   - 3 minutes of total execution time
        now_time = time.time()
        if automatic_stop and (same_lh == 35 or (now_time - start_time) > 180):
            # data.set_iterations(it)
            break

    # Adding backmutations
    old_lh = lh
    print("\n3) ADDING BACKMUTATIONS...")
    it += 1
    end = it + data.bm_iterations
    while it < end:
        start_it = time.time()
        cb_backmutations(add_back_mutations(it, helper.best_particle, helper))
        lh = helper.best_particle.best.likelihood
        data.best_iteration_likelihoods.append(lh)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

        if lh > old_lh:
            print("\t\t    %d\t\t     %s" % (it, str(round(lh, 2))))
            same_lh = 0
            same_lh_time = time.time()
        else:
            print("\t\t    %d\t\t        \"" % it)
            same_lh += 1
        old_lh = lh

        it += 1

    data.set_iterations(it)



if __name__ == "__main__":
    main(sys.argv[1:])
