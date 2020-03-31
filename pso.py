import multiprocessing as mp
import random
import sys
import time
import numpy as np
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
    result = Op.tree_operation(helper, tree_copy, op)

    # lh = Tree.greedy_loglikelihood(helper, tree_copy)
    # best_swarm_lh = helper.best_particle.best.likelihood
    #
    # if lh > best_swarm_lh:
    #     helper.best_particle.current_tree = tree_copy.copy()
    #     helper.best_particle.best = helper.best_particle.current_tree
    #     helper.best_particle.best.likelihood = lh
    #     print("Added backmutation, new swarm best: %s" % str(round(lh, 2)))

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

    # self movement
    ops = [2,3]
    n_op = int(helper.w * random.random() * 2)
    if n_op < 1:
        n_op = 1
    elif n_op > 3:
        n_op = 3
    for i in range(n_op):
        result = Op.tree_operation(helper, tree_copy, random.choice(ops))

    # movement to particle best
    particle_distance = p.current_tree.phylogeny.distance(helper, p.best.phylogeny)
    if particle_distance != 0:
        clade_to_be_attached = p.best.phylogeny.get_clade_by_distance(helper, particle_distance, it, helper.c1)
        clade_to_be_attached = clade_to_be_attached.copy().detach()
        clade_destination = random.choice(tree_copy.phylogeny.get_clades())
        clade_destination.attach_clade(helper, tree_copy, clade_to_be_attached)

    # movement to swarm best
    swarm_distance = p.current_tree.phylogeny.distance(helper, helper.best_particle.best.phylogeny)
    if swarm_distance != 0:
        clade_to_be_attached = helper.best_particle.best.phylogeny.get_clade_by_distance(helper, swarm_distance, it, helper.c2)
        clade_to_be_attached = clade_to_be_attached.copy().detach()
        clade_destination = random.choice(tree_copy.phylogeny.get_clades())
        clade_destination.attach_clade(helper, tree_copy, clade_to_be_attached)

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
    same_lh_time = time.time()
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
            same_lh_time = time.time()
        else:
            print("\t\t    %d\t\t        \"" % it)
            same_lh += 1
        old_lh = lh

        data.best_iteration_likelihoods.append(lh)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

        # it stops if any of the following conditions are met:
        #   - 25 iterations without improvements
        #   - 40 seconds without improvements
        #   - 3 minutes of total execution time
        now_time = time.time()
        if automatic_stop and (same_lh == 25 or (now_time - same_lh_time) > 40 or (now_time - start_time) > 180):
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
    same_lh_time = time.time()
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

        lh = helper.best_particle.best.likelihood
        if lh > old_lh:
            print("\t\t    %d\t\t     %s" % (it, str(round(lh, 2))))
            same_lh = 0
            same_lh_time = time.time()
        else:
            print("\t\t    %d\t\t        \"" % it)
            same_lh += 1
        old_lh = lh

        pool.close()
        pool.join()

        data.best_iteration_likelihoods.append(helper.best_particle.best.likelihood)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

        # it stops if any of the following conditions are met:
        #   - 25 iterations without improvements
        #   - 40 seconds without improvements
        #   - 3 minutes of total execution time
        now_time = time.time()
        if automatic_stop and (same_lh == 25 or (now_time - same_lh_time) > 40 or (now_time - start_time) > 180):
            # data.set_iterations(it)
            break

    # Adding backmutations
    old_lh = lh
    print("Best tree found with pso: %s" % str(round(lh, 2)))
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
