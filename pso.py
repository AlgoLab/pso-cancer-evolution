import multiprocessing as mp
import random
import sys
import time
import math
import copy
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


def init(nparticles, iterations, matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions):
    global helper
    global particles
    global data

    # calculating number of particles based on number of cells and mutations
    if nparticles == 0:
        x = 3*mutation_number+cells
        proc = mp.cpu_count()
        nparticles = - int(x/20) + 14 + proc
        if nparticles < 6:
            nparticles = 6

    helper = Helper(matrix, mutation_number, mutation_names, cells, alpha, beta, gamma, k, w, c1, c2, max_deletions)
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
    lh = Tree.greedy_loglikelihood(helper, particle.current_tree)
    particle.current_tree.likelihood = lh
    return i, particle



def add_back_mutations():
    best_swarm_lh = helper.best_particle.best.likelihood
    tree_copy = helper.best_particle.best.copy()

    op = 0
    result = Op.tree_operation(helper, tree_copy, op)
    lh = Tree.greedy_loglikelihood(helper, tree_copy)


    if lh > best_swarm_lh:
        increased = (best_swarm_lh - lh) / lh * 100
        print("Added backmutation to best particle, found new swarm best: %s, increased by %s%%" % (str(round(lh, 2)), str(round(increased,2))))
        helper.best_particle.current_tree = tree_copy.copy()
        helper.best_particle.best = helper.best_particle.current_tree
        helper.best_particle.best.likelihood = lh



def cb_particle_iteration(r):
    i, result, p, tree_copy, start_time = r
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



def particle_iteration_all(it, p, helper):

    start_time = time.time()
    result = -1
    tree_copy = p.current_tree.copy()

    # self
    ops = [0,1,2,3]
    for i in range(int(helper.w * random.random() * 6)):
        result = Op.tree_operation(helper, tree_copy, random.choice(ops))

    # particle
    current_tree_mutations, current_tree_mutation_number = p.current_tree.phylogeny.mutation_number(helper)
    particle_distance = p.current_tree.phylogeny.my_distance(helper, p.best.phylogeny)
    if particle_distance != 0:
        clade_to_be_attached = p.best.phylogeny.get_clade_by_distance(helper, particle_distance, it, "particle")
        clade_to_be_attached = clade_to_be_attached.copy().detach()
        clade_destination = random.choice(tree_copy.phylogeny.get_clades())
        clade_destination.attach_clade_and_fix(helper, tree_copy, clade_to_be_attached)

    # swarm
    current_tree_mutations, current_tree_mutation_number = p.current_tree.phylogeny.mutation_number(helper)
    swarm_distance = p.current_tree.phylogeny.my_distance(helper, helper.best_particle.best.phylogeny)
    if swarm_distance != 0:
        clade_to_be_attached = helper.best_particle.best.phylogeny.get_clade_by_distance(helper, swarm_distance, it, "swarm")
        clade_to_be_attached = clade_to_be_attached.copy().detach()
        clade_destination = random.choice(tree_copy.phylogeny.get_clades())
        clade_destination.attach_clade_and_fix(helper, tree_copy, clade_to_be_attached)

    tree_copy.phylogeny.fix_for_losses(helper, tree_copy)

    return it, result, p, tree_copy, start_time


def particle_iteration(it, p, helper):

    start_time = time.time()
    result = -1
    tree_copy = p.current_tree.copy()

    # self
    ops = [2,3]
    for i in range(int(helper.w * random.random() * 6)):
        result = Op.tree_operation(helper, tree_copy, random.choice(ops))

    # particle
    current_tree_mutations, current_tree_mutation_number = p.current_tree.phylogeny.mutation_number(helper)
    particle_distance = p.current_tree.phylogeny.my_distance(helper, p.best.phylogeny)
    if particle_distance != 0:
        clade_to_be_attached = p.best.phylogeny.get_clade_by_distance(helper, particle_distance, it, "particle")
        clade_to_be_attached = clade_to_be_attached.copy().detach()
        clade_destination = random.choice(tree_copy.phylogeny.get_clades())
        clade_destination.attach_clade(helper, tree_copy, clade_to_be_attached)

    # swarm
    current_tree_mutations, current_tree_mutation_number = p.current_tree.phylogeny.mutation_number(helper)
    swarm_distance = p.current_tree.phylogeny.my_distance(helper, helper.best_particle.best.phylogeny)
    if swarm_distance != 0:
        clade_to_be_attached = helper.best_particle.best.phylogeny.get_clade_by_distance(helper, swarm_distance, it, "swarm")
        clade_to_be_attached = clade_to_be_attached.copy().detach()
        clade_destination = random.choice(tree_copy.phylogeny.get_clades())
        clade_destination.attach_clade(helper, tree_copy, clade_to_be_attached)

    return it, result, p, tree_copy, start_time




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
    processes = []
    for i, p in enumerate(particles):
        processes.append(pool.apply_async(init_particle, args=(i, p, helper), callback=cb_init_particle))

    pool.close()
    pool.join()

    data.starting_likelihood = helper.best_particle.best.likelihood
    data.initialization_end = time.time()

    data.pso_start = time.time()

    # single_core_run(helper, data, particles, iterations)
    parallel_run(helper, data, particles, iterations)

    data.pso_end = time.time()

    print("\n4) FINAL RESULTS:        ")
    print(" - time to complete pso with %d particles: %s seconds" % (data.nofparticles, str(round(data.pso_passed_seconds(), 2))))
    print(" - best likelihood: %s\n" % str(round(helper.best_particle.best.likelihood, 2)))




def single_core_run(helper, data, particles, iterations):

    initial_w = helper.w
    initial_temp = helper.temperature
    old_lh = helper.best_particle.best.likelihood
    same_lh = 0

    print("\n2) PSO RUNNING...        ")
    print("\t\tIteration:\tBest likelihood so far:")
    print("\t\t    /\t\t     %s" % str(round(old_lh, 2)))

    automatic_stop = False
    if iterations == 0:
        iterations = 200
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

        # helper.w -= initial_w/iterations
        # helper.temperature -= initial_temp/iterations
        # print("temp="+str(helper.temperature))

        data.best_iteration_likelihoods.append(lh)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

        if automatic_stop and (same_lh == 20 or time.time() - data.pso_start > 120):
            # data.set_iterations(it)
            break



def parallel_run(helper, data, particles, iterations):

    # initial_w = helper.w
    # initial_temp = helper.temperature
    old_lh = helper.best_particle.best.likelihood
    same_lh = 0
    same_lh_time = time.time()
    start_time = data.pso_start

    automatic_stop = False
    if iterations == 0:
        iterations = 500
        automatic_stop = True

    print("\n2) PSO RUNNING...        ")
    print("\t\tIteration:\tBest likelihood so far:")
    print("\t\t    /\t\t     %s" % str(round(old_lh, 2)))

    for it in range(iterations):
        start_it = time.time()
        pool = mp.Pool(mp.cpu_count())
        processes = []
        for p in particles:
            processes.append(pool.apply_async(particle_iteration, args=(it, p, helper), callback=cb_particle_iteration))

        lh = helper.best_particle.best.likelihood
        if lh > old_lh:
            print("\t\t    %d\t\t     %s" % (it, str(round(lh, 2))))
            same_lh = 0
            same_lh_time = time.time()
        else:
            print("\t\t    %d\t\t        \"" % it)
            same_lh += 1
        old_lh = lh

        # helper.w -= initial_w/iterations
        # helper.temperature -= (0.5)/iterations


        pool.close()
        pool.join()

        data.best_iteration_likelihoods.append(helper.best_particle.best.likelihood)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

        # it stops if any of the following conditions are met:
        #   - same best likelihood for 25 iterations
        #   - same best likelihood for 45 seconds
        #   - total execution time over 2 minutes
        now_time = time.time()
        if automatic_stop and (same_lh == 25 or (now_time - same_lh_time) > 45 or (now_time - start_time) > 118):
            data.set_iterations(it)
            break

    # Adding backmutations
    print("\n3) ADDING BACKMUTATIONS...")
    i = 0
    while i < 30:
        add_back_mutations()
        i += 1
