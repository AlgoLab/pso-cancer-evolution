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
        decreased = (best_swarm_lh - lh) / lh * 100
        print(mytime() + "!!!!! ADDED BACKMUTATION, new SWARM best, before: %f, now: %f, increased by %f%%" % (best_swarm_lh, lh, decreased))
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
    # print("Operation %d" % (tree_copy.operation.type))

    best_particle_lh = p.best.likelihood
    best_swarm_lh = helper.best_particle.best.likelihood

    if lh > best_particle_lh:
        # updating particle best
        decreased = (best_particle_lh - lh) / lh * 100
        # print(mytime() + "!! %d new particle best, before: %f, now: %f, increased by %f%%" % (p.number, best_particle_lh, lh, decreased))
        data.iteration_new_particle_best[i][p.number] = lh
        p.best = tree_copy
    if lh > best_swarm_lh:
        # updating swarm best
        decreased = (best_swarm_lh - lh) / lh * 100
        print(mytime() + "!!!!! %d new SWARM best, before: %f, now: %f, increased by %f%%" % (p.number, best_swarm_lh, lh, decreased))
        data.iteration_new_best[i][p.number] = lh
        helper.best_particle = p

    data.particle_iteration_times[p.number].append(data._passed_seconds(start_time, time.time()))



def particle_iteration_attachment_plus_operations(it, p, helper):

    start_time = time.time()
    result = -1
    tree_copy = p.current_tree.copy()

    best_swarm_copy = helper.best_particle.best.copy()
    best_particle_copy = p.best.copy()

    current_tree_mutations, current_tree_mutation_number = p.current_tree.phylogeny.mutation_number(helper)
    max_clades = 2

    distance_particle = p.current_tree.phylogeny.distance(helper, best_particle_copy.phylogeny)
    distance_swarm = p.current_tree.phylogeny.distance(helper, best_swarm_copy.phylogeny)

    particle_clade = best_particle_copy.phylogeny.get_clade_distance(helper, max_clades, current_tree_mutation_number, distance_particle)
    swarm_clade = best_swarm_copy.phylogeny.get_clade_distance(helper, max_clades, current_tree_mutation_number, distance_swarm)

    if particle_clade is not None or swarm_clade is not None:
        clade_to_be_attached = None
        if distance_particle == 0 or particle_clade is None: # it is the same tree
            clade_to_be_attached = swarm_clade
        elif distance_swarm == 0 or swarm_clade is None: # it is the same tree
            clade_to_be_attached = particle_clade
        else:
            ran = random.random()

            #riscalo c1 e c2 tra 0 e 1, in modo che c1+c2=1
            #numero random per decidere se prendere dal particle_best o dallo swarm_best
            c1 = helper.c1/(helper.c1+helper.c2)
            c2 = helper.c2/(helper.c1+helper.c2)

            if ran < c1:
                clade_to_be_attached = particle_clade
            else:
                clade_to_be_attached = swarm_clade

        clade_destination = tree_copy.phylogeny.get_clade_distance(helper, max_clades, current_tree_mutation_number, max(distance_particle, distance_swarm), root=True)
        if clade_destination is not None:
            clade_to_be_attached = clade_to_be_attached.copy().detach()
            clade_destination.attach_clade_and_fix(helper, tree_copy, clade_to_be_attached)

    tree_copy.phylogeny.fix_for_losses(helper, tree_copy)
    tree_copy.phylogeny.fix_useless_losses(helper, tree_copy)


    # self movement
    ops = list(range(2, Op.NUMBER))
    temp = tree_copy.copy()
    op = ops[random.randint(0, len(ops) - 1)]
    result = Op.tree_operation(helper, temp, op)
    lh_before = Tree.greedy_loglikelihood(helper, tree_copy)
    lh_after = Tree.greedy_loglikelihood(helper, temp)
    if lh_after > lh_before or random.random() < helper.temperature:
        tree_copy = temp.copy()


    return it, result, p, tree_copy, start_time



def particle_iteration_attachment_plus_operations_2(it, p, helper):

    start_time = time.time()
    result = -1
    tree_copy = p.current_tree.copy()


    rand = random.random()

    if rand <= helper.w:
        # self movement
        ops = list(range(2, Op.NUMBER))
        temp = tree_copy.copy()
        op = ops[random.randint(0, len(ops) - 1)]
        result = Op.tree_operation(helper, temp, op)
        lh_before = Tree.greedy_loglikelihood(helper, tree_copy)
        lh_after = Tree.greedy_loglikelihood(helper, temp)
        if lh_after > lh_before or random.random() < helper.temperature:
            tree_copy = temp.copy()

    else:
        best_swarm_copy = helper.best_particle.best.copy()
        best_particle_copy = p.best.copy()

        current_tree_mutations, current_tree_mutation_number = p.current_tree.phylogeny.mutation_number(helper)
        max_clades = 2

        distance_particle = p.current_tree.phylogeny.distance(helper, best_particle_copy.phylogeny)
        distance_swarm = p.current_tree.phylogeny.distance(helper, best_swarm_copy.phylogeny)

        particle_clade = best_particle_copy.phylogeny.get_clade_distance(helper, max_clades, current_tree_mutation_number, distance_particle)
        swarm_clade = best_swarm_copy.phylogeny.get_clade_distance(helper, max_clades, current_tree_mutation_number, distance_swarm)

        if particle_clade is not None or swarm_clade is not None:
            clade_to_be_attached = None

            if distance_particle == 0 or particle_clade is None: # it is the same tree
                clade_to_be_attached = swarm_clade

            elif distance_swarm == 0 or swarm_clade is None: # it is the same tree
                clade_to_be_attached = particle_clade

            elif rand <= helper.c1:
                # movement to best particle tree
                clade_to_be_attached = particle_clade

            else:
                #movement to best swarm tree
                clade_to_be_attached = swarm_clade

            clade_destination = tree_copy.phylogeny.get_clade_distance(helper, max_clades, current_tree_mutation_number, max(distance_particle, distance_swarm), root=True)
            if clade_destination is not None:
                clade_to_be_attached = clade_to_be_attached.copy().detach()
                clade_destination.attach_clade_and_fix(helper, tree_copy, clade_to_be_attached)

        tree_copy.phylogeny.fix_for_losses(helper, tree_copy)
        tree_copy.phylogeny.fix_useless_losses(helper, tree_copy)


    return it, result, p, tree_copy, start_time








def mytime():
    return datetime.now().strftime("[%Y/%m/%d, %H:%M:%S] - ")



def pso(nparticles, iterations, matrix):
    global particles
    global helper
    global data

    print(mytime() + "Particle initialization...")

    # Random position, each tree is a binary tree at the beginning
    particles = [Particle(helper.cells, helper.mutation_number, helper.mutation_names, n) for n in range(nparticles)]

    helper.best_particle = particles[0]
    pool = mp.Pool(mp.cpu_count())

    data.initialization_start = time.time()

    # parallelizing tree initialization
    print("Starting trees:")
    processes = []
    for i, p in enumerate(particles):
        processes.append(pool.apply_async(init_particle, args=(i, p, helper), callback=cb_init_particle))
        print("   " + str(p.number) + ") " + str(Tree.greedy_loglikelihood(helper, p.current_tree)))
    print("")

    pool.close()
    pool.join()

    data.starting_likelihood = helper.best_particle.best.likelihood
    data.initialization_end = time.time()

    data.pso_start = time.time()

    single_core_run(helper, data, particles, iterations)
    # parallel_run(helper, data, particles, iterations)

    data.pso_end = time.time()




def single_core_run(helper, data, particles, iterations):

    initial_w = helper.w
    initial_temp = helper.temperature

    for it in range(iterations):
        start_it = time.time()

        print("\n--------- Iteration [%d] ---------" % it)
        for p in particles:
            cb_particle_iteration(particle_iteration_attachment_plus_operations_2(it, p, helper))

        helper.w -= initial_w/iterations
        helper.temperature -= initial_temp/iterations
        # print("temp="+str(helper.temperature))

        data.best_iteration_likelihoods.append(helper.best_particle.best.likelihood)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))



def parallel_run(helper, data, particles, iterations):

    initial_w = helper.w
    initial_temp = helper.temperature

    for it in range(iterations):
        print("\n------->  Iteration (%d)  <-------" % it)

        start_it = time.time()

        pool = mp.Pool(mp.cpu_count())
        processes = []
        for p in particles:
            processes.append(pool.apply_async(particle_iteration_attachment_plus_operations_2, args=(it, p, helper), callback=cb_particle_iteration))

        helper.w -= initial_w/iterations
        helper.temperature -= (initial_temp)/iterations

        pool.close()
        pool.join()

        data.best_iteration_likelihoods.append(helper.best_particle.best.likelihood)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))

    # Adding backmutations
    i = 0
    while i < 30:
        add_back_mutations()
        i += 1
