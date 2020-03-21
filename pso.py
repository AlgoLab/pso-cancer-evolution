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
        print(mytime() + "!! %d new particle best, before: %f, now: %f, increased by %f%%" % (p.number, best_particle_lh, lh, decreased))
        data.iteration_new_particle_best[i][p.number] = lh
        p.best = tree_copy
    if lh > best_swarm_lh:
        # updating swarm best
        decreased = (best_swarm_lh - lh) / lh * 100
        print(mytime() + "!!!!! %d new SWARM best, before: %f, now: %f, increased by %f%%" % (p.number, best_swarm_lh, lh, decreased))
        data.iteration_new_best[i][p.number] = lh
        helper.best_particle = p

    data.particle_iteration_times[p.number].append(data._passed_seconds(start_time, time.time()))


def particle_iteration_original(it, p, helper):

    """
        we're going to "mix" three trees:
        - the current tree
            (the current solution that has to be modified)
        - the best swarm tree
            (this guides us to the best solution we know of)
        - the best particle tree
            (this slows down the way we get to the best solution, in order
            to avoid getting stuck in a local optimum)

        We "move" by copying clades from a tree into another at a given height.

        We calculate the distance between:
        current_tree - best_swarm_tree
        current_tree - best_particle_tree

        Given the formula for the distance between phylogenies:
        d(T1, T2) = max ( sum_{x € T1}(m(x)), sum_{x € T2}(m(x)) ) - max_weight_matching(x)

        The more distant we are from an optimum, the faster we have to move.
        The less distant we are from an optimum, the slower we have to move.

        How fast we move defines how high the clades we copy are, so we define it
        as the velocity of our particle. The velocity is directly proportional to
        how distant we are from the other tree.
        v = d(T1, T2)

        Also, we will randomly pick the number "n" of clades, n € [1, 3],
        that we will pick from each tree.
    """

    start_time = time.time()
    ops = list(range(0, Op.NUMBER))
    result = -1
    op = ops.pop(random.randint(0, len(ops) - 1))

    tree_copy = p.current_tree.copy()
    # Higher probability of steep when no new best
    if it > 0 and data.iteration_new_particle_best[it - 1][p.number] == 0:
        p.climb_probability += 0.2

    if random.random() < math.log(p.climb_probability):
        print(mytime() + "/// %d Done operation" % p.number)
        p.climb_probability = 1.0
        result = Op.tree_operation(helper, tree_copy, op)
    else:
        print(mytime() + "/// %d Trying clade attachment" % p.number)

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

    return it, result, p, tree_copy, start_time


def particle_iteration_2(it, p, helper):

    start_time = time.time()
    tree_copy = p.current_tree.copy()
    result = -1
    ops = list(range(2, Op.NUMBER))
    n_operations = 10

    if it > 0 :

        distance_particle = tree_copy.phylogeny.distanza(helper, p.best)
        distance_swarm = tree_copy.phylogeny.distanza(helper, helper.best_particle.best)

        self_movement = (helper.w * p.velocity) / 3
        particle_movement = (helper.c1 * distance_particle) * 2
        swarm_movement = (helper.c2 * distance_swarm) * 5
        # print("self_movement: " + str(self_movement))
        # print("particle_movement: " + str(particle_movement))
        # print("swarm_movement: " + str(swarm_movement))

        p.velocity = self_movement + particle_movement + swarm_movement


        #self movement
        n_operations = int(round(n_operations * self_movement))
        print("self operations: "+str(n_operations))
        for i in range(n_operations):
            op = ops[random.randint(0, len(ops) - 1)]
            result = Op.tree_operation(helper, tree_copy, op)
            # if result == 0:
            #     print(mytime() + "/// Particle: " + str(p.number) + ", done operation: " + str(op))
            # else:
            #     print("nope: " + str(result))


        #movement to best particle
        n_operations = round(n_operations * particle_movement)
        print("particle operations: "+str(n_operations))

        valid = False
        offset = -0.1
        while not(valid) and n_operations > 0:
            temp = tree_copy.copy()

            for i in range(n_operations):
                op = ops[random.randint(0, len(ops) - 1)]
                result = Op.tree_operation(helper, temp, op)

            new_distance = temp.phylogeny.distanza(helper, p.best)
            # print("PARTICLE. old=" + str(distance_particle) + ", new=" + str(new_distance))
            if new_distance < distance_particle+offset:
                # print(mytime() + "/// Movement to best particle: " + str(p.number) + ", done some operations: ")
                tree_copy = temp.copy()
                valid = True
            else:
                offset += 0.01


        #movement to best swarm
        n_operations = round(n_operations * swarm_movement)
        print("swarm operations: "+str(n_operations))

        valid = False
        offset = -0.1
        while not(valid) and n_operations > 0:
            temp = tree_copy.copy()

            for i in range(n_operations):
                op = ops[random.randint(0, len(ops) - 1)]
                result = Op.tree_operation(helper, temp, op)

            new_distance = temp.phylogeny.distanza(helper, helper.best_particle.best)
            # print("SWARM. old=" + str(distance_swarm) + ", new=" + str(new_distance))
            if new_distance < distance_swarm+offset:
                # print(mytime() + "/// Movement to best swarm: " + str(p.number) + ", done some operations: ")
                tree_copy = temp.copy()
                valid = True
            else:
                offset += 0.01


    return it, result, p, tree_copy, start_time


def particle_iteration_attachment_plus_operations(it, p, helper):

    start_time = time.time()
    result = -1
    tree_copy = p.current_tree.copy()

    print(mytime() + "/// %d Trying clade attachment" % p.number)

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


    ops = list(range(2, Op.NUMBER))
    for i in range(int(helper.w)):
        op = ops.pop(random.randint(0, len(ops) - 1))
        result = Op.tree_operation(helper, tree_copy, op)
        print(mytime() + "/// %d Done operation" % p.number)

    return it, result, p, tree_copy, start_time


def particle_iteration_4(it, p, helper):

    start_time = time.time()
    tree_copy = p.current_tree.copy()
    result = -1
    ops = list(range(2, Op.NUMBER))

    if it > 0:

        particle_distance = tree_copy.phylogeny.distanza(helper, p.best)
        swarm_distance = tree_copy.phylogeny.distanza(helper, helper.best_particle.best)


        #self movement
        n_operations = int(round(p.velocity))
        # print("self operations: "+str(n_operations))
        for i in range(n_operations):
            op = ops[random.randint(0, len(ops) - 1)]
            result = Op.tree_operation(helper, tree_copy, op)
            # if result == 0:
            #     print(mytime() + "/// Particle: " + str(p.number) + ", done operation: " + str(op))


        # movement to best swarm
        new_particle_distance = 0.0
        if particle_distance > 0:
            offset = 0.15 * helper.c1
            particle_movement = 0.0
            while particle_movement < offset and offset > 0:
                temp = tree_copy.copy()
                op = ops[random.randint(0, len(ops) - 1)]
                result = Op.tree_operation(helper, temp, op)
                new_particle_distance = temp.phylogeny.distanza(helper, p.best)
                particle_movement = particle_distance - new_particle_distance
                offset -= 0.03 * helper.c1
            if particle_movement < offset:
                # print(mytime() + "/// Movement to best particle: " + str(p.number) + ", done operation: " + str(op))
                tree_copy = temp.copy()


        # movement to best swarm
        new_swarm_distance = 0.0
        if swarm_distance > 0:
            offset = 0.15 * helper.c2
            swarm_movement = 0.0
            while swarm_movement < offset and offset > 0:
                temp = tree_copy.copy()
                op = ops[random.randint(0, len(ops) - 1)]
                result = Op.tree_operation(helper, temp, op)
                new_swarm_distance = temp.phylogeny.distanza(helper, helper.best_particle.best)
                swarm_movement = swarm_distance - new_swarm_distance
                offset -= 0.03 * helper.c2
            if swarm_movement < offset:
                # print(mytime() + "/// Movement to best swarm: " + str(p.number) + ", done operation: " + str(op))
                tree_copy = temp.copy()



        # # print("curr velocity = " + str(p.velocity))
        # # print("next self = " + str(helper.w*p.velocity))
        #
        #
        # # print("new_particle_distance = " + str(new_particle_distance))
        # # print("new_swarm_distance  =   " + str(new_swarm_distance))
        #
        # self_factor = helper.w * p.velocity
        # particle_factor = helper.c1 * new_particle_distance
        # swarm_factor = helper.c2 * new_swarm_distance
        #
        #
        #
        #
        #
        # if self_factor + particle_factor + swarm_factor < (p.velocity-0.005):
        #     p.velocity -= 0.005
        # elif self_factor + particle_factor + swarm_factor > (p.velocity+0.005):
        #     p.velocity += 0.005
        #
        # # p.velocity = self_factor + particle_factor + swarm_factor
        #
        # if p.velocity > 0.06:
        #     p.velocity = 0.06
        # if p.velocity < 0.0:
        #     p.velocity = 0.0
        #
        # print("next = " + str(p.velocity))
        #
        # # print("self_factor:     " + str(self_factor))
        # # print("particle_factor: " + str(particle_factor))
        # # print("swarm_factor:    " + str(swarm_factor))
        # # print("\n")



        diff_p = 0
        diff_s = 0
        if p.best_particle_distance == 0 and p.best_swarm_distance == 0:
            p.best_particle_distance = new_particle_distance
            p.best_swarm_distance = new_swarm_distance
        else:
            diff_p = p.best_particle_distance - new_particle_distance
            diff_s = p.best_swarm_distance - new_swarm_distance
            p.best_particle_distance = new_particle_distance
            p.best_swarm_distance = new_swarm_distance


        change = 0
        if diff_p > 0:
            change += 0.2 * helper.c1
        elif diff_p < 0:
            change -= 0.2 * helper.c1
        if diff_s > 0:
            change += 0.2 * helper.c2
        elif diff_s < 0:
            change -= 0.2 * helper.c2

        p.velocity += change * (1 - helper.w)

        if p.velocity > 3:
            p.velocity = 3.0
        elif p.velocity < 1:
            p.velocity = 1.0

        print("vel: "+str(p.velocity))



    return it, result, p, tree_copy, start_time






def mytime():
    return( datetime.now().strftime("[%Y/%m/%d, %H:%M:%S] - ") )



def pso(nparticles, iterations, matrix):
    global particles
    global helper
    global data

    # Particle initialization
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
        print(" --- " + str(p.number) + ": " + str(Tree.greedy_loglikelihood(helper, p.current_tree)))
    print("")
    pool.close()
    pool.join()

    # non-parallel initialization
    # for i, p in enumerate(particles):
    #     cb_init_particle(init_particle(i, p, helper))

    data.starting_likelihood = helper.best_particle.best.likelihood
    data.initialization_end = time.time()

    data.pso_start = time.time()

    # single_core_run(helper, data, particles, iterations)
    parallel_run(helper, data, particles, iterations)

    data.pso_end = time.time()




def single_core_run(helper, data, particles, iterations):
    for it in range(iterations):
        start_it = time.time()

        print("\n------->  Iteration (%d)  <-------" % it)
        for p in particles:
            # if it == 20 and p.number == 20:
            #     p.current_tree.debug = True
            cb_particle_iteration(particle_iteration_4(it, p, helper))
            helper.w -= (10000)/((10000+Tree.greedy_loglikelihood(helper, helper.best_particle.best))*iterations)
            # helper.w -= 0.2/(iterations)

        data.best_iteration_likelihoods.append(helper.best_particle.best.likelihood)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))


def parallel_run(helper, data, particles, iterations):
    for it in range(iterations):
        print("\n------->  Iteration (%d)  <-------" % it)

        start_it = time.time()

        pool = mp.Pool(mp.cpu_count())
        processes = []
        for p in particles:
            processes.append(pool.apply_async(particle_iteration_4, args=(it, p, helper), callback=cb_particle_iteration))
            helper.w -= (10000)/((10000+Tree.greedy_loglikelihood(helper, helper.best_particle.best))*iterations)

        # before starting a new iteration we wait for every process to end
        # for p in processes:
        #     p.start()
        #     print("Got it")


        pool.close()
        pool.join()


        data.best_iteration_likelihoods.append(helper.best_particle.best.likelihood)
        data.iteration_times.append(data._passed_seconds(start_it, time.time()))


    # Adding backmutations
    i = 0
    while i < 30:
        add_back_mutations()
        i += 1
