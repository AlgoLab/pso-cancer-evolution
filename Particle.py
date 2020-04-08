import copy
from Operation import Operation
from Tree import Tree
import multiprocessing as mp
import time
import random
from Operation import Operation as Op
from Helper import Helper
from datetime import datetime
from collections import deque
import sys

class Particle(object):

    def __init__(self, cells, mutation_number, mutation_names, number):
        self.current_tree = Tree.random(cells, mutation_number, mutation_names)
        self.number = number
        self.best = self.current_tree # best tree found by this particle

        self.max_stall_iterations = 200
        self.tolerance = 0.003



    def particle_start(self, iterations, helper, ns, lock):
        if iterations == 0:
            iterations = 10000
        self.proc = mp.Process(target = self.run_iterations, args = (iterations, helper, ns, lock))
        self.proc.start()



    def particle_join(self):
        self.proc.join()



    def run_iterations(self, iterations, helper, ns, lock):
        start_time = time.time()
        old_lh = ns.best_swarm.likelihood
        improvements = deque([1] * self.max_stall_iterations) # queue
        bm = False
        ops = [2,3]

        for it in range(iterations):
            self.particle_iteration(it, helper, ns.best_swarm.copy(), ns, lock, ops)

            if self.number == 0:

                lh = ns.best_swarm.likelihood
                improvements.popleft()
                improvements.append(1-lh/old_lh)
                old_lh = lh

                if it % 10 == 0:
                    print("\t%s\t\t%s" % (datetime.now().strftime("%H:%M:%S"), str(round(lh, 2))))

                if not(bm) and sum(improvements) < self.tolerance:
                    improvements = deque([1] * self.max_stall_iterations)
                    bm = True
                    ops = [0,1]

                lock.acquire()

                ns.best_iteration_likelihoods = self.append_to_shared_array(ns.best_iteration_likelihoods, lh)

                if ns.automatic_stop and (sum(improvements) < self.tolerance or (time.time() - start_time) >= (helper.max_time)):
                    ns.stop = True

                lock.release()

            if ns.automatic_stop and ns.stop:
                break



    def append_to_shared_array(self, arr, v):
        arr.append(v)
        return arr



    def particle_iteration(self, it, helper, best_swarm, ns, lock, ops):
        start_it = time.time()
        tree_copy = self.current_tree.copy()

        # movement to particle best
        particle_distance = tree_copy.phylogeny.distance(self.best.phylogeny, helper.mutation_number)
        if particle_distance != 0:
            clade_to_be_attached = self.best.phylogeny.get_clade_by_distance(helper, particle_distance, it, helper.c1)
            clade_to_be_attached = clade_to_be_attached.copy().detach()
            clade_destination = random.choice(tree_copy.phylogeny.get_clades())
            clade_destination.attach_clade(helper, tree_copy, clade_to_be_attached)
            tree_copy.phylogeny.losses_fix(helper, tree_copy)

        # movement to swarm best
        swarm_distance = tree_copy.phylogeny.distance(best_swarm.phylogeny, helper.mutation_number)
        if swarm_distance != 0:
            clade_to_be_attached = best_swarm.phylogeny.get_clade_by_distance(helper, swarm_distance, it, helper.c2)
            clade_to_be_attached = clade_to_be_attached.copy().detach()
            clade_destination = random.choice(tree_copy.phylogeny.get_clades())
            clade_destination.attach_clade(helper, tree_copy, clade_to_be_attached)
            tree_copy.phylogeny.losses_fix(helper, tree_copy)

        # self movement
        op = random.choice(ops)
        Op.tree_operation(helper, tree_copy, op)
        tree_copy.phylogeny.losses_fix(helper, tree_copy)

        # updating log likelihood and bests
        lh = Tree.greedy_loglikelihood(helper, tree_copy)
        tree_copy.likelihood = lh
        self.current_tree = tree_copy

        # update particle best
        best_particle_lh = self.best.likelihood
        if lh > best_particle_lh:
            self.best = tree_copy

        lock.acquire()

        # update swarm best
        best_swarm_lh = ns.best_swarm.likelihood
        if lh > best_swarm_lh:
            ns.best_swarm = tree_copy

        # update particle iteration times
        tmp = ns.particle_iteration_times
        tmp[self.number].append(time.time() - start_it)
        ns.particle_iteration_times = tmp

        lock.release()
