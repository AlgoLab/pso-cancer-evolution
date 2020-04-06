import copy
from Operation import Operation
from Tree import Tree
import multiprocessing as mp
import time
import random
from Operation import Operation as Op
from Helper import Helper
from datetime import datetime

class Particle(object):

    def __init__(self, cells, mutation_number, mutation_names, number):
        self.current_tree = Tree.random(cells, mutation_number, mutation_names)
        self.number = number
        self.best = self.current_tree # best tree found by this particle

        self.max_stall_iterations = 100
        self.tolerance = 0.001



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
        improvements = [1] * self.max_stall_iterations #queue

        for it in range(iterations):
            start_it = time.time()
            self.cb_particle_iteration(self.particle_iteration(it, helper, ns.best_swarm.copy()), ns, lock)

            if self.number == 0:

                lh = ns.best_swarm.likelihood
                improvements.pop(0)
                improvements.append(1-lh/old_lh)
                old_lh = lh

                if it % 10 == 0:
                    print("\t%s\t\t%s" % (datetime.now().strftime("%H:%M:%S"), str(round(lh, 2))))

                lock.acquire()

                ns.best_iteration_likelihoods = self.append_to_shared_array(ns.best_iteration_likelihoods, lh)
                ns.iteration_times = self.append_to_shared_array(ns.iteration_times, time.time() - start_it)

                if ns.automatic_stop and (sum(improvements) < self.tolerance or (time.time() - start_time) >= (helper.max_time-2)):
                    ns.stop = True

                lock.release()

            if ns.automatic_stop and ns.stop:
                break



    def append_to_shared_array(self, arr, v):
        arr.append(v)
        return arr



    def cb_particle_iteration(self, r, ns, lock):
        helper, i, tree_copy, start_time = r

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
        tmp[self.number].append(time.time() - start_time)
        ns.particle_iteration_times = tmp

        lock.release()



    def particle_iteration(self, it, helper, best_swarm):
        start_time = time.time()
        tree_copy = self.current_tree.copy()

        # movement to particle best
        particle_distance = tree_copy.phylogeny.distance(self.best.phylogeny, helper.mutation_number)
        if particle_distance != 0:
            clade_to_be_attached = self.best.phylogeny.get_clade_by_distance(helper, particle_distance, it, helper.c1)
            clade_to_be_attached = clade_to_be_attached.copy().detach()
            clade_destination = random.choice(tree_copy.phylogeny.get_clades())
            clade_destination.attach_clade(helper, tree_copy, clade_to_be_attached)

        # movement to swarm best
        swarm_distance = tree_copy.phylogeny.distance(best_swarm.phylogeny, helper.mutation_number)
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

        return helper, it, tree_copy, start_time



    def add_back_mutation(self, it, helper, data):
        start_time = time.time()
        tree_copy = self.current_tree.copy()

        old_lh = tree_copy.likelihood
        Op.tree_operation(helper, tree_copy, 0)
        tree_copy.likelihood = Tree.greedy_loglikelihood(helper, tree_copy)
        new_lh = tree_copy.likelihood

        if new_lh > old_lh:
            self.current_tree = tree_copy.copy()
            self.best = self.current_tree

        data.particle_iteration_times[self.number].append(time.time() - start_time)

        return data
