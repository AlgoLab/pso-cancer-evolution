from Operation import Operation
from Tree import Tree
from Operation import Operation as Op
from Helper import Helper
import multiprocessing
from datetime import datetime
from collections import deque
import sys
import time
import random


class Particle(object):


    def __init__(self, cells, mutation_number, mutation_names, number):
        self.current_tree = Tree.random(cells, mutation_number, mutation_names)
        self.number = number
        self.best = self.current_tree # best tree found by this particle
        self.max_stall_iterations = 200
        self.tolerance = 0.002
        self.best_iteration_likelihoods = []


    def particle_start(self, iterations, helper, ns, lock):
        if iterations == 0:
            iterations = 50000
        self.proc = multiprocessing.Process(target = self.run_iterations, args = (iterations, helper, ns, lock))
        self.proc.start()


    def particle_join(self):
        self.proc.join()


    def run_iterations(self, iterations, helper, ns, lock):
        """execute the iterations and stops after reaching a stopping criteria"""
        start_time = time.time()
        old_lh = ns.best_swarm.likelihood
        improvements = deque([1] * self.max_stall_iterations) # queue
        bm = False

        for it in range(iterations):
            self.particle_iteration(it, helper, ns.best_swarm, ns, lock)

            if self.number == 0:

                lh = ns.best_swarm.likelihood
                improvements.popleft()
                improvements.append(1-lh/old_lh)
                old_lh = lh

                if it % 20 == 0:
                    print("\t%s\t\t%s" % (datetime.now().strftime("%H:%M:%S"), str(round(lh, 2))))

                if not(bm):
                    # if 3/4 of max time
                    b1 = (time.time() - start_time) >= (3/4)*(helper.max_time)

                    # if iterations given in input and 3/4 of iterations
                    b2 = not(ns.automatic_stop) and it >= (3/4)*iterations

                    # if iterations not given in input and stuck on fitness value
                    b3 = ns.automatic_stop and sum(improvements) < self.tolerance

                    if b1 or b2 or b3:
                        improvements = deque([1] * self.max_stall_iterations)
                        bm = True
                        ns.operations = [0,1,2,3]

                self.best_iteration_likelihoods.append(lh)

                if ns.automatic_stop and (sum(improvements) < self.tolerance or (time.time() - start_time) >= (helper.max_time)):
                    ns.stop = True

            if ns.automatic_stop and ns.stop:
                break

        if self.number == 0:
            ns.best_iteration_likelihoods = self.best_iteration_likelihoods


    def particle_iteration(self, it, helper, best_swarm, ns, lock):
        """The particle makes 3 movements and update the results"""
        start_it = time.time()

        # movement to particle best
        particle_distance = self.current_tree.phylogeny.distance(self.best.phylogeny, helper.mutation_number)
        if particle_distance != 0:
            clade_to_be_attached = self.best.phylogeny.get_clade_by_distance(helper, particle_distance)
            clade_to_be_attached = clade_to_be_attached.copy().detach()
            clade_destination = random.choice(self.current_tree.phylogeny.get_clades())
            clade_destination.attach_clade(helper, self.current_tree, clade_to_be_attached)
            self.current_tree.phylogeny.losses_fix(helper, self.current_tree)

        # movement to swarm best
        swarm_distance = self.current_tree.phylogeny.distance(best_swarm.phylogeny, helper.mutation_number)
        if swarm_distance != 0:
            clade_to_be_attached = best_swarm.phylogeny.get_clade_by_distance(helper, swarm_distance)
            clade_to_be_attached = clade_to_be_attached.copy().detach()
            clade_destination = random.choice(self.current_tree.phylogeny.get_clades())
            clade_destination.attach_clade(helper, self.current_tree, clade_to_be_attached)
            self.current_tree.phylogeny.losses_fix(helper, self.current_tree)

        # self movement
        op = random.choice(ns.operations)
        Op.tree_operation(helper, self.current_tree, op)
        self.current_tree.phylogeny.losses_fix(helper, self.current_tree)

        # updating log likelihood and bests
        lh = Tree.greedy_loglikelihood(helper, self.current_tree)
        self.current_tree.likelihood = lh

        # update particle best
        if lh > self.best.likelihood:
            self.best = self.current_tree

        lock.acquire()

        # update swarm best
        if lh > ns.best_swarm.likelihood:
            ns.best_swarm = self.current_tree

        # update average distance
        tmp = (particle_distance + swarm_distance) / 2
        helper.avg_dist = (helper.avg_dist * it + tmp) / (it + 1)

        # update particle iteration times
        tmp = ns.particle_iteration_times
        tmp[self.number].append(time.time() - start_it)
        ns.particle_iteration_times = tmp

        lock.release()
