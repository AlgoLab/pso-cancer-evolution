from Operation import Operation
from Tree import Tree
from Helper import Helper
from datetime import datetime
from collections import deque
import time
import numpy


class Particle(object):


    def __init__(self, cells, mutation_number, mutation_names, number):
        self.number = number
        self.current_tree = Tree.random(cells, mutation_number, mutation_names)
        self.best = self.current_tree.copy()
        self.swarm_best_likelihoods = []
        self.particle_best_likelihoods = []


    def run_iterations(self, helper, ns, lock):
        """execute the iterations and stops after reaching a stopping criteria"""
        start_time = time.time()
        old_lh = ns.best_swarm.likelihood
        improvements = deque([1] * helper.max_stall_iterations) # queue
        bm_phase = False
        exit_local_iterations = 0

        for it in range(helper.iterations):
            self.particle_iteration(it, helper, ns.best_swarm.copy(), ns, lock)
            self.particle_best_likelihoods.append(self.best.likelihood)

            if self.number == 1:
                # update last iterations relative improvement
                lh = ns.best_swarm.likelihood
                improvements.popleft()
                improvements.append(1 - lh / old_lh)
                old_lh = lh

                # update on screen
                if not helper.quiet and it % 25 == 0:
                    print("\t%s\t\t%s" % (datetime.now().strftime("%H:%M:%S"), str(round(lh, 2))))

                # try to exit local optimum
                if exit_local_iterations < -100 and sum(list(improvements)[100:]) < 0:
                    exit_local_iterations = 20
                    ns.attach = False
                elif exit_local_iterations < 0:
                    ns.attach = True
                exit_local_iterations -= 1

                # check if it's time to start adding backmutations
                if not(bm_phase) and self.start_backmutations((time.time()-start_time), helper.max_time, helper.automatic_stop, it, helper.iterations, improvements, helper.tolerance):
                    improvements = deque([1] * helper.max_stall_iterations)
                    ns.operations = [0,1,2,3]
                    bm_phase = True
                    if helper.max_deletions == 0:
                        ns.stop = True

                # update list for lh plot
                self.swarm_best_likelihoods.append(lh)

                # check if it's time to stop
                if helper.automatic_stop and (sum(improvements) < helper.tolerance or (time.time()-start_time) >= (helper.max_time)):
                    ns.stop = True

            if helper.automatic_stop and ns.stop:
                break

        if self.number == 1:
            ns.swarm_best_likelihoods = self.swarm_best_likelihoods

        lock.acquire()
        tmp = ns.particle_best_likelihoods
        tmp[self.number] = self.particle_best_likelihoods
        ns.particle_best_likelihoods = tmp

        tmp = ns.iterations_performed
        tmp[self.number] = it + 1
        ns.iterations_performed = tmp
        lock.release()


    def start_backmutations(self, elapsed_time, max_time, automatic_stop, it, iterations, improvements, tolerance):
        """Check if it's time to start adding backmutations"""
        b1 = elapsed_time >= (3/4) * max_time
        b2 = not(automatic_stop) and it >= (3/4) * iterations
        b3 = automatic_stop and sum(improvements) < tolerance
        return (b1 or b2 or b3)


    def particle_iteration(self, it, helper, best_swarm, ns, lock):
        """The particle makes 3 movements and update the results"""

        op = numpy.random.choice(ns.operations)

        if op in [2,3] and ns.attach:
            # movement to particle best
            particle_distance = 1 - self.best.likelihood / self.current_tree.likelihood
            if particle_distance > 0:
                clade_to_be_attached = self.best.phylogeny.get_clade_by_distance(particle_distance)
                clade_to_be_attached = clade_to_be_attached.copy().detach()
                clade_destination = numpy.random.choice(self.current_tree.phylogeny.get_clades())
                clade_destination.attach_clade(self.current_tree, clade_to_be_attached)
                self.current_tree.phylogeny.losses_fix(self.current_tree, helper.mutation_number, helper.k, helper.max_deletions)

            # movement to swarm best
            swarm_distance = 1 - best_swarm.likelihood / self.current_tree.likelihood
            if swarm_distance > 0:
                clade_to_be_attached = best_swarm.phylogeny.get_clade_by_distance(swarm_distance)
                clade_to_be_attached = clade_to_be_attached.copy().detach()
                clade_destination = numpy.random.choice(self.current_tree.phylogeny.get_clades())
                clade_destination.attach_clade(self.current_tree, clade_to_be_attached)
                self.current_tree.phylogeny.losses_fix(self.current_tree, helper.mutation_number, helper.k, helper.max_deletions)

        # inertia movement
        Operation.tree_operation(self.current_tree, op, helper.k, helper.gamma, helper.max_deletions)
        self.current_tree.phylogeny.losses_fix(self.current_tree, helper.mutation_number, helper.k, helper.max_deletions)

        # calculate lh
        lh = Tree.greedy_loglikelihood(self.current_tree, helper.matrix, helper.cells, helper.mutation_number)
        self.current_tree.likelihood = lh

        # update particle best
        if lh > self.best.likelihood:
            self.best = self.current_tree.copy()

            # update swarm best
            lock.acquire()
            if lh > ns.best_swarm.likelihood or lh == ns.best_swarm.likelihood and len(self.current_tree.losses_list) < len(ns.best_swarm.losses_list):
                ns.best_swarm = self.current_tree.copy()
            lock.release()
