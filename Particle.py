from Operation import Operation
from Tree import Tree
from Helper import Helper
from datetime import datetime
from collections import deque
import time
import numpy
import threading


class Particle(object):


    def __init__(self, cells, mutation_number, mutation_names, number, quiet):
        self.number = number
        self.current_tree = Tree.random(cells, mutation_number, mutation_names)
        self.quiet = quiet
        self.best = self.current_tree.copy() # best tree found by this particle
        self.max_stall_iterations = 500
        self.best_iteration_likelihoods = []


    def particle_start(self, helper, ns, lock):
        self.thread = threading.Thread(target = self.run_iterations, args = (helper, ns, lock))
        self.thread.start()


    def particle_join(self):
        self.thread.join()


    def run_iterations(self, helper, ns, lock):
        """execute the iterations and stops after reaching a stopping criteria"""
        start_time = time.time()
        old_lh = ns.best_swarm.likelihood
        improvements = deque([1] * self.max_stall_iterations) # queue
        bm_phase = False
        exit_local_iterations = 0

        for it in range(helper.iterations):
            self.particle_iteration(it, helper, ns.best_swarm.copy(), ns, lock)

            if self.number == 1:
                lh = ns.best_swarm.likelihood
                improvements.popleft()
                improvements.append(1 - lh / old_lh)
                old_lh = lh

                # update on screen
                if not self.quiet and it % 20 == 0:
                    print("\t%s\t\t%s" % (datetime.now().strftime("%H:%M:%S"), str(round(lh, 2))))

                # exit local optimum
                if exit_local_iterations < -100 and sum(list(improvements)[350:]) < helper.tolerance:
                    exit_local_iterations = 25
                    ns.attach = False
                elif exit_local_iterations < 0:
                    ns.attach = True
                exit_local_iterations -= 1

                # check if it's time to start adding backmutations
                if not(bm_phase) and self.start_backmutations((time.time()-start_time), helper.max_time, helper.automatic_stop, it, helper.iterations, improvements, helper.tolerance):
                    improvements = deque([1] * self.max_stall_iterations)
                    ns.operations = [0,1,2,3]
                    bm_phase = True

                # update list for lh plot
                self.best_iteration_likelihoods.append(lh)

                # check if it's time to stop
                if helper.automatic_stop and (sum(improvements) < helper.tolerance or (time.time()-start_time) >= (helper.max_time)):
                    ns.stop = True

            if helper.automatic_stop and ns.stop:
                break

        if self.number == 1:
            ns.best_iteration_likelihoods = self.best_iteration_likelihoods
        lock.acquire()
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

        # calculating distances
        particle_distance = 1 -  self.best.likelihood / self.current_tree.likelihood
        swarm_distance = 1 -  best_swarm.likelihood / self.current_tree.likelihood
        iteration_dist = max(particle_distance, swarm_distance)

        if ns.attach:

            # movement to particle best
            if particle_distance != 0:
                clade_to_be_attached = self.best.phylogeny.get_clade_by_distance(ns.max_dist, particle_distance)
                clade_to_be_attached = clade_to_be_attached.copy().detach()
                clade_destination = numpy.random.choice(self.current_tree.phylogeny.get_clades())
                clade_destination.attach_clade(self.current_tree, clade_to_be_attached)
                self.current_tree.phylogeny.losses_fix(self.current_tree, helper.mutation_number, helper.k, helper.max_deletions)

            # movement to swarm best
            if swarm_distance != 0:
                clade_to_be_attached = best_swarm.phylogeny.get_clade_by_distance(ns.max_dist, swarm_distance)
                clade_to_be_attached = clade_to_be_attached.copy().detach()
                clade_destination = numpy.random.choice(self.current_tree.phylogeny.get_clades())
                clade_destination.attach_clade(self.current_tree, clade_to_be_attached)
                self.current_tree.phylogeny.losses_fix(self.current_tree, helper.mutation_number, helper.k, helper.max_deletions)

        # self movement
        op = numpy.random.choice(ns.operations)
        Operation.tree_operation(self.current_tree, op, helper.k, helper.gamma, helper.max_deletions)
        self.current_tree.phylogeny.losses_fix(self.current_tree, helper.mutation_number, helper.k, helper.max_deletions)

        # updating log likelihood and bests
        lh = Tree.greedy_loglikelihood(self.current_tree, helper.matrix, helper.cells, helper.mutation_number)

        self.current_tree.likelihood = lh

        # update particle best
        if lh > self.best.likelihood:
            self.best = self.current_tree.copy()

        lock.acquire()

        # update swarm best
        if lh > ns.best_swarm.likelihood:
            ns.best_swarm = self.current_tree.copy()

        # update max distance
        if iteration_dist > ns.max_dist:
            ns.max_dist = iteration_dist

        lock.release()
