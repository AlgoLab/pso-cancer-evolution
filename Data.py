from Tree import Tree
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
import sys
import os
import numpy as np

class Data(object):

    def __init__(self, nofparticles, iterations):
        self.pso_start = 0
        self.pso_end = 0
        self.initialization_start = 0
        self.initialization_end = 0
        self.initialization_times = []
        self.starting_likelihood = 0

        self.nofparticles = nofparticles
        self.iterations = iterations
        self.particle_iteration_times = [[] for p in range(nofparticles)]
        self.iteration_times = []

        self.best_iteration_likelihoods = []

        self.true_positive = 0
        self.true_negative = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.missing_values = 0



    def pso_passed_seconds(self):
        return self._passed_seconds(self.pso_start, self.pso_end)



    def initialization_passed_seconds(self):
        return self._passed_seconds(self.initialization_start, self.initialization_end)



    def iteration_passed_seconds(self, start, end):
        return self._passed_seconds(start, end)



    def average_iteration_time(self):
        """Returns the average time for every iteration """
        sum = 0
        for t in self.iteration_times:
            sum += t
        return t / self.iterations



    def average_particle_time(self):
        """
            For each particle, return the average time
            that particle has taken to complete its step
        """
        avg = [0] * self.nofparticles
        for i in range(self.nofparticles):
            avg[i] = np.mean(self.particle_iteration_times[i])

        return avg



    def average_iteration_particle_time(self):
        average_times = [0] * self.iterations
        n = [0] * self.iterations

        for j in range(self.iterations):
            for i in range(self.nofparticles):
                if len(self.particle_iteration_times[i]) > j:
                    average_times[j] += self.particle_iteration_times[i][j]
                    n[j] += 1

        for j in range(self.iterations):
            if n[j] == 0:
                average_times[j] = 0
            else:
                average_times[j] /= n[j]

        return average_times



    def average_overall_particle(self):
        p_times = self.average_particle_time()
        sum = 0
        for t in p_times:
            sum += t
        return sum / self.nofparticles



    def _passed_seconds(self, start, end):
        return end - start



    def set_iterations(self, iterations):
        self.iterations = iterations



    def summary(self, helper, dir):
        Tree.greedy_loglikelihood_with_data(helper, helper.best_particle.best, self)
        f = open(dir + "/results.txt", "w+")
        f.write(">> Number of particles: %d\n" % self.nofparticles)
        f.write(">> Number of iterations: %d\n" % self.iterations)
        f.write(">> Number of cells: %d\n" % helper.cells)
        f.write(">> Number of mutations: %d\n" % helper.mutation_number)
        f.write(">> Starting likelihood: %f\n" % self.starting_likelihood)
        f.write(">> Best likelihood: %f\n" % helper.best_particle.best.likelihood)
        f.write(">> Added mutations: %s\n" % ', '.join(map(str, helper.best_particle.best.losses_list)))
        f.write(">> False negatives: %d\n" % self.false_negatives)
        f.write(">> False positives: %d\n" % self.false_positives)
        f.write(">> Added missing values: %d\n" % self.missing_values)
        f.write(">> PSO completed in %f seconds\n" % self.pso_passed_seconds())
        f.write(">> Initialization took %f seconds\n" % self.initialization_passed_seconds())
        f.write(">> Average iteration time: %f seconds\n" % self.average_iteration_time())
        f.write(">> Average particle time: %f seconds\n" % self.average_overall_particle())

        ax = plt.figure().gca()

        # first subplot
        plt.subplot(3, 1, 1)
        plt.title("Average Particle Time")
        plt.xlabel("Particle number")
        plt.ylabel("Time (in seconds)")
        avg_particle_time_ = self.average_particle_time()
        plt.plot(avg_particle_time_)
        plt.ylim(bottom=0, top=max(avg_particle_time_))

        # second subplot
        plt.subplot(3, 1, 2)
        plt.title("Average Particle Time per Iteration")
        plt.xlabel("Iteration number")
        plt.ylabel("Time (in seconds)")
        avg_it_particle_time = self.average_iteration_particle_time()
        plt.plot([i for i in range(len(avg_it_particle_time))], avg_it_particle_time)
        plt.ylim(top = max(avg_it_particle_time))

        # third subplot
        plt.subplot(3, 1, 3)
        plt.title("Likelihood over Time")
        plt.xlabel("Iteration number")
        plt.ylabel("Log Likelihood")
        plt.plot(self.best_iteration_likelihoods)

        plt.tight_layout()
        plt.savefig(dir + "/data.pdf")

        helper.best_particle.best.phylogeny.save(dir + "/best.gv")

        f.write("Best Tree in Tikz format:\n")
        f.write(helper.best_particle.best.phylogeny.to_tikz())

        f.close()
        plt.clf()



    @classmethod
    def runs_summary(cls, runs, runs_data, dir):
        likelihoods = []
        for data in runs_data:
            likelihoods.append(data.helper.best_particle.best.likelihood)

        ax = plt.figure().gca()
        plt.title("Likelihood per run")
        plt.xlabel("Run number")
        plt.ylabel("Log Likelihood")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(runs, likelihoods)
        plt.savefig(dir + "/likelihood.pdf")

        f = open(dir + "/results.txt", "w+")
        f.write("Run results table for LaTeX:\n")
        f.write("\\begin{tabular}{*{5}{c}}\n")
        f.write("\tParticelle & Iterazioni & Likelihood Iniziale & Likelihood Migliore & CPU Time (s) \\\\ \\midrule \\midrule\n")
        for data in runs_data:
            f.write("\t%s & %s & %s & %s & %s \\\\\n" % (data.nofparticles, data.iterations, data.starting_likelihood, data.helper.best_particle.best.likelihood, data.pso_passed_seconds()))
        f.write("\\end{tabular}")
        f.close()
