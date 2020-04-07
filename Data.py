from Tree import Tree
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

        self.best_iteration_likelihoods = []

        self.true_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.missing_values = 0

        self.true_positives_relative = 0
        self.true_negatives_relative = 0
        self.false_negatives_relative = 0
        self.false_positives_relative = 0
        self.missing_values_relative = 0
        self.accuracy = 0

        self.iterations_performed = []



    def average_iteration_time_per_particle(self):
        """
            For each particle, return the average time
            that particle has taken to complete its step
        """
        avg = [0] * self.nofparticles
        for i in range(self.nofparticles):
            avg[i] = np.mean(self.particle_iteration_times[i])
        return avg



    def set_iterations(self, iterations):
        self.iterations = iterations



    def calculate_accuracy(self, helper):
        Tree.greedy_loglikelihood_with_data(helper, helper.best_particle.best, self)
        tot = self.true_positives + self.true_negatives + self.false_negatives + self.false_positives + self.missing_values
        self.true_positives_relative = (100 * self.true_positives) / tot
        self.true_negatives_relative = (100 * self.true_negatives) / tot
        self.false_positives_relative = (100 * self.false_positives) / tot
        self.false_negatives_relative = (100 * self.false_negatives) / tot
        self.missing_values_relative = (100 * self.missing_values) / tot
        self.accuracy = 100 - (self.false_positives_relative + self.false_negatives_relative)



    def summary(self, helper, dir):

        # tree image
        helper.best_particle.best.phylogeny.save(dir + "/best.gv")

        # likelihood plot pdf
        plt.title("Likelihood over Time")
        plt.xlabel("Iteration number")
        plt.ylabel("Log Likelihood")
        plt.plot(self.best_iteration_likelihoods)
        plt.tight_layout()
        plt.savefig(dir + "/lh.pdf")
        plt.clf()

        # text file with info
        f = open(dir + "/results.txt", "w+")
        f.write(">> Number of particles: %d\n" % self.nofparticles)
        f.write(">> Number of iterations for each particle:\n")
        for i,its in enumerate(self.iterations_performed):
            f.write("\t- particle %d: %d\n" % (i, its))
        f.write("\n>> Number of cells: %d\n" % helper.cells)
        f.write(">> Number of mutations: %d\n" % helper.mutation_number)
        f.write("\n>> Starting likelihood: %f\n" % self.starting_likelihood)
        f.write(">> Final likelihood:    %f\n" % helper.best_particle.best.likelihood)
        f.write("\n>> Added mutations: %s\n" % ', '.join(map(str, helper.best_particle.best.losses_list)))
        f.write("\n>> False negatives:      %d\t(%s%%)\n" % (self.false_negatives, str(round(self.false_negatives_relative, 1))))
        f.write(">> False positives:      %d\t(%s%%)\n" % (self.false_positives, str(round(self.false_positives_relative, 1))))
        f.write(">> True negatives:       %d\t(%s%%)\n" % (self.true_negatives, str(round(self.true_negatives_relative, 1))))
        f.write(">> True positives:       %d\t(%s%%)\n" % (self.true_positives, str(round(self.true_positives_relative, 1))))
        f.write(">> Added missing values: %d\t(%s%%)\n" % (self.missing_values, str(round(self.missing_values_relative, 1))))
        f.write(">> Accuracy: %s%%\n" % str(round(self.accuracy, 1)))
        f.write("\n>> PSO completed in %f seconds\n" % (self.pso_end - self.pso_start))
        f.write(">> Initialization took %f seconds\n" % (self.initialization_end - self.initialization_start))
        f.write(">> Average iteration time per particle:\n")
        for i,t in enumerate(self.average_iteration_time_per_particle()):
            f.write("\t- particle %d: %s\n" % (i, str(round(t, 4))))
        f.write("\nBest Tree in Tikz format:\n")
        f.write(helper.best_particle.best.phylogeny.to_tikz())
        f.close()




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
        f.write("\tParticles & Iterations & StartingLikelihood & Final Likelihood & CPU Time (s) \\\\ \\midrule \\midrule\n")
        for data in runs_data:
            f.write("\t%s & %s & %s & %s & %s \\\\\n" % (data.nofparticles, data.iterations, data.starting_likelihood, data.helper.best_particle.best.likelihood, data.pso_passed_seconds()))
        f.write("\\end{tabular}")
        f.close()
