from Tree import Tree
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import numpy


class Data(object):


    def __init__(self, filename, nofparticles, output):
        self.filename = filename[filename.rfind('/')+1:]
        self.pso_start = 0
        self.pso_end = 0
        self.starting_likelihood = 0
        self.nofparticles = nofparticles
        self.iterations_performed = [0] * nofparticles
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
        self.mem_over_time = []
        self.output = output
        self.best = None


    def get_total_time(self):
        return (self.pso_end - self.pso_start)


    def calculate_relative_data(self, helper, current_matrix):
        """Calculates percentage of a part of the collected data"""
        lh = Tree.greedy_loglikelihood_with_data(self.best, current_matrix, helper.cells, helper.mutation_number, self)
        tot = self.true_positives + self.true_negatives + self.false_negatives + self.false_positives + self.missing_values
        self.true_positives_relative = (100 * self.true_positives) / tot
        self.true_negatives_relative = (100 * self.true_negatives) / tot
        self.false_positives_relative = (100 * self.false_positives) / tot
        self.false_negatives_relative = (100 * self.false_negatives) / tot
        self.missing_values_relative = (100 * self.missing_values) / tot
        return lh


    def summary(self, helper, dir):
        """
            Creates a text file with the data of the execution, an image of the
            best tree found, a plot of the likelihood over time
        """
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.calculate_relative_data(helper, helper.matrix)

        # tree [image]
        if self.output in ["image", "all"]:
            self.best.phylogeny.save(dir + "/best.gv")

        # likelihood plot [pdf]
        if self.output in ["plot", "all"]:
            plt.title("Likelihood over Time")
            plt.xlabel("Iteration number")
            plt.ylabel("Log Likelihood")
            plt.plot(self.best_iteration_likelihoods)
            plt.tight_layout()
            plt.savefig(dir + "/lh.pdf")
            plt.clf()

        # execution info [text file]
        if self.output in ["text_file", "all"]:
            f = open(dir + "/results_" + self.filename, "w+")
            f.write("\n---------------------------------\n")
            f.write("\n>> Starting likelihood: %f\n" % self.starting_likelihood)
            f.write(">> Final likelihood:    %f\n" % self.best.likelihood)
            f.write("\n>> False negatives:      %d\t(%s%%)\n" % (self.false_negatives, str(round(self.false_negatives_relative, 1))))
            f.write(">> False positives:      %d\t(%s%%)\n" % (self.false_positives, str(round(self.false_positives_relative, 1))))
            f.write(">> True negatives:       %d\t(%s%%)\n" % (self.true_negatives, str(round(self.true_negatives_relative, 1))))
            f.write(">> True positives:       %d\t(%s%%)\n" % (self.true_positives, str(round(self.true_positives_relative, 1))))
            f.write(">> Added missing values: %d\t(%s%%)\n" % (self.missing_values, str(round(self.missing_values_relative, 1))))
            f.write("\n---------------------------------\n")
            if helper.truematrix != 0:
                lh = self.calculate_relative_data(helper, helper.truematrix)
                f.write("\n[Actual correct matrix]\n")
                f.write(">> Starting lh = %f\n" % (self.starting_likelihood_true))
                f.write(">> Final lh =    %f\n" % (lh))
                f.write("\n>> False negatives:      %d\t(%s%%)\n" % (self.false_negatives, str(round(self.false_negatives_relative, 1))))
                f.write(">> False positives:      %d\t(%s%%)\n" % (self.false_positives, str(round(self.false_positives_relative, 1))))
                f.write(">> True negatives:       %d\t(%s%%)\n" % (self.true_negatives, str(round(self.true_negatives_relative, 1))))
                f.write(">> True positives:       %d\t(%s%%)\n" % (self.true_positives, str(round(self.true_positives_relative, 1))))
                f.write(">> Added missing values: %d\t(%s%%)\n" % (self.missing_values, str(round(self.missing_values_relative, 1))))
                f.write("\n---------------------------------\n")
            f.write("\n>> Number of cores: %d\n" % helper.cores)
            f.write(">> Number of particles: %d\n" % self.nofparticles)
            f.write(">> Number of iterations for each particle:\n")
            for i in range(self.nofparticles):
                f.write("\t- particle %d: %d\n" % (i, self.iterations_performed[i]))
            f.write("\n>> Number of cells: %d\n" % helper.cells)
            f.write(">> Number of mutations: %d\n" % helper.mutation_number)
            f.write("\n>> Added mutations: %s\n" % ', '.join(map(str, self.best.losses_list)))
            f.write(">> PSO completed in %f seconds\n" % self.get_total_time())
            f.write("\n---------------------------------\n")
            f.write("\nBest Tree in Tikz format:\n")
            f.write(self.best.phylogeny.to_tikz())
            f.write("\n\n---------------------------------\n\n")
            f.close()


    @classmethod
    def runs_summary(cls, runs, runs_data, dir):
        """Creates the summary for each run and a plot"""
        likelihoods = []
        for data in runs_data:
            likelihoods.append(max(data.best_iteration_likelihoods))

        # pdf plot
        ax = plt.figure().gca()
        plt.title("Likelihood per run")
        plt.xlabel("Number of particles")
        plt.ylabel("Log Likelihood")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(runs, likelihoods)
        plt.savefig(dir + "/likelihood.pdf")

        # info text file
        f = open(dir + "/results_multiple_run.txt", "w+")
        f.write("\\begin{tabular}{||*{5}{c}||}\n")
        f.write("\t\\hline\n")
        f.write("\tParticles & Iterations & Starting Likelihood & Final Likelihood & CPU Time (s)\\\\\n")
        f.write("\t\\hline\\hline\n")
        for data in runs_data:
            nofp = str(data.nofparticles)
            if min(data.iterations_performed) == max(data.iterations_performed):
                its = str(data.iterations_performed[0])
            else:
                its = str("[" + str(min(data.iterations_performed)) + "," + str(max(data.iterations_performed)) + "]")
            lh1 = data.starting_likelihood
            lh2 = max(data.best_iteration_likelihoods)
            t = data.pso_end-data.pso_start
            f.write("\t%s & %s & %f & %f & %f \\\\\n" % (nofp, its, lh1, lh2, t))
            f.write("\t\\hline\n")
        f.write("\\end{tabular}")
        f.close()
