from Tree import Tree
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import numpy


class Data(object):


    def __init__(self, filename, n_particles, output):
        self.filename = filename[filename.rfind('/')+1:]
        self.pso_start = 0
        self.pso_end = 0
        self.starting_likelihood = 0
        self.n_particles = n_particles
        self.iterations_performed = [0] * n_particles
        self.swarm_best_likelihoods = []
        self.particles_best_likelihoods = []
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

        # swarm best likelihood plot [pdf]
        if self.output in ["plots", "all"]:
            plt.title("Likelihood over Time")
            plt.xlabel("Iteration number")
            plt.ylabel("Log Likelihood")
            plt.plot(self.swarm_best_likelihoods)
            plt.tight_layout()
            plt.savefig(dir + "/swarm_best_lh.pdf")
            plt.clf()

            plt.title("Particles likelihood over Time")
            plt.xlabel("Iteration number")
            plt.ylabel("Log Likelihood")
            for lh in self.particles_best_likelihoods:
                plt.plot(lh)
            plt.tight_layout()
            plt.savefig(dir + "/particles_best_lh.pdf")
            plt.clf()

        # execution info [text file]
        if self.output in ["text_file", "all"]:
            f = open(dir + "/results_" + self.filename, "w+")

            f.write("\n---------------------------------\n")
            f.write("\n>> Starting likelihood:\t%f\n" % self.starting_likelihood)
            f.write(">> Final likelihood:\t%f\n" % self.best.likelihood)
            f.write("\n>> False negatives:\t %d\t(%s%%)\n" % (self.false_negatives, str(round(self.false_negatives_relative, 2))))
            f.write(">> False positives:\t %d\t(%s%%)\n" % (self.false_positives, str(round(self.false_positives_relative, 2))))
            f.write(">> True negatives:\t %d\t(%s%%)\n" % (self.true_negatives, str(round(self.true_negatives_relative, 2))))
            f.write(">> True positives:\t %d\t(%s%%)\n" % (self.true_positives, str(round(self.true_positives_relative, 2))))
            f.write(">> Added missing values: %d\t(%s%%)\n" % (self.missing_values, str(round(self.missing_values_relative, 2))))
            f.write("\n---------------------------------\n")

            if helper.truematrix != None:
                lh = self.calculate_relative_data(helper, helper.truematrix)
                f.write("\n[Actual correct matrix]\n")
                f.write(">> Starting likelihood:\t%f\n" % (self.starting_likelihood_true))
                f.write(">> Final likelihood:\t%f\n" % (lh))
                f.write("\n>> False negatives:\t %d\t(%s%%)\n" % (self.false_negatives, str(round(self.false_negatives_relative, 2))))
                f.write(">> False positives:\t %d\t(%s%%)\n" % (self.false_positives, str(round(self.false_positives_relative, 2))))
                f.write(">> True negatives:\t %d\t(%s%%)\n" % (self.true_negatives, str(round(self.true_negatives_relative, 2))))
                f.write(">> True positives:\t %d\t(%s%%)\n" % (self.true_positives, str(round(self.true_positives_relative, 2))))
                f.write(">> Added missing values: %d\t(%s%%)\n" % (self.missing_values, str(round(self.missing_values_relative, 2))))
                f.write("\n---------------------------------\n")

            f.write("\n>> PSO completed in %f seconds\n" % self.get_total_time())
            f.write(">> Added mutations: %d\n" % len(self.best.losses_list))
            if len(self.best.losses_list) > 0:
                names = [x.name for x in self.best.losses_list]
                names.sort()
                i = 0
                tmp = names[0]
                for loss in names:
                    if tmp == loss:
                        i += 1
                    else:
                        f.write("\t- \"%s\" -> %d\n" % (str(tmp), i))
                        i = 1
                        tmp = loss
                f.write("\t- \"%s\" -> %d\n" % (str(tmp), i))
            f.write("\n---------------------------------\n")

            f.write("\n[Parameters used]\n")
            f.write(">> Particles: %d\n" % self.n_particles)
            f.write(">> Cores: %d\n" % helper.cores)
            f.write(">> Cells: %d\n" % helper.cells)
            f.write(">> Mutations: %d\n" % helper.mutation_number)
            f.write(">> Alpha: ")
            if numpy.sum(numpy.array([x==helper.alpha[0] for x in helper.alpha])) == helper.mutation_number:
                f.write("%s\n" % str(helper.alpha[0]))
            else:
                f.write("%s\n" % str(helper.alpha))
            f.write(">> Beta: %s\n" % str(helper.beta))
            f.write(">> Gamma: ")
            if numpy.sum(numpy.array([x==helper.gamma[0] for x in helper.gamma])) == helper.mutation_number:
                f.write("%s\n" % str(helper.gamma[0]))
            else:
                f.write("%s\n" % str(helper.gamma))
            f.write(">> K: %d\n" % helper.k)
            f.write(">> Max deletions: %s\n" % str(helper.max_deletions))
            f.write(">> Tolerance: %s\n" % str(helper.tolerance))
            f.write(">> Max time: %s\n" % str(helper.max_time))
            if not helper.automatic_stop:
                f.write(">> Iterations: %d\n" % helper.cores)
            else:
                f.write(">> Iterations:\n")
                for i in range(self.n_particles):
                    f.write("\t- particle %d: %d\n" % (i, self.iterations_performed[i]))
            f.write("\n---------------------------------\n\n")
            f.close()


    @classmethod
    def runs_summary(cls, n_particles, runs_data, dir):
        """Creates the summary for each run and a plot"""
        likelihoods = []
        for data in runs_data:
            likelihoods.append(max(data.swarm_best_likelihoods))

        # pdf plot
        ax = plt.figure().gca()
        plt.title("Likelihood per run")
        plt.xlabel("Number of particles")
        plt.ylabel("Log Likelihood")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(n_particles, likelihoods)
        plt.savefig(dir + "/likelihood.pdf")

        # info text file
        f = open(dir + "/results_multiple_run.txt", "w+")
        f.write("\\begin{tabular}{||*{5}{c}||}\n")
        f.write("\t\\hline\n")
        f.write("\tParticles & Iterations & Starting Likelihood & Final Likelihood & CPU Time (s)\\\\\n")
        f.write("\t\\hline\\hline\n")
        for data in runs_data:
            nofp = str(data.n_particles)
            if min(data.iterations_performed) == max(data.iterations_performed):
                its = str(data.iterations_performed[0])
            else:
                its = str("[" + str(min(data.iterations_performed)) + "," + str(max(data.iterations_performed)) + "]")
            lh1 = data.starting_likelihood
            lh2 = max(data.swarm_best_likelihoods)
            t = data.pso_end-data.pso_start
            f.write("\t%s & %s & %f & %f & %f \\\\\n" % (nofp, its, lh1, lh2, t))
            f.write("\t\\hline\n")
        f.write("\\end{tabular}")
        f.close()
