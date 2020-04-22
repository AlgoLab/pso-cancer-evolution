from Node import Node
from Operation import Operation as Op
import numpy
import math
import copy

# for a uniform random tree generation
used_combinations = []


class Tree(object):


    def __init__(self, cells, mutations):
        self.cells = cells
        self.mutations = mutations
        self.losses_list = []
        self.k_losses_list = [0] * mutations
        self.best_sigma = [0] * cells
        self.likelihood = float("-inf")
        self.phylogeny = None


    def update_losses_list(self):
        """Update the two losses lists of this tree"""
        ll = []
        kll = [0] * self.mutations
        for n in self.phylogeny.traverse():
            if n.loss:
                ll.append(n)
                kll[n.mutation_id] += 1
        self.losses_list = ll
        self.k_losses_list = kll


    def copy(self):
        """Copies everything in this tree"""
        t = Tree(self.cells, self.mutations)
        t.likelihood = self.likelihood
        t.phylogeny = self.phylogeny.copy()
        t.losses_list = copy.copy(self.losses_list)
        t.k_losses_list = copy.copy(self.k_losses_list)
        return t


    @classmethod
    def random(cls, cells, mutations, mutation_names):
        """Generates a random binary tree"""
        root = Node("germline", None, -1, 0)
        rantree = cls._new_unique_tree(mutations)

        nodes = [root]
        append_node = 0
        i = 0
        while i < mutations:
            nodes.append(Node(mutation_names[rantree[i]], nodes[append_node], rantree[i]))
            i += 1

            if i < mutations:
                nodes.append(Node(mutation_names[rantree[i]], nodes[append_node], rantree[i]))
            append_node += 1
            i += 1

        t = Tree(cells, mutations)
        t.phylogeny = root
        return t


    @classmethod
    def _new_unique_tree(cls, mutation_names):
        """
            Generate a new tree without using combinations of 3, 4 or 5
            mutations already used in other trees, in order to have the most
            random uniform generation possible.
        """
        global used_combinations
        valid = False
        mutations = [i for i in range(mutation_names)]
        attempts = 0

        while not valid:
            tree = mutations.copy()
            numpy.random.shuffle(tree)

            temp_list = []

            valid = True
            for length in [3,4,5]:
                for i in range (len(tree) - length + 1):
                    temp = tree[i : i + length]
                    temp_list.append(temp)
                    if temp in used_combinations:
                        valid = False

            if valid:
                used_combinations += temp_list
            elif attempts > 200:
                used_combinations = []
                attempts = 0
            attempts += 1

        return tree


    @classmethod
    def greedy_loglikelihood(cls, tree, matrix, cells, mutation_number, alpha, beta):
        """Gets maximum likelihood of a tree"""
        nodes_list = tree.phylogeny.get_cached_content()
        node_genotypes = [[0 for j in range(mutation_number)] for i in range(len(nodes_list))]
        for i, n in enumerate(nodes_list):
            n.get_genotype_profile(node_genotypes[i])

        lh_00 = math.log(1-beta)
        lh_10 = math.log(beta)
        lh_01 = math.log(alpha)
        lh_11 = math.log(1-alpha)

        maximum_likelihood = 0

        for i in range(cells):
            best_sigma = -1
            best_lh = float("-inf")

            for n in range(len(nodes_list)):
                lh = 0

                for j in range(mutation_number):

                    if matrix[i][j] == 0 and node_genotypes[n][j] == 0:
                        p = lh_00
                    elif matrix[i][j] == 0 and node_genotypes[n][j] == 1:
                        p = lh_01
                    elif matrix[i][j] == 1 and node_genotypes[n][j] == 0:
                        p = lh_10
                    elif matrix[i][j] == 1 and node_genotypes[n][j] == 1:
                        p = lh_11
                    elif matrix[i][j] == 2:
                        p = 0
                    else:
                        raise SystemError("Unknown value!")

                    lh += p

                if lh > best_lh:
                    best_sigma = n
                    best_lh = lh

            tree.best_sigma[i] = best_sigma
            maximum_likelihood += best_lh

        return maximum_likelihood


    @classmethod
    def greedy_loglikelihood_with_data(cls, tree, matrix, cells, mutation_number, alpha, beta, data):
        """Gets maximum likelihood of a tree, updating additional info in data"""
        nodes_list = tree.phylogeny.get_cached_content()
        node_genotypes = [
            [0 for j in range(mutation_number)]
            for i in range(len(nodes_list))
        ]
        for i, n in enumerate(nodes_list):
            n.get_genotype_profile(node_genotypes[i])

        maximum_likelihood = 0
        final_values = [0]*5

        for i in range(cells):
            best_sigma = -1
            best_lh = float("-inf")
            cell_values = [0]*5

            for n in range(len(nodes_list)):
                lh = 0
                values = [0]*5

                for j in range(mutation_number):
                    p, tmp_values = Tree._prob(matrix[i][j], node_genotypes[n][j], alpha, beta)
                    lh += math.log(p)
                    values = [sum(x) for x in zip(values, tmp_values)]

                if lh > best_lh:
                    best_sigma = n
                    best_lh = lh
                    cell_values = values

            tree.best_sigma[i] = best_sigma
            maximum_likelihood += best_lh
            final_values = [sum(x) for x in zip(final_values, cell_values)]

        data.false_positives = final_values[0]
        data.false_negatives = final_values[1]
        data.true_positives = final_values[2]
        data.true_negatives = final_values[3]
        data.missing_values = final_values[4]

        return maximum_likelihood


    @classmethod
    def _prob(cls, I, E, alpha, beta):

        fp = 0 # false positives
        fn = 0 # false negatives
        tp = 0 # true positives
        tn = 0 # true negatives
        missing = 0 # missing values

        p = 0
        if I == 0:
            if E == 0:
                tn += 1
                p = 1 - beta
            elif E == 1:
                fn += 1
                p = alpha
            else:
                raise SystemError("Unknown value for E: %d" % E)
        elif I == 1:
            if E == 0:
                fp += 1
                p = beta
            elif E == 1:
                tp += 1
                p = 1 - alpha
            else:
                raise SystemError("Unknown value for E: %d" % E)
        elif I == 2:
            missing += 1
            p = 1
        else:
            raise SystemError("Unknown value for I: %d" % I)
        return p, [fp, fn, tp, tn, missing]
