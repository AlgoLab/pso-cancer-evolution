from Node import Node
from Operation import Operation as Op
import random
import copy
import math

# for a uniform random trees generation
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
        self.operation = None
        self.debug = False


    def update_losses_list(self):
        ll = []
        kll = [0] * self.mutations
        for n in self.phylogeny.traverse():
            if n.loss:
                ll.append(n)
                kll[n.mutation_id] += 1
        self.losses_list = ll
        self.k_losses_list = kll


    def copy(self):
        "Copies everything in this tree"
        t = Tree(self.cells, self.mutations)
        t.likelihood = self.likelihood
        t.phylogeny = self.phylogeny.copy()
        for n in t.phylogeny.traverse():
            if n.loss:
                t.losses_list.append(n)
                t.k_losses_list[n.mutation_id] += 1
        if self.operation == None:
            t.operation = None
        else:
            t.operation = Op(self.operation.type, self.operation.node_name_1, self.operation.node_name_2, self.operation.node_name_3)
        return t



    @classmethod
    def random(cls, cells, mutations, mutation_names):
        t = Tree(cells, mutations)
        random_tree = cls._generate_random_btree(mutations, mutation_names)
        t.phylogeny = random_tree
        return t



    @classmethod
    def _generate_random_btree(cls, mutations, mutation_names):
        """ Generates a random binary tree """
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

        return root



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
            random.shuffle(tree, random.random)

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
    def greedy_loglikelihood_with_data(cls, helper, tree, data):
        "Gets maximum likelihood of a tree"

        nodes_list = tree.phylogeny.get_cached_content()
        node_genotypes = [
            [0 for j in range(helper.mutation_number)]
            for i in range(len(nodes_list))
        ]
        for i, n in enumerate(nodes_list):
            n.get_genotype_profile(node_genotypes[i])

        # per ogni nodo, viene creato un array con tutte le mutazioni presenti, che vengono ereditate dai figli
        #   es. nodi nell'albero:       {c: {c},       b: {b},       d: {c, b},    a: {a},       germline: {c, a, b}}
        #       mutazioni dei nodi:     [[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]]

        maximum_likelihood = 0
        final_values = [0]*5

        for i in range(helper.cells):
            best_sigma = -1
            best_lh = float("-inf")
            best_values = [0]*5

            for n in range(len(nodes_list)):
                lh = 0
                values = [0]*5

                for j in range(helper.mutation_number):
                    p, tmp_values = Op.prob(helper.matrix[i][j], node_genotypes[n][j], helper.alpha, helper.beta)
                    lh += math.log(p)
                    values = [sum(x) for x in zip(values, tmp_values)]

                if lh > best_lh:
                    best_sigma = n
                    best_lh = lh
                    best_values = values

            tree.best_sigma[i] = best_sigma
            maximum_likelihood += best_lh
            final_values = [sum(x) for x in zip(final_values, best_values)]

        data.false_positives = final_values[0]
        data.false_negatives = final_values[1]
        data.true_positives = final_values[2]
        data.true_negatives = final_values[3]
        data.missing_values = final_values[4]

        return maximum_likelihood



    @classmethod
    def greedy_loglikelihood(cls, helper, tree):
        nodes_list = tree.phylogeny.get_cached_content()
        node_genotypes = [[0 for j in range(helper.mutation_number)] for i in range(len(nodes_list))]
        for i, n in enumerate(nodes_list):
            n.get_genotype_profile(node_genotypes[i])

        lh_00 = math.log(1-helper.beta)
        lh_10 = math.log(helper.beta)
        lh_01 = math.log(helper.alpha)
        lh_11 = math.log(1-helper.alpha)

        maximum_likelihood = 0

        for i in range(helper.cells):
            best_sigma = -1
            best_lh = float("-inf")

            for n in range(len(nodes_list)):
                lh = 0

                for j in range(helper.mutation_number):

                    I = helper.matrix[i][j]
                    E = node_genotypes[n][j]

                    if I == 0 and E == 0:
                        p = lh_00
                    elif I == 0 and E == 1:
                        p = lh_01
                    elif I == 1 and E == 0:
                        p = lh_10
                    elif I == 1 and E == 1:
                        p = lh_11
                    elif I == 2:
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
