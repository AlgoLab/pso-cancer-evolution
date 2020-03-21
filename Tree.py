from Node import Node
from Operation import Operation as Op
import random as r
import copy
import math

lista3 = [] #per la creazione random di alberi
lista4 = []
lista5 = []

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


    def calculate_losses_list(self, k):
        losses_list = []
        k_losses_list = [0] * self.mutations
        for n in self.phylogeny.traverse():
            if n.loss:
                losses_list.append(n)
                k_losses_list[n.mutation_id] += 1
        return losses_list, k_losses_list


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
    def generate_tree(cls, mutations):
        global lista3
        global lista4
        global lista5
        alfabeto = [i for i in range(mutations)]
        lunghezza_parole = 5
        tree = None
        accettabile = False

        while accettabile == False:
            tree = alfabeto.copy()
            r.shuffle(tree, r.random)

            accettabile = True
            for i in range (len(tree)-lunghezza_parole+1):
                temp = tree[i:i+lunghezza_parole]
                if temp in lista5:
                    accettabile = False

            for i in range (len(tree)-lunghezza_parole+2):
                temp = tree[i:i+lunghezza_parole-1]
                if temp in lista4:
                    accettabile = False

            for i in range (len(tree)-lunghezza_parole+3):
                temp = tree[i:i+lunghezza_parole-2]
                if temp in lista3:
                    accettabile = False

            if accettabile == True:
                for i in range (len(tree)-lunghezza_parole+1):
                    lista5.append(tree[i:i+lunghezza_parole])
                for i in range (len(tree)-lunghezza_parole+2):
                    lista4.append(tree[i:i+lunghezza_parole-1])
                for i in range (len(tree)-lunghezza_parole+3):
                    lista3.append(tree[i:i+lunghezza_parole-2])

        return tree


    @classmethod
    def random(cls, cells, mutations, mutation_names):
        t = Tree(cells, mutations)
        random_tree = cls._generate_random_btree(mutations, mutation_names)
        t.phylogeny = random_tree
        return t


    @classmethod
    def germline_node(cls):
        return Node("germline", None, -1, 0)


    @classmethod
    def _generate_random_btree(cls, mutations, mutation_names):
        """ Generates a random binary tree """
        root = cls.germline_node()
        rantree = cls.generate_tree(mutations)
        # print(rantree)

        nodes = [root]
        append_node = 0
        i = 0
        while i < mutations:
            nodes.append(
                Node(mutation_names[rantree[i]], nodes[append_node], rantree[i])
            )
            i += 1

            if i < mutations:
                nodes.append(
                    Node(mutation_names[rantree[i]], nodes[append_node], rantree[i])
                )
            append_node += 1
            i += 1

        return root


    @classmethod
    def greedy_loglikelihood(cls, helper, tree, data=None):
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

        for i in range(helper.cells):
            best_sigma = -1
            best_lh = float("-inf")

            for n in range(len(nodes_list)):
                lh = 0
                for j in range(helper.mutation_number):
                    p = Op.prob(helper.matrix[i][j], node_genotypes[n][j], node_genotypes, helper, tree, data)
                    lh += math.log(p)

                if lh > best_lh:
                    best_sigma = n
                    best_lh = lh

            tree.best_sigma[i] = best_sigma
            maximum_likelihood += best_lh

        return maximum_likelihood
