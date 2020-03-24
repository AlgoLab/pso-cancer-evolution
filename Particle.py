import copy
from Operation import Operation
from Tree import Tree

class Particle(object):

    def __init__(self, cells, mutations, mutation_names, number):
        # tree linked list
        self.current_tree = Tree.random(cells, mutations, mutation_names)
        self.number = number
        # best tree found by this particle
        self.best = self.current_tree
        self.velocity = 3


    def __repr__(self):
        return "bl: " + str(self.best.likelihood)
