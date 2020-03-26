import copy
from Operation import Operation
from Tree import Tree

class Particle(object):

    def __init__(self, cells, mutation_number, mutation_names, number):
        # tree linked list
        self.current_tree = Tree.random(cells, mutation_number, mutation_names)
        self.number = number
        # best tree found by this particle
        self.best = self.current_tree


    def __repr__(self):
        return "bl: " + str(self.best.likelihood)
