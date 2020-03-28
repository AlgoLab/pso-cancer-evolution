import copy
from Operation import Operation
from Tree import Tree

class Particle(object):

    def __init__(self, cells, mutation_number, mutation_names, number, starting_tree=None):
        # tree linked list
        if starting_tree is None:
            self.current_tree = Tree.random(cells, mutation_number, mutation_names)
        else:
            self.current_tree = starting_tree
        self.number = number
        # best tree found by this particle
        self.best = self.current_tree


    def __repr__(self):
        return "bl: " + str(self.best.likelihood)
