import numpy

a = [[1.2, 2.3],[9.2, 5.3]]

print(a)

b = [list(map(int, x)) for x in a]

print(b)

# import math
# import time
# from Helper import Helper
# from Node import Node
# from Tree import Tree
# import matplotlib.pyplot as plt
# import random
#
# def plotGraph(graph,ax,title):
#     pos=[(ii[1],ii[0]) for ii in graph.nodes()]
#     pos_dict=dict(zip(graph.nodes(),pos))
#     nx.draw(graph,pos=pos_dict,ax=ax,with_labels=True)
#     ax.set_title(title)
#     return
#
#
# mutation_number = 8
# cells = 11
#
# helper = Helper(None, None, mutation_number, [], cells, 0, 0, 0, 3, 5, 0, 100)
#
#
# a = Node('germline', None, -1, False)
# a1 = Node('1', a, 1)
# a2 = Node('2', a1, 2)
# a3 = Node('3', a1, 3)
# a1loss = Node('1', a2, 1, True)
#
# tree = Tree(cells, mutation_number)
# tree.phylogeny = a
#
# tree.update_losses_list()
#
# tree.phylogeny.save("test0")
# print(tree.losses_list)
#
# tree2 = tree.copy()
#
# tree2.phylogeny.save("test1")
# print(tree2.losses_list)
