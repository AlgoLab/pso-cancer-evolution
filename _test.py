


# import math
# import time
# from Helper import Helper
# from Node import Node
# from Tree import Tree
# import matplotlib.pyplot as plt
# import networkx as nx
#
# def plotGraph(graph,ax,title):
#     pos=[(ii[1],ii[0]) for ii in graph.nodes()]
#     pos_dict=dict(zip(graph.nodes(),pos))
#     nx.draw(graph,pos=pos_dict,ax=ax,with_labels=True)
#     ax.set_title(title)
#     return
#
# helper = Helper(None, 10, [], 10, 0, 0, 0, 3, 1, 1, 1, 10)
#
#
# a = Node('germline', None, -1, False)
# a1 = Node('1', a, 0)
# al1 = Node('1', a1, 0, True)
# al2 = Node('1', a1, 0, True)
# al3 = Node('1', a1, 0, True)
# a2 = Node('2', a1, 1)
# a3 = Node('3', a2, 2)
#
# tree = Tree(10, 10)
# tree.phylogeny = a
#
# a.update_losses_list(helper, tree)
# print(tree.losses_list)
#
# a.save("test1")
# time1 = time.time()
# a.fix_for_losses(helper, tree)
# time2 = time.time()
# print(100 * (time2 - time1))
# a.save("test2")
