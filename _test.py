import math
import time
from Helper import Helper
from Node import Node
from Tree import Tree
import matplotlib.pyplot as plt
import networkx as nx
import random

def plotGraph(graph,ax,title):
    pos=[(ii[1],ii[0]) for ii in graph.nodes()]
    pos_dict=dict(zip(graph.nodes(),pos))
    nx.draw(graph,pos=pos_dict,ax=ax,with_labels=True)
    ax.set_title(title)
    return


mutation_number = 8
cells = 11

helper = Helper(None, mutation_number, [], cells, 0, 0, 0, 3, 10, 100)


a = Node('germline', None, -1, False)
a1 = Node('1', a, 1)
a2 = Node('2', a1, 2)
a3 = Node('3', a1, 3)
a1loss = Node('1', a2, 1, True)
a4 = Node('4', a2, 4)
a5 = Node('5', a3, 5)
a6 = Node('6', a3, 6)
a7 = Node('7', a1loss, 7)
a6loss = Node('6', a6, 6, True)
a8 = Node('8', a6, 8)
a9 = Node('9', a7, 9)
a10 = Node('10', a8, 10)
a11 = Node('11', a9, 11)
a12 = Node('12', a10, 12)
a13 = Node('13', a11, 13)
a14 = Node('14', a12, 14)
a15 = Node('15', a13, 15)
a16 = Node('16', a14, 16)

tree = Tree(cells, mutation_number)
tree.phylogeny = a

# a.update_losses_list(helper, tree)
# print(tree.losses_list)


nodes = list(a.get_cached_content().keys())


time1 = time.time()
max_h = max([n.get_height() for n in nodes])
time2 = time.time()
print("old: "+str(max_h)+" -> "+str(time2 - time1))



time1 = time.time()
max_h = a.get_height()
time2 = time.time()
print("new: "+str(max_h)+" -> "+str(time2 - time1))






a.save("test0")

# for i in range(1,10):
#     clade_to_be_attached = random.choice(a.get_clades())
#     # print("clade_to_be_attached: "+str(clade_to_be_attached))
#     clade_to_be_attached = clade_to_be_attached.copy().detach()
#     clade_destination = random.choice(a.get_clades())
#     # print("clade_destination:    "+str(clade_destination))
#     clade_destination.attach_clade_and_fix(helper, tree, clade_to_be_attached)
#     tree.phylogeny.fix_useless_losses(helper, tree)
#
#     a.save("test"+str(i))
#     print(i)
#     print(tree.losses_list)
#     print(tree.k_losses_list)
#     print("")


# time1 = time.time()
# a.fix_for_losses(helper, tree)
# time2 = time.time()
# print(100 * (time2 - time1))
# a.save("test2")
