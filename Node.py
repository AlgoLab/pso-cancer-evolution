from ete3 import Tree
from graphviz import Source
import numpy
import json
import math

class Node(Tree):

    def _get_uid(self):
        return hex(id(self))

    def __init__(self, name, parent, mutation_id, loss=False):
        self.mutation_id = mutation_id
        self.loss = loss
        super().__init__(newick=None, name=name)
        if parent: # automatically add this node to its parent on creation
            parent.add_child(self)


    uid = property(fget=_get_uid)


    def __str__(self):
        return str(self.name) + ("-" if self.loss else "")

    def __repr__(self):
        return str(self.name) + ("-" if self.loss else "")


    def delete_node(self, tree):
        """Delete a node from the given tree"""
        if self in tree.losses_list:
            tree.losses_list.remove(self)
            tree.k_losses_list[self.mutation_id] -= 1
            self.delete(prevent_nondicotomic=False)


    def is_mutation_already_lost(self, mutation_id):
        """Checks if mutation is already lost in the current tree"""
        for par in self.iter_ancestors():
            if par.loss and par.mutation_id == mutation_id:
                return True
        return False


    def is_ancestor_of(self, node):
        """Checks if current node is parent of the given node"""
        par = self.up
        while par != None:
            if par.uid == node.uid:
                return True
            par = par.up
        return False


    def prune_and_reattach(self, node_reattach):
        """Detaches current node (with all its descendants) and reattaches it into another node"""
        if node_reattach.is_ancestor_of(self):
            return 1
        if node_reattach.up.uid == self.uid:
            return 1
        if self.up is None:
            return 1
        if self.uid == node_reattach.uid:
            return 1
        self.detach()
        node_reattach.add_child(self)
        return 0


    def get_height(self):
        """Returns the tree height from the current node"""
        height = 0
        for child in self.children:
            height = max(height, child.get_height())
        return height + 1


    def swap(self, node):
        """Switch this data with with that of another node"""
        self.name, node.name = node.name, self.name
        self.mutation_id, node.mutation_id = node.mutation_id, self.mutation_id
        self.loss, node.loss = node.loss, self.loss


    def get_clades(self):
        """Clades are defined as every node in the tree, excluding the root"""
        if self.mutation_id != -1:
            raise SystemError("Cannot get clades from a non-root node!")
        nodes_list = list(self.get_cached_content().keys())
        nodes_list.remove(self)
        return nodes_list


    def get_genotype_profile(self, genotypes):
        """Walks up to the root and maps the genotype for the current node mutation"""
        if not self.up:
            return
        if not self.loss:
            genotypes[self.mutation_id] += 1
        else:
            genotypes[self.mutation_id] -= 1
        self.up.get_genotype_profile(genotypes)


    def get_clade_by_distance(self, max_dist, distance):
        """
            Choose a clade, from this tree, that will be attached in another
            tree. It's chosen either randomly or after calculating the
            difference between the distance tree-tree and the average distance
            in the past clade attachments: based on that, it chooses
            the height of the clade, and randomly, it chooses which clade.
        """
        nodes = self.get_clades()

        if numpy.random.uniform() < 0.8:

            max_h = self.get_height()
            perc = (distance/max_dist)
            if perc > 1:
                perc = 1
            level = math.ceil(perc*(max_h-1))

            numpy.random.shuffle(nodes)
            for n in nodes:
                if n.get_height() == level:
                    return n

        else:
            return numpy.random.choice(nodes)


    def attach_clade(self, tree, clade):
        """Attach the clade to this tree"""
        nodes_list = self.get_tree_root().get_cached_content()
        clade_to_be_attached = clade.get_cached_content()
        clade_destination = self

        for cln in clade_to_be_attached:
            removed = []
            if cln.loss:
                if clade_destination.is_mutation_already_lost(cln.mutation_id):
                    cln.delete(prevent_nondicotomic=False)
                else:
                    tree.losses_list.append(cln)
                    tree.k_losses_list[cln.mutation_id] += 1
            else:
                for n in nodes_list:
                    if n.mutation_id != -1 and cln.mutation_id == n.mutation_id and not n.loss:
                        # moving up
                        if clade_destination == n:
                            clade_destination = n.up
                        n.delete(prevent_nondicotomic=False)
                        removed.append(n)

            for r in removed:
                nodes_list.pop(r)

        clade_destination.add_child(clade)


    def losses_fix(self, tree, mutation_number, k, max_deletions):
        """Fixes errors with losses in the given tree"""
        tree.update_losses_list()
        if tree.losses_list != []:
            families = []
            for n in self.traverse():
                if n.loss:

                    # delete loss if there are more than k of the same mutation
                    if tree.k_losses_list[n.mutation_id] > k or len(tree.losses_list) > max_deletions:
                        n.delete_node(tree)

                    # delete loss if not valid or useless
                    else:
                        genotypes = [0]*mutation_number
                        n.get_genotype_profile(genotypes)
                        if min(genotypes) < 0 or sum(genotypes) == 0:
                            n.delete_node(tree)

                        # delete loss if duplicate
                        else:
                            family = [n.mutation_id, n.up]
                            if (family in families and n.children == []):
                                n.delete_node(tree)
                            else:
                                families.append(family)


    def _to_dot_label(self, d={}):
        """
        Returns a string representing the list of properties
        indicated by d.
        Ex.: d = {
            "label": "name",
            "color": "red"
        }
        Will result in:
        [label="name",color="red"]
        """
        if not len(d):
            return ''

        out = '['
        for i, (key, value) in enumerate(d.items()):
            if isinstance(value, (int, float, complex)):
                out += '%s=%s' % (key, str(value))
            else:
                out += '%s="%s"' % (key, str(value))
            if i < len(d) - 1: # last
                out += ','
        out += ']'
        return out


    def _to_dot_node(self, nodeFromId, nodeToId=None, props={}):
        if nodeToId:
            return '\n\t"%s" -- "%s" %s;' % (nodeFromId, nodeToId, self._to_dot_label(props))
        else: # printing out single node
            return '\n\t"%s" %s;' % (nodeFromId, self._to_dot_label(props))


    def to_dot(self, root=False):
        out = ''
        if not self.up or root: # first graph node
            out += 'graph {\n\trankdir=UD;\n\tsplines=line;\n\tnode [shape=circle]'
            out += self._to_dot_node(self.uid, props={"label": self.name})
        for n in self.children:
            props = {"label": "%s" % (n.name)}
            if n.loss: # marking back-mutations
                props["color"] = "red"
            out += self._to_dot_node(n.uid, props=props)
            out += self._to_dot_node(self.uid, n.uid)
            if not n.is_leaf():
                out += n.to_dot()

        if not self.up: # first
            out += '\n}\n'
        return out


    def save(self, filename):
        Source(self.to_dot(), filename=filename, format="png").render()
