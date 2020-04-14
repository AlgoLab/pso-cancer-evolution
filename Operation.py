from Node import Node
import random
import numpy


class Operation(object):

    BACK_MUTATION = 0
    DELETE_MUTATION = 1
    SWITCH_NODES = 2
    PRUNE_REGRAFT = 3


    @classmethod
    def tree_operation(cls, tree, operation, k, gamma, max_deletions):
        if operation == cls.BACK_MUTATION:
            return cls.add_back_mutation(tree, k, gamma, max_deletions)

        elif operation == cls.DELETE_MUTATION:
            return cls.mutation_delete(tree)

        elif operation == cls.SWITCH_NODES:
            return cls.switch_nodes(tree)

        elif operation == cls.PRUNE_REGRAFT:
            return cls.prune_regraft(tree)

        else:
            raise SystemError("Something has happened while choosing an operation")


    @classmethod
    def add_back_mutation(cls, tree, k, gamma, max_deletions):
        """Adds a new random backmutation to the given tree"""

        # gets a list of all the nodes from cache
        cached_nodes = tree.phylogeny.get_cached_content()
        keys = list(cached_nodes.keys())

        # select a random node
        # root has no parent, hence cannot add a back mutation
        # keep trying till we find a suitable node
        node = random.choice(keys)
        while node.up == None or node.up.up == None:
            node = random.choice(keys)

        # if losses list has reached its maximum, then we can't procede
        if (len(tree.losses_list) >= max_deletions):
            return 1

        # selecting possible node candidates (every ancestor)
        candidates = [p for p in node.iter_ancestors() if (p.loss == False) and (p.mutation_id != -1)]
        if len(candidates) == 0:
            return 2

        # selecting one random ancestor, based on gamma probabilities
        found = False
        while not found and len(candidates) > 0:
            candidate = random.choice(candidates)
            candidates.remove(candidate)
            if random.random() < gamma[candidate.mutation_id]:
                found = True
        if not(found):
            return 3

        # Ensuring we have no more than k mutations per mutation type
        if (tree.k_losses_list[candidate.mutation_id] >= k):
            return 4

        # If the mutation is already lost in the current tree, no way to remove it again
        if (node.is_mutation_already_lost(candidate.mutation_id)):
            return 5

        # If there are already k mutation of candidate mutation_id
        if (tree.k_losses_list[candidate.mutation_id] >= k):
            return 6

        node_deletion = Node(candidate.name, None, candidate.mutation_id, True)
        tree.losses_list.append(node_deletion)
        tree.k_losses_list[node_deletion.mutation_id] += 1

        # saving parent before detaching
        par = node.up
        current = node.detach()
        par.add_child(node_deletion)
        node_deletion.add_child(current)
        return 0


    @classmethod
    def mutation_delete(cls, tree):
        """Delete a random mutation from the given tree"""
        if (len(tree.losses_list) == 0):
            return 1
        node = random.choice(tree.losses_list)
        node.delete_node(tree)
        return 0


    @classmethod
    def switch_nodes(cls, tree):
        """Switch two random nodes of the given tree"""
        nodes = tree.phylogeny.get_cached_content()

        keys = list(nodes.keys())
        u = None
        while (u == None or u.up == None or u.loss):
            u = random.choice(keys)
            keys.remove(u)

        keys = list(nodes.keys())
        v = None
        while (v == None or v.up == None or v.loss or u.name == v.name):
            v = random.choice(keys)
            keys.remove(v)

        u.swap(v)
        return 0


    @classmethod
    def prune_regraft(cls, tree):
        """Prune-regraft two random nodes of the given tree"""
        nodes_list = tree.phylogeny.get_cached_content()

        prune_res = -1
        while prune_res != 0:

            keys = list(nodes_list.keys())
            u = None
            while (u == None or u.up == None or u.loss):
                u = random.choice(keys)
                keys.remove(u)

            keys = list(nodes_list.keys())
            v = None
            while (v == None or v.up == None or v.loss):
                v = random.choice(keys)
                keys.remove(v)

            prune_res = u.prune_and_reattach(v)

        return 0
