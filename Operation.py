from Node import Node
import random
import numpy

def accept(currentIteration, iterations):
    return random.random() < (currentIteration / iterations)

class Operation(object):

    BACK_MUTATION = 0
    DELETE_MUTATION = 1
    SWITCH_NODES = 2
    PRUNE_REGRAFT = 3



    @classmethod
    def tree_operation(cls, helper, tree, operation):
        if operation == cls.BACK_MUTATION:
            return cls.add_back_mutation(helper, tree) # add a new backmutation

        elif operation == cls.DELETE_MUTATION:
            return cls.mutation_delete(helper, tree) # delete a random mutation

        elif operation == cls.SWITCH_NODES:
            return cls.switch_nodes(helper, tree) # switch two random nodes

        elif operation == cls.PRUNE_REGRAFT:
            return cls.prune_regraft(helper, tree) # prune-regraft two random nodes

        else:
            raise SystemError("Something has happened while choosing an operation")



    @classmethod
    def add_back_mutation(cls, helper, tree):

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
        if (len(tree.losses_list) >= helper.max_deletions):
            return 1

        # selecting possible node candidates (every ancestor)
        candidates = [p for p in node.iter_ancestors() if (p.loss == False) and (p.mutation_id != -1)]
        if len(candidates) == 0:
            return 1

        # selecting one random ancestor, based on gamma probabilities
        found = False
        while not found and len(candidates) > 0:
            candidate = random.choice(candidates)
            candidates.remove(candidate)
            if random.random() < helper.gamma[candidate.mutation_id]:
                found = True
        if not(found):
            return 1

        # Ensuring we have no more than k mutations per mutation type
        if (tree.k_losses_list[candidate.mutation_id] >= helper.k):
            return 1

        # If the mutation is already lost in the current tree, no way to remove it again
        if (node.is_mutation_already_lost(candidate.mutation_id)):
            return 1

        # If there are already k mutation of candidate mutation_id
        if (tree.k_losses_list[candidate.mutation_id] >= helper.k):
            return 1

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
    def mutation_delete(cls, helper, tree):
        if (len(tree.losses_list) == 0):
            return 1
        node_delete = random.choice(tree.losses_list)
        node_delete.delete_b(helper, tree)
        return 0



    @classmethod
    def switch_nodes(cls, helper, tree):
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
    def prune_regraft(cls, helper, tree):
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



    @classmethod
    def prob(cls, I, E, alpha, beta):

        fp = 0 # false positives
        fn = 0 # false negatives
        tp = 0 # true positives
        tn = 0 # true negatives
        missing = 0 # missing values

        p = 0
        if I == 0:
            if E == 0:
                tn += 1
                p = 1 - beta
            elif E == 1:
                fn += 1
                p = alpha
            else:
                raise SystemError("Unknown value for E: %d" % E)
        elif I == 1:
            if E == 0:
                fp += 1
                p = beta
            elif E == 1:
                tp += 1
                p = 1 - alpha
            else:
                raise SystemError("Unknown value for E: %d" % E)
        elif I == 2:
            missing += 1
            p = 1
        else:
            raise SystemError("Unknown value for I: %d" % I)
        return p, [fp, fn, tp, tn, missing]
