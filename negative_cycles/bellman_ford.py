#
# bellman_ford.py
#

import numpy as np


#
# Multiplicative Bellman Ford
#

def multiplicative_bellman_ford(graph):
    return bellman_ford(graph, is_multiplicative=True)


def multiplicative_bellman_ford_with_term_status(graph):
    return bellman_ford_with_term_status(graph, is_multiplicative=True)


#
# Additive Bellman Ford
#

def additive_bellman_ford(graph):
    return bellman_ford(graph, is_multiplicative=False)


def additive_bellman_ford_with_term_status(graph):
    return bellman_ford_with_term_status(graph, is_multiplicative=False)

#
# Agnostic Bellman Ford
#


def bellman_ford(graph, is_multiplicative=False):
    dist, pred, _ = bellman_ford_with_term_status(graph, is_multiplicative=is_multiplicative)
    return dist, pred


def bellman_ford_with_term_status(graph, is_multiplicative=False):
    """
    An implementation of the multiplication-based Bellman-Ford algorithm.

    :param: graph - The graph on which to operate. Should be a square matrix, where edges that don't
                    exist have value None
    :param: graph_labels - An ordered list of labels that correspond to the indices in the input
                           graph.
    :param: is_multiplicative - If this is True, performs multiplication-based Bellman Ford, where
                                the distances between two nodes are based on the smallest PRODUCT of
                                the edge weights between them. If it is False, then performs
                                addition-based Bellman Ford, where the distances between two nodes
                                are based on the SUM of the edge weights between them.

    :return: a tuple, where the zero-th item is the distance array output from the Bellman-Ford
             Algorithm, as well as the predecessor array to find paths
    """

    # print '[{0}] Entered Bellman Ford'.format(multiprocessing.current_process().pid)
    operator = (lambda x, y: x * y) if is_multiplicative else (lambda x, y: x + y)

    # Create a distance array with value infinity
    distance = np.zeros(len(graph)).astype(np.float128)
    distance.fill(float('inf'))
    distance[0] = 1.0 if is_multiplicative else 0.0
    prev_distance = list(distance)

    # Create a predecessor array with value None
    predecessor = np.zeros(len(graph))
    predecessor.fill(-1)

    # print '[{0}] Initialized Bellman Ford'.format(multiprocessing.current_process().pid)

    for _ in range(len(graph) - 1):

        # Iterate through all the vertices
        for i, node_a_weights in enumerate(graph):
            for j, weight in enumerate(node_a_weights):

                if weight is None:
                    continue

                new_dist = operator(distance[i], weight)

                if new_dist - distance[j] < -1.0e-8:  # Accounts for floating-pt error.
                    distance[j] = new_dist
                    predecessor[j] = i

        # Check for early termination
        if np.all(distance == prev_distance):
            return distance, predecessor, True
        prev_distance = list(distance)

    return distance, predecessor, False
