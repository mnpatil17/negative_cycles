#
# bellman_ford.py
#

import numpy as np
from utils import get_prec, is_int


def multiplicative_bellman_ford(graph):
    return bellman_ford(graph, is_multiplicative=True)


def additive_bellman_ford(graph):
    return bellman_ford(graph, is_multiplicative=False)


def bellman_ford(graph, is_multiplicative=False):
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
    operator = (lambda x, y: x * y) if is_multiplicative else (lambda x, y: x + y)
    lowest_precision = float('inf')

    # Create a distance array with value infinity
    distance = np.zeros(len(graph)).astype(np.float64)
    distance.fill(float('inf'))
    distance[0] = 1.0 if is_multiplicative else 0

    # Create a predecessor array with value None
    predecessor = np.zeros(len(graph))
    predecessor.fill(-1)

    for _ in range(len(graph) - 1):

        # Iterate through all the vertices
        for i, node_a_weights in enumerate(graph):
            for j, weight in enumerate(node_a_weights):

                if weight is None:
                    continue

                # Handle precision development
                precision = min(get_prec(distance[i]), get_prec(weight), get_prec(distance[j]))
                lowest_precision = min(lowest_precision, precision)

                # Find the correct precision, compensating for infinity
                dt_j = distance[j] if distance[j] == float('inf') or is_int(distance[j]) \
                    else round(distance[j], precision)

                raw_new_dist = operator(distance[i], weight)
                new_dist = raw_new_dist if is_int(raw_new_dist) else round(raw_new_dist, precision)

                if new_dist < dt_j:
                    distance[j] = new_dist
                    predecessor[j] = i

    return distance, predecessor