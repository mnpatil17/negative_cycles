#
# bellman_ford.py
#

import numpy as np
from negative_cycles.utils import get_prec


def multiplicative_bellman_ford(graph, graph_labels):
    return bellman_ford(graph, graph_labels, is_multiplicative=True)


def additive_bellman_ford(graph, graph_labels):
    return bellman_ford(graph, graph_labels, is_multiplicative=False)


def bellman_ford(graph, graph_labels, is_multiplicative=False):
    """
    An implementation of the multiplication-based Bellman-Ford algorithm.

    :param: graph - The graph on which to operate.
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
    operator = lambda x, y: x * y if is_multiplicative else lambda x, y: x + y
    lowest_precision = float('inf')

    # Create a distance array with value infinity
    distance = np.zeros(len(graph_labels)).astype(np.float64)
    distance.fill(float('inf'))
    distance[0] = 1.0

    # Create a predecessor array with value None
    predecessor = np.zeros(len(graph_labels))
    predecessor.fill(-1)

    for _ in range(len(graph_labels) - 1):

        # Iterate through all the vertices
        for i, node_a_weights in enumerate(graph):
            for j, weight in enumerate(node_a_weights):

                if weight is None:
                    continue

                # Handle precision development
                precision = min(get_prec(distance[i]), get_prec(weight), get_prec(distance[j]))
                lowest_precision = min(lowest_precision, precision)

                # Find the correct precision, compensating for infinity
                dt_j = distance[j] if distance[j] == float('inf') else round(distance[j], precision)
                if round(operator(distance[i], weight), precision) < dt_j:
                    distance[j] = round(operator(distance[i], weight), precision)
                    predecessor[j] = i

    return distance, predecessor