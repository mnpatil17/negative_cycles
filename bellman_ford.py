#
# bellman_ford.py
#

import numpy as np
from negative_cycles.utils import get_prec


def multiplicative_bellman_ford(graph, graph_labels):
    """
    An implementation of the multiplication-based Bellman-Ford algorithm.

    :param: graph The graph on which to operate.
    :param: graph_labels A ordered list of labels that correspond to the indices in the input graph.
    :return: a tuple, where the zero-th item is the distance array output from the Bellman-Ford
             Algorithm, as well as the predecessor array to find paths
    """

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
        for i, curr_a_conversions in enumerate(graph):
            for j, weight in enumerate(curr_a_conversions):

                if weight is None:
                    continue

                # Handle precision development
                precision = min(get_prec(distance[i]), get_prec(weight), get_prec(distance[j]))
                lowest_precision = min(lowest_precision, precision)

                # Find the correct precision, compensating for infinity
                dt_j = distance[j] if distance[j] == float('inf') else round(distance[j], precision)
                if round(distance[i] * weight, precision) < dt_j:
                    distance[j] = round(distance[i] * weight, precision)
                    predecessor[j] = i

    return distance, predecessor