#
# find_negative_cycle.py
# An implementation of the multiplication-based Bellman-Ford Algorithm for negative-cycle finding
# Author: Mihir Patil
#

import numpy as np
from bellman_ford import multiplicative_bellman_ford
from utils import get_prec


def find_negative_cycle(graph, graph_labels):
    """
    Finds negative cycles in the specified graph

    :param: graph The graph on which to operate.
    :param: graph_labels A ordered list of labels that correspond to the indices in the input graph.

    :return: If a negative cycle is found, returns a tuple with the negative cycle as a list,
             followed by the gain of pursuing the negative cycle. If no negative cycle is found,
             returns (None, None).
    """

    distance, predecessor = multiplicative_bellman_ford(graph)

    for i, node_a_weights in enumerate(graph):
        for j, weight in enumerate(node_a_weights):

            if weight is None:
                continue

            precision = min(get_prec(distance[i]), get_prec(weight), get_prec(distance[j]))

            dt_j = distance[j] if distance[j] == float('inf') else round(distance[j], precision)
            if round(distance[i] * weight, precision) < round(dt_j, precision):
                negative_cycle = []
                negative_cycle.append(graph_labels[i])

                prev = i
                curr = int(predecessor[i])
                gain = np.float128(graph[prev][curr])

                while curr != i:
                    negative_cycle.append(graph_labels[int(curr)])
                    prev = curr
                    curr = int(predecessor[curr])
                    gain *= np.float128(graph[prev][curr])

                negative_cycle.append(graph_labels[i])
                return negative_cycle, gain
    return None, None
