#
# find_negative_cycle.py
# An implementation of the multiplication-based Bellman-Ford Algorithm for negative-cycle finding
# Author: Mihir Patil
#

import numpy as np


def find_negative_cycle(graph, graph_labels):
    """
    Finds negative cycles in the specified graph

    :param: graph The graph on which to operate.
    :param: graph_labels A ordered list of labels that correspond to the indices in the input graph.

    :return: If a negative cycle is found, returns a tuple with the negative cycle as a list,
             followed by the gain of pursuing the negative cycle. If no negative cycle is found,
             returns (None, None).
    """

    distance, predecessor = multiplicative_bellman_ford(graph, graph_labels)

    for i, curr_a_conversions in enumerate(graph):
        for j, weight in enumerate(curr_a_conversions):

            if weight is None:
                continue

            precision = min(get_prec(distance[i]), get_prec(weight), get_prec(distance[j]))

            dt_j = distance[j] if distance[j] == float('inf') else round(distance[j], precision)
            if round(distance[i] * weight, precision) < round(dt_j, precision):
                gain = 1.0
                negative_cycle = []
                negative_cycle.append(graph_labels[i])

                prev = i
                curr = int(predecessor[i])
                gain *= graph[prev][curr]

                while curr != i:
                    negative_cycle.append(graph_labels[int(curr)])
                    prev = curr
                    curr = int(predecessor[curr])
                    gain *= graph[prev][curr]

                negative_cycle.append(graph_labels[i])
                return negative_cycle, gain
    return None, None


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


def get_prec(val):
    """
    Gets the precision of the val passed in.

    :param: val The number to get the precision of
    """
    if np.isnan(val) or val == float('inf') or int(val) == val:
        return 9999999

    split_arr_dot = str(val).split('.')
    if len(split_arr_dot) == 2:
        return len(split_arr_dot[1])
    else:
        split_arr_e = str(val).split('e')
        exp = int(split_arr_e[1])
        return -exp if exp < 0 else 0