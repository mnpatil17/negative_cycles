#
# find_negative_cycle.py
# An implementation of the multiplication-based Bellman-Ford Algorithm for negative-cycle finding
# Author: Mihir Patil
#

import numpy as np
from bellman_ford import multiplicative_bellman_ford_with_term_status


def find_negative_cycle(graph, graph_labels):
    """
    Finds negative cycles in the specified graph

    :param: graph The graph on which to operate.
    :param: graph_labels A ordered list of labels that correspond to the indices in the input graph.

    :return: If a negative cycle is found, returns a tuple with the negative cycle as a list,
             followed by the gain of pursuing the negative cycle. If no negative cycle is found,
             returns (None, None).
    """

    distance, predecessor, terminated_early = multiplicative_bellman_ford_with_term_status(graph)
    if terminated_early:
        return None, None

    for i, node_a_weights in enumerate(graph):
        for j, weight in enumerate(node_a_weights):

            if weight is None:
                continue

            if distance[i] * weight - distance[j] < -1.0e-8:  # Accounts for floating-pt error.
                negative_cycle = []
                negative_cycle.append(graph_labels[i])

                prev = i
                curr = int(predecessor[i])
                gain = np.float128(graph[prev][curr])

                count = 0
                try:
                    while curr != i:
                        negative_cycle.append(graph_labels[curr])
                        prev = curr
                        curr = int(predecessor[curr])
                        gain *= np.float128(graph[prev][curr])
                        count += 1
                except MemoryError as e:
                    print 'Graph: {0}'.format(graph)
                    print 'Predecessor: {0}'.format(predecessor)
                    print 'Cycle[:10]: {0}'.format(negative_cycle[:10])
                    print 'distance[i] * weight: {0}'.format(distance[i] * weight)
                    print 'distance[j]: {0}'.format(distance[j])
                    print 'Count: {0}'.format(count)
                    print 'Gain: {0}'.format(gain)
                    print 'Distance: {0}'.format(distance)
                    print e
                    return None, None

                negative_cycle.append(graph_labels[i])
                return negative_cycle, gain

    return None, None
