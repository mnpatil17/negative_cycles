#
# find_negative_cycle.py
# An implementation of the multiplication-based Bellman-Ford Algorithm for negative-cycle finding
# Author: Mihir Patil
#


import numpy as np


def find_negative_cycle(graph, currency_list):
    """
    An implementation of the multiplication-based Bellman-Ford algorithm for curreny exchanges.

    :param: graph The currency graph on which to operate. The currency at index 0 is the starting
                  currency.
    :param: currency_list A ordered list of currency names whose indices correspond to their indices
                          in the input graph. The currency at index 0 is the starting currency.
    :return: If a negative cycle is found, returns a tuple with the negative cycle as a list,
             followed by the gain of pursuing the negative cycle. If no negative cycle is found,
             returns (None, None).
    """

    lowest_precision = float('inf')

    # Create a distance array with value infinity
    distance = np.zeros(len(currency_list)).astype(np.float64)
    distance.fill(float('inf'))
    distance[0] = 1.0

    # Create a predecessor array with value None
    predecessor = np.zeros(len(currency_list))
    predecessor.fill(-1)

    for _ in range(len(currency_list) - 1):

        # Iterate through all the vertices
        for i, curr_a_conversions in enumerate(graph):
            for j, conversion in enumerate(curr_a_conversions):

                if conversion is None:
                    continue

                # Handle precision development
                precision = min(get_prec(distance[i]), get_prec(conversion), get_prec(distance[j]))
                lowest_precision = min(lowest_precision, precision)

                # Find the correct precision, compensating for infinity
                dt_j = distance[j] if distance[j] == float('inf') else round(distance[j], precision)
                if round(distance[i] * conversion, precision) < dt_j:
                    distance[j] = round(distance[i] * conversion, precision)
                    predecessor[j] = i


    for i, curr_a_conversions in enumerate(graph):
        for j, conversion in enumerate(curr_a_conversions):

            if conversion is None:
                continue

            precision = min(get_prec(distance[i]), get_prec(conversion), get_prec(distance[j]))

            dt_j = distance[j] if distance[j] == float('inf') else round(distance[j], precision)
            if round(distance[i] * conversion, precision) < round(dt_j, precision):
                gain = 1.0
                negative_cycle = []
                negative_cycle.append(currency_list[i])

                prev = i
                curr = int(predecessor[i])
                gain *= graph[prev][curr]

                while curr != i:
                    negative_cycle.append(currency_list[int(curr)])
                    prev = curr
                    curr = int(predecessor[curr])
                    gain *= graph[prev][curr]

                negative_cycle.append(currency_list[i])
                return negative_cycle, gain


    return None, None


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