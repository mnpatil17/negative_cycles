import unittest
import numpy as np
from negative_cycles import additive_bellman_ford


class TestAdditiveBellmanFord(unittest.TestCase):

    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def _check_eq(self, actual, expected):
        assert actual == expected, 'Two expected {0}, but got {1}'.format(expected, actual)

    def _check_none(self, actual):
        np.testing.assert_equal(actual, None)

    def _check_array(self, actual, expected):
        np.testing.assert_array_equal(actual, expected)

    def test_small_graph(self):

        graph = np.eye(3)

        # A to B and back
        graph[0][1] = 1
        graph[1][0] = 2

        # A to C and back
        graph[0][2] = 7
        graph[2][0] = 1

        # B to C and back
        graph[1][2] = 3
        graph[2][1] = 4

        distance, predecessor = additive_bellman_ford(graph)
        expected_distance = np.array([0, 1, 4])
        expected_predecessor = np.array([-1, 0, 1])

        self._check_array(distance, expected_distance)
        self._check_array(predecessor, expected_predecessor)

    def test_four_node_graph_with_negative_weight(self):

        graph = np.eye(4)

        # A to B and back
        graph[0][1] = 5
        graph[1][0] = 7

        # A to C and back
        graph[0][2] = 10
        graph[2][0] = 2

        # B to C and back
        graph[1][2] = 4
        graph[2][1] = 1

        # A to D and back
        graph[0][3] = 5
        graph[3][0] = 3

        # B to D and back
        graph[1][3] = 1
        graph[3][1] = -1

        # C to D and back
        graph[2][3] = 1
        graph[3][2] = 7

        distance, predecessor = additive_bellman_ford(graph)
        expected_distance = np.array([0, 4, 8, 5])
        expected_predecessor = np.array([-1, 3, 1, 0])

        self._check_array(distance, expected_distance)
        self._check_array(predecessor, expected_predecessor)
