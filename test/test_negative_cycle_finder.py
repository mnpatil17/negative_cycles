import unittest
import numpy as np
from negative_cycles import find_negative_cycle


class TestNegativeCycles(unittest.TestCase):

    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def _check_eq(self, actual, expected):
        assert actual == expected, 'Two expected {0}, but got {1}'.format(expected, actual)

    def _check_none(self, actual):
        np.testing.assert_equal(actual, None)

    def _check_array(self, actual, expected):
        np.testing.assert_array_equal(actual, expected)

    def test_small_graph(self):

        node_labels = np.array(['A', 'B', 'C'])
        graph = np.eye(len(node_labels))

        # A to B and back
        graph[0][1] = 1 / 1.5
        graph[1][0] = 1.5

        # A to C and back
        graph[0][2] = 10.0
        graph[2][0] = 1 / 10.0

        # B to C
        graph[1][2] = 16.0
        graph[2][1] = 1 / 16.0

        cycle, gain = find_negative_cycle(graph, node_labels)
        self._check_array(cycle, ['A', 'B', 'C', 'A'])
        self._check(gain, 1.0666666666666667)

    def test_four_node_graph(self):

        node_labels = np.array(['A', 'B', 'C', 'D'])
        graph = np.eye(len(node_labels))

        # A to B and back
        graph[0][1] = 1 / 1.09
        graph[1][0] = 1.09

        # A to C and back
        graph[0][2] = 6.48
        graph[2][0] = 1 / 6.48

        # B to C
        graph[1][2] = 7.08
        graph[2][1] = 1 / 7.08

        # A to D and back
        graph[0][3] = 66.027
        graph[3][0] = 1 / 66.027

        # B to D and back
        graph[1][3] = 72.197
        graph[3][1] = 1 / 72.197

        # C to D and back
        graph[2][3] = 10.191
        graph[3][2] = 1 / 10.191

        cycle, gain = find_negative_cycle(graph, node_labels)
        self._check_none(cycle)
        self._check_none(gain)

    def test_live_example_graph(self):

        node_labels = np.array(['A', 'B', 'C', 'D'])
        graph = np.array([[1.00000000e+00, 1.00000000e+00, 1.09110747e-02, 4.21229992e-02],
                          [1.00000000e+00, 1.00000000e+00, 1.87899286e+01, 7.40192450e+01],
                          [9.16500000e+01, 5.32200000e-02, 1.00000000e+00, None],
                          [2.37400000e+01, 1.35100000e-02, None, 1.00000000e+00]])

        cycle, gain = find_negative_cycle(graph, node_labels)
        self._check_array(cycle, ['B', 'C', 'A', 'B'])
        self._check(gain, 1722.0969)
