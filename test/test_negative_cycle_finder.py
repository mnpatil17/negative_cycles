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
        self._check_array(cycle, ['A', 'B', 'D', 'A'])
        self._check(gain, 1.003162037)

    def test_low_float_precision_pass(self):
        """
        This test requires some degree of rounding (low floating point precision) in order to pass.
        """

        node_labels = np.array(['A', 'B', 'C', 'D'])
        graph = np.array([[1.00000000e+00, 1.00000000e+00, 1.09110747e-02, 4.21229992e-02],
                          [1.00000000e+00, 1.00000000e+00, 1.87899286e+01, 7.40192450e+01],
                          [9.16500000e+01, 5.32200000e-02, 1.00000000e+00, None],
                          [2.37400000e+01, 1.35100000e-02, None, 1.00000000e+00]])

        cycle, gain = find_negative_cycle(graph, node_labels)
        self._check_array(cycle, ['B', 'C', 'A', 'B'])
        self._check(gain, 1722.0969)

    def test_high_float_precision_pass(self):
        """
        This test requires high floating point precision to pass correctly.
        """

        node_labels = np.array(['A', 'B', 'C', 'D'])
        graph = np.array([[1.0, .00037823019673030513, .0045147343459408405, .024006070586130614],
                          [2643.8925517970850, 1.0, 11.978045876628919000, 75.3012048],
                          [221.49697487717998, 0.08348607196029861, 1.0, None],
                          [41.65613011976, 0.01328, None, 1.0]])

        cycle, gain = find_negative_cycle(graph, node_labels)
        self._check_array(cycle, ['B', 'D', 'A', 'B'])
        self._check(gain, 1.18641613601)
