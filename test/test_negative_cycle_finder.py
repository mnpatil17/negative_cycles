import unittest
import numpy as np
from find_negative_cycle import find_negative_cycle


class TestBelmanFord(unittest.TestCase):

    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)


    def _check_eq(self, actual, expected):
        assert actual == expected, 'Two expected {0}, but got {1}'.format(expected, actual)


    def _check_none(self, actual):
        np.testing.assert_equal(actual, None)


    def _check_array(self, actual, expected):
        np.testing.assert_array_equal(actual, expected)


    def test_small_graph(self):

        currency_list = np.array(["Dollar", "Euro", "Yuan"])
        graph = np.eye(len(currency_list))

        # Dollar to Euro and back
        graph[0][1] = 1 / 1.5
        graph[1][0] = 1.5       # one euro is 1.5 dollars

        # Dollar to Yuan and back
        graph[0][2] = 10.0      # one dollar is 10 yuan
        graph[2][0] = 1 / 10.0

        # Euro to Yuan
        graph[1][2] = 16.0      # one euro is 16 yuan
        graph[2][1] = 1 / 16.0

        cycle, gain = find_negative_cycle(graph, currency_list)
        self._check_array(cycle, ['Dollar', 'Euro', 'Yuan', 'Dollar'])
        self._check(gain, 1.0666666666666667)


    def test_four_currency_graph(self):

        currency_list = np.array(["Dollar", "Euro", "Yuan", "Rupee"])
        graph = np.eye(len(currency_list))

        # Dollar to Euro and back
        graph[0][1] = 1 / 1.09
        graph[1][0] = 1.09       # one euro is 1.09 dollars

        # Dollar to Yuan and back
        graph[0][2] = 6.48      # one dollar is 6.48 yuan
        graph[2][0] = 1 / 6.48

        # Euro to Yuan
        graph[1][2] = 7.08      # one euro is 7.08 yuan
        graph[2][1] = 1 / 7.08

        # Dollar to Rupee and back
        graph[0][3] = 66.027     # one dollar is 66.027 rupees
        graph[3][0] = 1 / 66.027

        # Euro to Rupee and back
        graph[1][3] = 72.197    # one Euro is 72.197 rupees
        graph[3][1] = 1 / 72.197

        # Yuan to Rupee and back
        graph[2][3] = 10.191     # one Yuan is 10.191 rupees
        graph[3][2] = 1 / 10.191

        cycle, gain = find_negative_cycle(graph, currency_list)
        self._check_none(cycle)
        self._check_none(gain)


    def test_four_crypto_currency_graph(self):

        currency_list = np.array(["USD", "BTC", "ETH", "LTC"])
        graph = np.array([[1.00000000e+00, 1.00000000e+00, 1.09110747e-02, 4.21229992e-02],
                          [1.00000000e+00, 1.00000000e+00, 1.87899286e+01, 7.40192450e+01],
                          [9.16500000e+01, 5.32200000e-02, 1.00000000e+00, None],
                          [2.37400000e+01, 1.35100000e-02, None,           1.00000000e+00]])

        cycle, gain = find_negative_cycle(graph, currency_list)
        self._check_array(cycle, ['BTC', 'ETH', 'USD', 'BTC'])
        self._check(gain, 1722.0969)
