# Negative Cycler Finder in Python
[![Build Status](https://travis-ci.org/mnpatil17/negative-cycles.svg?branch=master)](https://travis-ci.org/mnpatil17/negative-cycles)
[![Latest Version](https://img.shields.io/pypi/v/negative-cycles.svg)](https://pypi.python.org/pypi/negative-cycles/)


### Installation

This package uses pip for installation. You can find out more information about pip [here](https://pip.pypa.io/en/stable/quickstart/).

Installation can be done directly through pip, using `pip install negative-cycles`.

If this doesn't work, or you prefer having the source code on your machine, you can also execute the
following:

1. `git clone https://github.com/mnpatil17/negative_cycles.git`
2. `cd negative_cycles`
3. `pip install -e .`


### Usage

This package allows you to perform the following tasks in Python:

1. Run the multiplicative Bellman-Ford algorithm
2. Run the additive Bellman-Ford algorithm
3. Find negative cycles based on the multiplicative Bellman-Ford algorithm


#### Multiplicative Bellman-Ford Algorithm

You can run the multiplicative Bellman-Ford Algorithm as follows:

    from negative_cycles import multiplicative_bellman_ford

    graph = <YOUR GRAPH HERE>  # Note: for nodes without a connecting edge, the weights must be None
    distance_vector, predecessor_list = multiplicative_bellman_ford(graph)


#### Additive Bellman-Ford Algorithm

You can run the additive Bellman-Ford Algorithm as follows:

    from negative_cycles import additive_bellman_ford

    graph = <YOUR GRAPH HERE>  # Note: for nodes without a connecting edge, the weights must be None
    distance_vector, predecessor_list = additive_bellman_ford(graph)


#### Negative Cycles (Multiplicative)

You can run the multiplicative negative cycles as follows:

    from negative_cycles import find_negative_cycle

    graph = <YOUR GRAPH HERE>  # Note: for nodes without a connecting edge, the weights must be None
    graph_labels = <YOUR NODE NAMES>  # The names of the nodes of the graph, in order
    cycle, gain = find_negative_cycle(graph, graph_labels)

Note that if a negative cycle is NOT found, both `cycle` and `gain` will be `None`.

If a negative cycle is found, the `cycle` returned will be of the form
`['Node A', 'Node C', 'Node D', 'Node A']`, where the names in the list are based on your
`graph_labels`, and the first and last elements of the list are repeated to emphasize that the list
represents a cycle.

Additionally, if a negative cycle is found, the `gain` variable will be a number greater than
(or equal to) `1.0`. This is to say that if you start with 1 unit and follow the`cycle`, then you
will get `gain` units at the end of the cycle.


### Testing

If you choose, you can clone this repository locally and run the tests yourself.
To run tests, simply run `nosetests` from the `negative_cycles/` directory.
