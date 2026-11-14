from itertools import pairwise

import networkx as nx
import pytest
from numpy.random._examples.cffi.extending import rng
import random
from routing.algorithm import GridGraph
import math

test_matrix_1 = [
        [1.0, 1.5, 1.0, 0.0, 3.9],
        [1.0, 0.0, 1.0, 0.0, 8.9],
        [6.0, 1.0, 1.0, 8.0, 7.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 9.0],
    ]

test_matrix_2 = [
    [1.0, 1.5, 1.0],
    [2.0, 3.2, 0.0],
    [1.9, 0.0, 1.0]
]

def generate_slope_list(matrix, seed = 37):
    rng = random.Random(seed)

    rows = len(matrix)
    cols = len(matrix[0])
    slope_list = []


    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),  # Orthogonal
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
    ]

    for r in range(rows):
        for c in range(cols):
            src = (c, r)  # (x, y) -> (col, row)

            for dc, dr in directions:
                nc, nr = c + dc, r + dr

                if 0 <= nr < rows and 0 <= nc < cols:
                    dst = (nc, nr)

                    slope_cost = rng.uniform(0.0, 50.0)

                    slope_list.append((src, dst, slope_cost))

    return slope_list

class Test_graph:
    def test_creating_graph(self):
        graph = GridGraph(test_matrix_1, (0.0, 0.0), step_length=1.0)

        # cell (x=0, y=0) -> cell (x=2, y=2): latitude goes DOWN with the rows
        path, cost = graph.shortest_path((0.0, 0.0), (2.0, -2.0))

        expected_path = [(0,0), (0,1), (1,2), (2,2)]
        expected_cost = 2.0 + math.sqrt(2.0)

        assert (path == expected_path)
        assert (cost == expected_cost)

    def test_path_graph_out_of_bounds(self):
        graph = GridGraph(test_matrix_1, (0.0, 0.0), step_length=1.0)
        # the coordinate is out of bounds, so it raises an error
        with pytest.raises(ValueError):
            graph.shortest_path((0.0, 0.0), (2.0, 2.0))

    def test_unreachable_destination(self):
        # the position (4.0,-4.0) is unreachable. So the function must return an empty list and
        # a None cost
        graph = GridGraph(test_matrix_1, (0.0, 0.0), step_length=1.0)

        path, cost = graph.shortest_path((0.0, 0.0), (4.0, -4.0))
        expected_path = []
        assert (path == expected_path)
        assert (cost is None)

    def test_starting_from_an_unreachable_location(self):
        # if we start from an "obstacle" location, it raises an error
        graph = GridGraph(test_matrix_1, (0.0, 0.0), step_length=1.0)
        with pytest.raises(ValueError):
            graph.shortest_path((3.0, 0.0), (0.0, 0.0))


    def test_ending_in_an_unreachable_location(self):
        # if our destination is an "obstacle" location, it raises an error
        graph = GridGraph(test_matrix_1, (0.0, 0.0), step_length=1.0)
        with pytest.raises(ValueError):
            graph.shortest_path((0.0, 0.0), (3.0, 0.0))

    def test_starting_already_in_destination(self):
        # The pathfinding algorithm needs just to return the source cell as result, the cost is 0.
        graph = GridGraph(test_matrix_1, (0.0, 0.0), step_length=1.0)
        path, cost = graph.shortest_path((0.0, 0.0), (0.0, 0.0))

        assert (path == [(0, 0)])
        assert (cost == 0.0)

    # -------------APPLY SLOPE COST--------------------------------------------------
    def test_building_with_slope(self):
        # testing simple implementation of slope addition
        graph = GridGraph(test_matrix_1, (0.0, 0.0), step_length=1.0)

        slope_list = generate_slope_list(test_matrix_1)
        graph.apply_slope_costs(slope_list)

        path, cost = graph.shortest_path((0.0, 0.0), (3.0, -2.0))

        expected_path = [(0, 0), (1, 0), (2, 1), (3, 2)]

        expected_cost = 0.0
        for src, dst in pairwise(path):
            if graph.graph.has_edge(src, dst):
                segment_cost = graph.graph[src][dst]["weight"]
                expected_cost += segment_cost

        assert (path == expected_path)
        assert (cost == pytest.approx(expected_cost))
    #----------------------------------TEST ON CONVERTING FUNCTIONS-------------------------
    def test_converting_from_index_to_polar(self):
        # test converting from index to polar
        graph = GridGraph(test_matrix_1, (0.0, 0.0), step_length=1.0)
        idx_list = [(0,0), (0,1), (1,2), (2,2)]
        res = graph.convert_idx_to_polar(idx_list)

        assert(res == [(0.0,0.0), (0.0,-1.0), (1.0,-2.0), (2.0,-2.0)])

    def test_converting_using_different_step(self):
        # test converting from index to polar, step_length = 2.5
        graph = GridGraph(test_matrix_1, (0.0, 0.0), step_length=2.5)
        idx_list = [(0, 0), (0, 1), (1, 2), (2, 2)]
        res = graph.convert_idx_to_polar(idx_list)

        assert (res == [(0.0, 0.0), (0.0, -2.5), (2.5, -5.0), (5.0, -5.0)])

    def test_converting_using_particular_top_left_corner(self):
        # test converting from index to polar, step_length = 3.0
        graph = GridGraph(test_matrix_1, (9.0, -2.0), step_length=3.0)
        idx_list = [(0, 0), (0, 1), (1, 2), (2, 2)]
        res = graph.convert_idx_to_polar(idx_list)
        assert (res == [(9.0, -2.0), (9.0, -5.0), (12.0, -8.0), (15.0, -8.0)])