from itertools import pairwise
from typing import Tuple, Union, Optional

from enum import Enum, auto
import networkx as nx
from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder

import math
from typing import Dict, List, Tuple

class Direction(Enum):
    North = auto()
    East = auto()
    West = auto()
    South = auto()

Node = Tuple[int, int]                 # (x, y) = (column, row)
SlopeCost = Tuple[Node, Node, float]   # (from node, to node, slope cost)

SQRT2 = math.sqrt(2)
DIRECTIONS = [(0, -1), (1, -1), (1, 0), (1, 1),
              (0, 1), (-1, 1), (-1, 0), (-1, -1)]   # N, NE, E, SE, S, SW, W, NW


class GridGraph:

    def __init__(self, cost_matrix: List[List[float]],
                 top_left_corner: Tuple[float, float],   # (lon, lat)
                 step_length: float,
                 allow_diagonal: bool = True,
                 slope_list: List[SlopeCost] = None
                 ):
        # cost_matrix: each cell has a particular cost
        self.cost_matrix = cost_matrix
        # top left corner: to set polar coordinates
        self.top_left_lon, self.top_left_lat = top_left_corner
        # step length
        self.step_length = step_length
        # boolean to allow diagonal movement
        self.allow_diagonal = allow_diagonal

        self.rows = len(cost_matrix)
        self.cols = len(cost_matrix[0])

        self.graph = self._build_graph()
        # slope_list applied to the graph
        if slope_list is not None:
            self.apply_slope_costs(slope_list)

    # ---------- Method 1: matrix -> graph ----------
    def _build_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()

        # Nodes: every passable cell (cost <= 0 or infinite = obstacle)
        for y in range(self.rows):
            for x in range(self.cols):
                c = self.cost_matrix[y][x]
                if c > 0 and not math.isinf(c):
                    g.add_node((x, y), cost=c)

        # Arcs: one per direction, base cost = cost of the destination cell
        for (x, y) in list(g.nodes):
            for dx, dy in DIRECTIONS:
                diagonal = dx != 0 and dy != 0
                if diagonal and not self.allow_diagonal:
                    continue
                target = (x + dx, y + dy)
                if target not in g:          # out of bounds or obstacle
                    continue
                base = self.cost_matrix[target[1]][target[0]] * (SQRT2 if diagonal else 1.0)
                g.add_edge((x, y), target, base_cost=base, slope_cost=0.0, weight=base)
        return g

    # ---------- Method 2: add slope cost to every arc ----------
    def apply_slope_costs(self, slope_list: List[SlopeCost]) -> None:
        """slope_list: [(from_node, to_node, slope_cost), ...] with nodes as (x, y)."""
        for src, dst, cost in slope_list:
            if self.graph.has_edge(src, dst):
                data = self.graph[src][dst]
                data["slope_cost"] = cost
                data["weight"] = data["base_cost"] + cost

    # ---------- Method 3: shortest path from coordinates ----------
    def shortest_path(self, start: Tuple[float, float], end: Tuple[float, float]
                      ) -> Tuple[List[Node], Optional[float]]:
        """
        :param start: (lon, lat) of the starting point
        :param end:   (lon, lat) of the destination
        :return: (path as list of (x, y) nodes, total cost); ([], None) if no path exists
        """
        s, t = self._to_node(*start), self._to_node(*end)

        # Admissible heuristic: octile/Manhattan distance x cheapest possible step
        weights = [d["weight"] for _, _, d in self.graph.edges(data=True)]
        min_step = min(weights) / (SQRT2 if self.allow_diagonal else 1.0) if weights else 1.0

        def heuristic(a: Node, b: Node) -> float:
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            dist = dx + dy + ((SQRT2 - 2) * min(dx, dy) if self.allow_diagonal else 0)
            return dist * min_step

        try:
            path = nx.astar_path(self.graph, s, t, heuristic=heuristic, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], None

        return path, nx.path_weight(self.graph, path, weight="weight")

    def _to_node(self, lon: float, lat: float) -> Node:
        x = int(round((lon - self.top_left_lon) / self.step_length))
        y = int(round((self.top_left_lat - lat) / self.step_length))
        if not (0 <= x < self.cols and 0 <= y < self.rows):
            raise ValueError(
                f"Point (lon={lon}, lat={lat}) maps to cell ({x}, {y}), "
                f"outside the {self.cols}x{self.rows} grid"
            )
        if (x, y) not in self.graph:
            raise ValueError(f"Cell ({x}, {y}) is an obstacle (cost <= 0)")
        return x, y

    #--------------------------METHODS FOR COST OF SINGLE ARC-------------------------------------------
    def cost_of_each_cell(self, path):
        path_with_cost = []
        for src, dst in pairwise(path):
            if self.graph.has_edge(src, dst):
                segment_cost = self.graph[src][dst]["weight"]
                path_with_cost.append((src, segment_cost))

        return path_with_cost
    # -------------------------METHODS TO CONVERT THE RESULTS IN USEFUL FORMATS-------------------------
    def convert_idx_to_polar(self, idx_list: List[Node]) -> List[Tuple[float, float]]:
        """
        Converts grid indices (x, y) = (column, row) into (lon, lat) coordinates.
        :param idx_list: list of (x, y) nodes, e.g. the path returned by shortest_path
        :return: list of (lon, lat)
        """
        polar_list = []
        for x, y in idx_list:
            lon = self.top_left_lon + x * self.step_length
            lat = self.top_left_lat - y * self.step_length  # latitude decreases as the row grows
            polar_list.append((lon, lat))
        return polar_list





@DeprecationWarning
class SpatialAStar:
    top_left_corner = Tuple[float, float]

    # Builder
    def __init__(self, top_left_corner, grid_cost, step_length):
        # top left corner as a tuple of floats
        self.top_left_corner = top_left_corner
        # map on which compute computation
        self.grid_cost = grid_cost

        self.step_length = step_length



    def movement(self, initial_position, direction):
        """
        This method updates the position of the subject based on its initial position and on the direction
        of the next step.
        :param initial_position: tuple of float, representing the initial position (from Coordinate class)
        :param direction: enum value that specifies the direction of the movement
        """
        final_position = initial_position

        # longitude
        # south: -1
        # north: +1

        # latitude
        # east: +1
        # west: -1

        match direction:
            # longitude
            case Direction.North: final_position.lat += self.step_length
            case Direction.South: final_position.lat -= self.step_length

            # latitude
            case Direction.East: final_position.lon += self.step_length
            case Direction.West: final_position.lon -= self.step_length


        return final_position

    def _to_grid_index(self, position: Union[Tuple[float, float], object]) -> Tuple[int, int]:
        """
        Converts coordinate (lat, lon) in integer indexes (x, y).
        """
        top_lat, left_lon = self.top_left_corner

        lat = position.lat if hasattr(position, 'lat') else position[0]
        lon = position.lon if hasattr(position, 'lon') else position[1]

        x = int(round((lon - left_lon) / self.step_length))

        y = int(round((top_lat - lat) / self.step_length))

        return x, y

    def algorithm(self, start_pos, destination_pos, convert_type = True):
        """
        :param start_pos: tuple of float or Coordinate, representing the initial position
        :param destination_pos: tuple of float or Coordinate, representing the destination position
        :return: It returns the path from source to destination and the number of runs.
        """
        # todo: modify this to make it native of class
        grid = Grid(matrix=self.grid_cost)

        # todo: modify algorithm if needed
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

        # converting from cardinals coordinate to grid index
        start_x, start_y = self._to_grid_index(start_pos)
        dest_x, dest_y = self._to_grid_index(destination_pos)

        start = grid.node(start_x, start_y)
        destination = grid.node(dest_x, dest_y)

        # if no path is found: empty list
        path, runs = finder.find_path(start, destination, grid)
        # runs represents the number of iterations needed to find the path

        # debug print
        #print(grid.grid_str(path=path, start=start, end=destination))

        # converting type into native python type
        if convert_type:
            path = self.extract_path_details(path)

        return path, runs


    # ---------------------------------------------------
    # In this section we define methods for converting specific library types into
    # native Python types.
    @staticmethod
    def extract_path_details(pathfinding_path):
        return [((node.x, node.y), node.weight) for node in pathfinding_path]