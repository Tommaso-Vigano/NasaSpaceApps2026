from typing import Tuple, Union
from unittest import case

import numpy as np
from enum import Enum, auto
from basic import Coordinate

from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder

class Direction(Enum):
    North = auto()
    East = auto()
    West = auto()
    South = auto()

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
            case Direction.North: final_position.lon += self.step_length
            case Direction.South: final_position.lon -= self.step_length

            # latitude
            case Direction.East: final_position.lat += self.step_length
            case Direction.West: final_position.lat -= self.step_length


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

    def algorithm(self, start_pos, destination_pos):
        """
        :param start_pos: tuple of int, representing the initial position
        :param destination_pos: tuple of int, representing the destination position
        :return: It returns the path from source to destination.
            "Runs" returns the total number of steps taken.
        """
        # todo: modify this to make it native of class
        grid = Grid(matrix=self.grid_cost)

        # todo: modify algorithm if needed
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

        start_x = start_pos[0]
        start_y = start_pos[1]

        dest_x = destination_pos[0]
        dest_y = destination_pos[1]

        start = grid.node(start_x, start_y)
        destination = grid.node(dest_x, dest_y)

        # if no path is found: empty list
        path, runs = finder.find_path(start, destination, grid)
        # runs represents the number of iterations needed to find the path

        # debug print
        #print(grid.grid_str(path=path, start=start, end=destination))

        return path, runs



if __name__ == "__main__":
    matrix = [
        [1.0, 1.5, 1.0, 0.0, 3.9],
        [1.0, 0.0, 1.0, 0.0, 8.9],
        [6.0, 1.0, 1.0, 8.0, 7.0],
        [1.0, 1.0, 1.0, 8.0, 7.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
    ]

    sp = SpatialAStar((0,0), matrix, 1.0)

    path, run = sp.algorithm((0,1), (1,2))

    print(run)
    print(path[-1].g)