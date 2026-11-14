from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.basic import Coordinate
from src.aliases import percentage
from src.data.map import Map
from src.routing.algorithm import GridGraph, Node
from src.routing.profiles import (
    Profile,
    HumanProfile,
    RoverProfile,
    ExpeditionProfile,
)
from src.app.penalty.penalty_manager import PenaltyManager


@dataclass
class RoutingResult:
    """Selected path, geographic cell centres, and routing score."""

    path: list[Node]                         # (column, row)
    coordinates: list[tuple[float, float]]    # (longitude, latitude)
    total_cost: float | None

    @property
    def found(self) -> bool:
        return bool(self.path)


class RoutingExecutor:
    def __init__(
        self,
        top_left: Coordinate,
        bot_right: Coordinate,
        profile: Profile,
        fineness: percentage,
    ):
        self.top_left = top_left
        self.bot_right = bot_right
        self.profile = profile

        # The WorldSlice must already be prepared at this fineness.
        self.fineness = fineness

        self.penalty_manager = PenaltyManager()
        self.grid_graph: GridGraph | None = None

    @staticmethod
    def load_profile_from_json(json: dict) -> Profile:
        """Create a profile, copying defaults before applying overrides."""
        if not isinstance(json, dict):
            raise ValueError("A profile must be a JSON object.")

        kind = str(json.get("type", "")).lower()

        if kind == "expedition":
            members = json.get("profiles", [])

            if not isinstance(members, list) or not members:
                raise ValueError(
                    "An expedition needs a nonempty 'profiles' list."
                )

            profile = ExpeditionProfile()

            for member in members:
                participant = RoutingExecutor.load_profile_from_json(
                    member
                )
                profile.add_participant(participant)

            return profile

        profile_classes = {
            "human": HumanProfile,
            "rover": RoverProfile,
        }

        if kind not in profile_classes:
            raise ValueError(
                "Profile type must be 'human', 'rover', or 'expedition'."
            )

        profile = profile_classes[kind]()

        for section in (
            "hard_constraints",
            "soft_constraints",
            "parameters",
            "path",
        ):
            # Defaults are class attributes in your current profiles.
            values = deepcopy(getattr(profile, section, {}))
            overrides = json.get(section, {})

            if not isinstance(overrides, dict):
                raise ValueError(f"'{section}' must be a JSON object.")

            values.update(deepcopy(overrides))
            setattr(profile, section, values)

        return profile

    @staticmethod
    def _coordinate_to_node(
        point: Coordinate,
        reference: Map,
    ) -> Node:
        """Snap a geographic coordinate to the nearest grid cell."""
        dx, dy = reference.granularity

        column = int(round(
            (point.lon - reference.top_left[0]) / dx
        ))
        row = int(round(
            (reference.top_left[1] - point.lat) / dy
        ))

        rows, columns = reference.values.shape

        if not (
            0 <= column < columns
            and 0 <= row < rows
        ):
            raise ValueError(
                "Coordinate is outside the supplied map."
            )

        return column, row

    def execute(
        self,
        world_slice: Iterable[Map],
        start: Coordinate,
        end: Coordinate,
        *,
        elevation_map: Map,
        allow_diagonal: bool = True,
    ) -> RoutingResult:
        """
        Compute a route using an already aligned WorldSlice.

        Elevation values must be in metres.
        Map coordinates and granularity must be longitude/latitude degrees.
        Every non-elevation layer needs a registered penalty selector.

        Bounds must not cross the longitude seam.
        PenaltyError exceptions propagate to the caller.
        """
        for point in (start, end):
            if not (
                self.top_left.lon <= point.lon <= self.bot_right.lon
                and self.bot_right.lat <= point.lat <= self.top_left.lat
            ):
                raise ValueError(
                    "Start and end must lie inside the routing bounds."
                )

        # Elevation contributes slope penalties to edges.
        # Other layers contribute penalties to cells.
        layers = [
            layer
            for layer in world_slice
            if layer.value_type != elevation_map.value_type
        ]

        if layers:
            graph_cost_map = self.penalty_manager.build_from(
                self.profile,
                layers,
            )

            # Already converted by PenaltyManager:
            # positive values = traversable, zero = blocked.
            costs = np.array(
                graph_cost_map.values,
                dtype=np.float64,
                copy=True,
            )
        else:
            # Elevation-only routing has no additional cell penalties.
            costs = np.ones_like(
                elevation_map.values,
                dtype=np.float64,
            )

        # Restrict routing to the requested geographic area.
        rows, columns = costs.shape
        dx, dy = elevation_map.granularity

        longitudes = (
            elevation_map.top_left[0] + np.arange(columns) * dx
        )
        latitudes = (
            elevation_map.top_left[1] - np.arange(rows) * dy
        )

        inside = (
            (longitudes[None, :] >= self.top_left.lon)
            & (longitudes[None, :] <= self.bot_right.lon)
            & (latitudes[:, None] <= self.top_left.lat)
            & (latitudes[:, None] >= self.bot_right.lat)
        )

        costs[~inside] = 0.0

        # Keep the original GridGraph unchanged.
        # Its coordinate conversion runs in index space.
        self.grid_graph = GridGraph(
            cost_matrix=costs,
            top_left_corner=(0.0, 0.0),
            step_length=1.0,
            allow_diagonal=allow_diagonal,
            slope_list=None,
        )

        # The manager removes missing-elevation nodes and blocked
        # slope edges, then applies the remaining slope penalties.
        self.penalty_manager.apply_slope_penalties(
            self.grid_graph,
            self.profile,
            elevation_map,
        )

        source = self._coordinate_to_node(start, elevation_map)
        target = self._coordinate_to_node(end, elevation_map)

        # Avoid the original graph's obstacle exception for endpoints.
        if (
            source not in self.grid_graph.graph
            or target not in self.grid_graph.graph
        ):
            return RoutingResult(
                path=[],
                coordinates=[],
                total_cost=None,
            )

        # With graph origin (0, 0) and step 1:
        # graph longitude = column; graph latitude = -row.
        path, total_cost = self.grid_graph.shortest_path(
            start=(float(source[0]), float(-source[1])),
            end=(float(target[0]), float(-target[1])),
        )

        # Convert the resulting indices into actual geographic coordinates.
        coordinates = [
            (
                elevation_map.top_left[0] + column * dx,
                elevation_map.top_left[1] - row * dy,
            )
            for column, row in path
        ]

        return RoutingResult(
            path=path,
            coordinates=coordinates,
            total_cost=total_cost,
        )