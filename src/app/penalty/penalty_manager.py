from dataclasses import replace
from typing import Iterable
from typing import Callable

import numpy as np

from src.data.map import Map
from src.data.types import Datatype
from src.routing.algorithm import SlopeCost
from math import asin, atan2, cos, degrees, radians, sin, sqrt

from src.routing.profiles import (
    Profile,
    HumanProfile,
    RoverProfile,
    ExpeditionProfile,
)
from src.app.penalty.implementations.progressive_cutoff import ProgressiveCutoff
from .implementations.penalty import Penalty


class PenaltyComponents:
    """Store the component penalty maps on the same grid."""

    def __init__(self, maps: Iterable[Map]):
        self.maps = list(maps)

    def get_total_penalty(self) -> Map:
        """Sum all component maps into a new Map."""
        if not self.maps:
            raise ValueError("No penalty maps to sum.")

        reference = self.maps[0]
        total = np.zeros_like(reference.values, dtype=np.float64)

        for penalty_map in self.maps:
            total += penalty_map.values

        return replace(
            reference,
            values=total,
            value_type=Datatype.PENALTY,
            title="Total penalty",
            metadata="Sum of all penalty components.",
        )


PenaltySelector = Callable[[Profile], Penalty]


class PenaltyManager:
    """Build map penalties and select penalties for movement slopes."""

    def __init__(self):
        # Datatype -> function that configures a penalty from a profile.
        self.penalties: dict[Datatype, PenaltySelector] = {}

    def register_penalty(
        self,
        data_type: Datatype,
        selector: PenaltySelector,
    ) -> None:
        """Register how a penalty is selected for a data category."""
        self.penalties[data_type] = selector

    def select_cost_function(
        self,
        profile: Profile,
        data_type: Datatype,
    ) -> Penalty:
        """Return a penalty configured for the supplied profile."""
        if data_type not in self.penalties:
            raise NotImplementedError(
                f"No penalty selection configured for {data_type.name}."
            )

        selector = self.penalties[data_type]
        return selector(profile)

    def _get_slope_limit(self, profile: Profile) -> float:
        """Find the maximum allowed slope, including expedition members."""
        if isinstance(profile, ExpeditionProfile):
            if not profile.profiles:
                raise ValueError(
                    "An expedition must contain at least one participant."
                )

            # Every participant must be able to traverse the route.
            return min(
                self._get_slope_limit(participant)
                for participant in profile.profiles
            )

        if isinstance(profile, (HumanProfile, RoverProfile)):
            return float(profile.hard_constraints["slope"])

        raise TypeError(
            f"Unsupported profile: {type(profile).__name__}"
        )

    def select_slope_cost_function(self, profile: Profile) -> Penalty:

        return ProgressiveCutoff(
            threshold=self._get_slope_limit(profile),
            weight=1.0,
        )

    def _construct_cost_pipeline(
        self,
        profile: Profile,
        world_slice: Iterable[Map],
    ) -> PenaltyComponents:
        """Compute one component penalty map for each supplied layer."""
        penalty_maps = []

        for source_map in world_slice:
            penalty = self.select_cost_function(
                profile,
                source_map.value_type,
            )

            # Penalty implementations accept the entire values array.
            costs = penalty.compute(source_map.values)

            penalty_maps.append(
                replace(
                    source_map,
                    values=costs,
                    title=f"{source_map.title} — penalty",
                    metadata=(
                        f"Penalty computed with {type(penalty).__name__}. "
                        f"Source metadata: {source_map.metadata}"
                    ),
                )
            )

        return PenaltyComponents(penalty_maps)

    def build_from(
        self,
        profile: Profile,
        world_slice: Iterable[Map],
    ) -> Map:
        """Compute and sum the component penalty maps."""
        components = self._construct_cost_pipeline(profile, world_slice)
        return components.get_total_penalty()


    def build_slopes_list(
        self,
        profile,
        ElevationMap: Map,
    ) -> Iterable[SlopeCost]:
        """
        Yield (from_node, to_node, slope_cost) for adjacent cells.

        Nodes are (column, row) matrix indices.
        Slopes are signed angles in degrees.
        Cells with missing elevations are excluded.
        """
        elevations = np.asarray(ElevationMap.values, dtype=np.float64)
        rows, columns = elevations.shape

        delta_lon, delta_lat = ElevationMap.granularity
        top_latitude = ElevationMap.top_left[1]

        mars_radius = 3_389_500.0  # Metres; spherical approximation.

        neighbours = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1),
        ]

        # Placeholder selector: implemented separately below.
        cost_function = self.select_slope_cost_function(profile)

        for row in range(rows):
            latitude_from = radians(top_latitude - row * delta_lat)

            for column in range(columns):
                elevation_from = elevations[row, column]

                if not np.isfinite(elevation_from):
                    continue

                for dc, dr in neighbours:
                    next_column = column + dc
                    next_row = row + dr

                    if not (
                            0 <= next_column < columns
                            and 0 <= next_row < rows
                    ):
                        continue

                    elevation_to = elevations[next_row, next_column]

                    if not np.isfinite(elevation_to):
                        continue

                    latitude_to = radians(
                        top_latitude - next_row * delta_lat
                    )
                    latitude_difference = latitude_to - latitude_from
                    longitude_difference = radians(dc * delta_lon)

                    # Great-circle horizontal distance using the haversine formula.
                    a = (
                        sin(latitude_difference / 2.0) ** 2
                        + cos(latitude_from)
                        * cos(latitude_to)
                        * sin(longitude_difference / 2.0) ** 2
                    )

                    # Protect against floating-point rounding.
                    a = min(1.0, max(0.0, a))

                    horizontal_distance = (
                        2.0 * mars_radius * asin(sqrt(a))
                    )

                    if horizontal_distance <= 1e-9:
                        continue

                    elevation_difference = float(
                        elevation_to - elevation_from
                    )

                    slope = degrees(
                        atan2(elevation_difference, horizontal_distance)
                    )

                    cost = float(cost_function(slope))

                    yield (
                        (column, row),
                        (next_column, next_row),
                        cost,
                    )

