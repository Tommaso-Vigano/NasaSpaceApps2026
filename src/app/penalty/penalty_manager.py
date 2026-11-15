from dataclasses import replace, fields
from enum import Enum
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from typing import Callable, Iterable
import json
import numpy as np

from src.data.map import Map
from src.data.types import Datatype
from src.routing.algorithm import SlopeCost
from src.routing.profiles import (
    Profile,
    HumanProfile,
    RoverProfile,
    ExpeditionProfile,
)

from .implementations.penalty import Penalty
from .implementations.combined_penalty import CombinedPenalty
from .implementations.cutoff_penalty import CutOffPenalty
from .implementations.lower_cutoff_penalty import LowerCutOffPenalty
from .implementations.progressive_cutoff import ProgressiveCutoff
from .implementations.soft_threshold_penalty import SoftThresholdPenalty
from .implementations.weighing_penalty import WeightPenalty
from .implementations.zero_penalty import ZeroPenalty


class PenaltyErrorCode(Enum):
    EMPTY_COMPONENTS = "empty_components"
    INVALID_REGISTRATION = "invalid_registration"
    PENALTY_NOT_REGISTERED = "penalty_not_registered"
    SELECTION_FAILED = "selection_failed"
    INVALID_PENALTY = "invalid_penalty"
    UNSUPPORTED_PROFILE = "unsupported_profile"
    EMPTY_EXPEDITION = "empty_expedition"
    INVALID_SLOPE_LIMIT = "invalid_slope_limit"
    NEGATIVE_PENALTY = "negative_penalty"
    COMPUTATION_FAILED = "computation_failed"


class PenaltyError(Exception):
    def __init__(self, code: PenaltyErrorCode, message: str):
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _penalty_values(values):
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise PenaltyError(
            PenaltyErrorCode.COMPUTATION_FAILED,
            "Penalty values must be numeric.",
        ) from error

    if np.any(values < 0):
        raise PenaltyError(
            PenaltyErrorCode.NEGATIVE_PENALTY,
            "Penalties must be nonnegative.",
        )

    return values


class PenaltyComponents:
    def __init__(self, maps: Iterable[Map]):
        self.maps = list(maps)

    def get_total_penalty(self) -> Map:
        if not self.maps:
            raise PenaltyError(
                PenaltyErrorCode.EMPTY_COMPONENTS,
                "No penalty maps to sum.",
            )

        reference = self.maps[0]
        total = np.zeros_like(reference.values, dtype=np.float64)

        for penalty_map in self.maps:
            values = _penalty_values(penalty_map.values)

            with np.errstate(over="ignore", invalid="ignore"):
                total += values

        return replace(
            reference,
            values=total,
            value_type=Datatype.PENALTY,
            title="Total penalty",
            metadata="Sum of all penalty components.",
        )


PenaltySelector = Callable[[Profile], Penalty]


class PenaltyManager:
    MAX_GRAPH_COST = 1e9
    MARS_RADIUS = 3_389_500.0

    def __init__(self):
        self.penalties: dict[Datatype, PenaltySelector] = {}
        self._configured_penalties: dict[Datatype, Penalty] | None = None
        self.slope_enabled = True

    def configure_from_json(self, profile: Profile, config: dict) -> None:
        if not isinstance(config, dict):
            raise ValueError("Expected a JSON object of layer names and booleans.")

        supported = {
            "SLOPE",
            "STEP_HEIGHT",
            "TERRAIN_ETA",
            "THERMAL_INERTIA",
            "LANDER_DISTANCE",
            "RETURN_TIME",
            "EVA_TIME",
            "VISIBILITY",
            "DUST_TAU",
            "SOLAR_FLUX",
            "ALBEDO",
        }

        for name, enabled in config.items():
            if name not in {dt.name for dt in Datatype} | {"SLOPE"}:
                raise ValueError(f"Unknown layer: {name}")
            if type(enabled) is not bool:
                raise ValueError(f"{name} must be true or false.")
            if enabled and name not in supported:
                raise ValueError(f"No built-in penalty for {name}.")

        rules = self._profile_penalties(profile)
        slope_enabled = config.get("SLOPE", False)

        if slope_enabled:
            self._get_slope_limit(profile)

        self._configured_penalties = {
            data_type: penalty
            for data_type, penalty in rules.items()
            if config.get(data_type.name, False)
        }
        self.slope_enabled = slope_enabled

    def _profile_penalties(self, profile: Profile) -> dict[Datatype, Penalty]:
        if isinstance(profile, ExpeditionProfile):
            if not profile.profiles:
                raise ValueError("An expedition must contain participants.")

            members = [
                self._profile_penalties(member)
                for member in profile.profiles
            ]

            return {
                data_type: CombinedPenalty([
                    member.get(data_type, ZeroPenalty())
                    for member in members
                ])
                for data_type in set().union(
                    *(member.keys() for member in members)
                )
            }

        rules = {
            Datatype.VISIBILITY: WeightPenalty(0.5),
            Datatype.ALBEDO: WeightPenalty(2.0),
            Datatype.DUST_TAU: CutOffPenalty(2.0),
            Datatype.SOLAR_FLUX: LowerCutOffPenalty(100.0),
        }

        if isinstance(profile, HumanProfile):
            rules.update({
                Datatype.STEP_HEIGHT: CutOffPenalty(0.30),
                Datatype.TERRAIN_ETA: SoftThresholdPenalty(1.0, 1.0),
                Datatype.LANDER_DISTANCE: CutOffPenalty(2000.0),
                Datatype.RETURN_TIME: CutOffPenalty(1800.0),
                Datatype.EVA_TIME: CutOffPenalty(21600.0),
            })

        elif isinstance(profile, RoverProfile):
            rules.update({
                Datatype.STEP_HEIGHT: CutOffPenalty(0.40),
                Datatype.TERRAIN_ETA: CutOffPenalty(1.5),
                Datatype.THERMAL_INERTIA: LowerCutOffPenalty(100.0),
            })

        else:
            raise ValueError(
                f"Unsupported profile: {type(profile).__name__}"
            )

        return rules

    def register_penalty(
        self,
        data_type: Datatype,
        selector: PenaltySelector,
    ) -> None:
        if not isinstance(data_type, Datatype) or not callable(selector):
            raise PenaltyError(
                PenaltyErrorCode.INVALID_REGISTRATION,
                "Provide a Datatype and a callable penalty selector.",
            )

        self.penalties[data_type] = selector

    def select_cost_function(
        self,
        profile: Profile,
        data_type: Datatype,
    ) -> Penalty:
        if self._configured_penalties is not None:
            return self._configured_penalties.get(
                data_type,
                ZeroPenalty(),
            )

        if data_type not in self.penalties:
            raise PenaltyError(
                PenaltyErrorCode.PENALTY_NOT_REGISTERED,
                f"No penalty selector registered for {data_type}.",
            )

        try:
            penalty = self.penalties[data_type](profile)
        except PenaltyError:
            raise
        except Exception as error:
            raise PenaltyError(
                PenaltyErrorCode.SELECTION_FAILED,
                f"Could not select a penalty for {data_type}.",
            ) from error

        if not isinstance(penalty, Penalty):
            raise PenaltyError(
                PenaltyErrorCode.INVALID_PENALTY,
                "The selector must return a Penalty instance.",
            )

        return penalty

    def _get_slope_limit(self, profile: Profile) -> float:
        if isinstance(profile, ExpeditionProfile):
            if not profile.profiles:
                raise PenaltyError(
                    PenaltyErrorCode.EMPTY_EXPEDITION,
                    "An expedition must contain at least one participant.",
                )

            return min(
                self._get_slope_limit(participant)
                for participant in profile.profiles
            )

        if not isinstance(profile, (HumanProfile, RoverProfile)):
            raise PenaltyError(
                PenaltyErrorCode.UNSUPPORTED_PROFILE,
                f"Unsupported profile: {type(profile).__name__}.",
            )

        try:
            limit = float(profile.hard_constraints["slope"])
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise PenaltyError(
                PenaltyErrorCode.INVALID_SLOPE_LIMIT,
                "The profile needs a numeric hard_constraints['slope'].",
            ) from error

        if not np.isfinite(limit) or limit <= 0:
            raise PenaltyError(
                PenaltyErrorCode.INVALID_SLOPE_LIMIT,
                "The slope limit must be finite and positive.",
            )

        return limit

    def select_slope_cost_function(self, profile: Profile) -> Penalty:
        if not self.slope_enabled:
            return ZeroPenalty()

        return ProgressiveCutoff(
            threshold=self._get_slope_limit(profile),
            weight=8.0,
        )

    def _construct_cost_pipeline(
        self,
        profile: Profile,
        world_slice: Iterable[Map],
    ) -> PenaltyComponents:
        penalty_maps = []

        for source_map in world_slice:
            penalty = self.select_cost_function(
                profile,
                source_map.value_type,
            )

            try:
                costs = penalty.compute(source_map.values)
            except PenaltyError:
                raise
            except Exception as error:
                raise PenaltyError(
                    PenaltyErrorCode.COMPUTATION_FAILED,
                    f"Failed to compute penalties for {source_map.title}.",
                ) from error

            costs = _penalty_values(costs)

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
        components = self._construct_cost_pipeline(profile, world_slice)
        total_penalty = components.get_total_penalty()
        values = _penalty_values(total_penalty.values)

        with np.errstate(over="ignore"):
            graph_costs = values + 1.0

        blocked = (
            ~np.isfinite(graph_costs)
            | (graph_costs > self.MAX_GRAPH_COST)
        )
        graph_costs[blocked] = 0.0

        return replace(
            total_penalty,
            values=graph_costs,
            title="Graph cell costs",
            metadata=(
                "Penalty + 1; missing, infinite, or excessive "
                "costs become zero."
            ),
        )

    def build_slopes_list(
        self,
        profile: Profile,
        ElevationMap: Map,
    ) -> Iterable[SlopeCost]:
        elevations = np.asarray(ElevationMap.values, dtype=np.float64)
        rows, columns = elevations.shape

        delta_lon, delta_lat = ElevationMap.granularity
        top_latitude = ElevationMap.top_left[1]

        neighbours = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        ]

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
                    dlat = latitude_to - latitude_from
                    dlon = radians(dc * delta_lon)

                    a = (
                        sin(dlat / 2.0) ** 2
                        + cos(latitude_from)
                        * cos(latitude_to)
                        * sin(dlon / 2.0) ** 2
                    )
                    a = min(1.0, max(0.0, a))

                    distance = (
                        2.0 * self.MARS_RADIUS * asin(sqrt(a))
                    )

                    if distance <= 1e-9:
                        yield (
                            (column, row),
                            (next_column, next_row),
                            np.inf,
                        )
                        continue

                    elevation_difference = (
                        float(elevation_to) - float(elevation_from)
                    )
                    slope = degrees(
                        atan2(elevation_difference, distance)
                    )

                    try:
                        cost = float(cost_function(slope))
                    except PenaltyError:
                        raise
                    except Exception as error:
                        raise PenaltyError(
                            PenaltyErrorCode.COMPUTATION_FAILED,
                            "Failed to compute a slope penalty.",
                        ) from error

                    if cost < 0:
                        raise PenaltyError(
                            PenaltyErrorCode.NEGATIVE_PENALTY,
                            "Slope penalties must be nonnegative.",
                        )

                    yield (
                        (column, row),
                        (next_column, next_row),
                        cost,
                    )

    def apply_slope_penalties(
        self,
        grid_graph,
        profile: Profile,
        elevation_map: Map,
    ) -> None:
        if self.slope_enabled:
            self._get_slope_limit(profile)

        missing = ~np.isfinite(elevation_map.values)

        grid_graph.graph.remove_nodes_from(
            (int(column), int(row))
            for row, column in np.argwhere(missing)
        )

        for source, target, cost in self.build_slopes_list(
            profile,
            elevation_map,
        ):
            if not grid_graph.graph.has_edge(source, target):
                continue

            grid_graph.apply_slope_costs([
                (source, target, cost)
            ])