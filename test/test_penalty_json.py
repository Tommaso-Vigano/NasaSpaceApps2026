"""JSON configuration selects existing profile rules without changing routing."""
import json

import numpy as np
import pytest

from src.app.penalty.penalty_manager import PenaltyManager
from src.app.penalty.penalty_manager_fixed import FixedPenaltyManager
from src.app.penalty.penalty_config import register_default_penalties
from src.app.penalty.implementations.cutoff_penalty import CutOffPenalty
from src.app.penalty.configure_penalty import configure_penalties
from src.app.routing_executor import RoutingExecutor
from src.basic import Coordinate
from src.data.map import Map
from src.data.types import Datatype
from src.routing.profiles import HumanProfile, RoverProfile, ExpeditionProfile, Profile


def layer(values, kind):
    values = np.asarray(values, dtype=float)
    h, w = values.shape
    return Map(values, (.0001, .0001), (77., 18.),
               (77.+(w-1)*.0001, 18.-(h-1)*.0001), "{}", kind, kind.name)


@pytest.mark.parametrize('manager_type', [PenaltyManager, FixedPenaltyManager])
def test_profile_selects_rule_and_disabled_missing_data_is_ignored(manager_type):
    manager = manager_type()
    config = json.dumps({'penalties': {'STEP_HEIGHT': True, 'THERMAL_INERTIA': True}})
    for profile, blocked in [(HumanProfile(), True), (RoverProfile(), False)]:
        manager.configure_from_json(profile, config)
        assert np.isinf(manager.select_cost_function(profile, Datatype.STEP_HEIGHT)(.35)) == blocked
        assert np.isinf(manager.select_cost_function(profile, Datatype.THERMAL_INERTIA)(50)) != blocked
        costs = manager.build_from(profile, [layer([[np.nan]], Datatype.RADIATION)])
        assert costs.values[0, 0] == 1
    assert HumanProfile.hard_constraints['slope'] == 30


def test_expedition_uses_strictest_member():
    profile = ExpeditionProfile()
    profile.add_participant(HumanProfile())
    profile.add_participant(RoverProfile())
    manager = PenaltyManager()
    manager.configure_from_json(profile, {'penalties': {'STEP_HEIGHT': True, 'THERMAL_INERTIA': True}})
    assert np.isinf(manager.select_cost_function(profile, Datatype.STEP_HEIGHT)(.35))
    assert np.isinf(manager.select_cost_function(profile, Datatype.THERMAL_INERTIA)(50))


@pytest.mark.parametrize('bad_config', [
    [], {'unknown': {}}, {'penalties': []}, {'weights': []},
    {'penalties': {'STEP_HEIGHT': 'false'}}, {'penalties': {'STEP_HEIGHT': 1}},
    {'penalties': {'TYPO': True}}, {'penalties': {'ELEVATION': True}},
    {'penalties': {'RADIATION': True}}, {'penalties': {'TEMPERATURE': True}},
    {'penalties': {'WIND_SPEED': True}}, {'penalties': {'ALBEDO': True}},
    {'weights': {'unknown': 1}}, {'weights': {'visibility': None}},
    {'weights': {'visibility': float('nan')}}, {'weights': {'visibility': -1}},
    {'weights': {'visibility': True}},
    {'weights': {'temperature_min_k': 300, 'temperature_max_k': 200}},
])
def test_bad_configuration_is_atomic(bad_config):
    manager = PenaltyManager()
    profile = HumanProfile()
    manager.configure_from_json(profile, {'penalties': {'STEP_HEIGHT': True, 'SLOPE': True}})
    original = manager.penalties
    with pytest.raises(ValueError):
        manager.configure_from_json(profile, bad_config)
    assert manager.penalties is original
    assert manager.slope_enabled


def test_reconfiguration_clears_previous_rules_and_can_enable_again():
    manager = PenaltyManager()
    profile = HumanProfile()
    config = {'penalties': {'TEMPERATURE': True, 'ALBEDO': True},
              'weights': {'temperature_max_k': 300, 'albedo_weight': 2}}
    manager.configure_from_json(profile, config)
    assert np.isinf(manager.select_cost_function(profile, Datatype.TEMPERATURE)(301))
    assert manager.select_cost_function(profile, Datatype.ALBEDO)(.5) == 1
    manager.configure_from_json(profile, {})
    assert manager.select_cost_function(profile, Datatype.TEMPERATURE)(float('nan')) == 0
    manager.configure_from_json(profile, config)
    assert np.isinf(manager.select_cost_function(profile, Datatype.TEMPERATURE)(301))


@pytest.mark.parametrize('manager_type', [PenaltyManager, FixedPenaltyManager])
def test_slope_switch_keeps_missing_dem_nodes_blocked(manager_type):
    profile = HumanProfile()
    manager = manager_type()
    manager.configure_from_json(profile, {'penalties': {'SLOPE': True}})
    assert np.isinf(manager.select_slope_cost_function(profile)(89))
    manager.configure_from_json(profile, {'penalties': {'SLOPE': False}})
    assert manager.select_slope_cost_function(profile)(89) == 0
    # Exercise the unchanged graph builder for the fixed implementation.
    if isinstance(manager, FixedPenaltyManager):
        graph = manager.build_graph(profile, [layer([[0, 1000, np.nan]], Datatype.ELEVATION)])
        assert (2, 0) not in graph.graph
        assert np.isfinite(graph.graph[(0, 0)][(1, 0)]['weight'])


@pytest.mark.parametrize('compact', [False, True])
def test_executor_uses_configured_manager_without_api_changes(compact):
    z = layer([[0, 1000, np.nan]], Datatype.ELEVATION)
    executor = RoutingExecutor(Coordinate(18, 77), Coordinate(18, 77.0002), HumanProfile())
    executor.penalty_manager.configure_from_json(executor.profile, {'penalties': {'SLOPE': False}})
    result = executor.execute([z, layer([[np.nan]*3], Datatype.DUST)],
                              Coordinate(18, 77), Coordinate(18, 77.0001),
                              elevation_map=z, compact=compact)
    assert result.path == [(0, 0), (1, 0)]
    assert result.total_cost == 1
    result = executor.execute([z], Coordinate(18, 77), Coordinate(18, 77.0002),
                              elevation_map=z, compact=compact)
    assert not result.found


def test_legacy_configuration_still_works():
    manager = PenaltyManager()
    configure_penalties(manager, 300, 2)
    assert isinstance(manager.select_cost_function(HumanProfile(), Datatype.TEMPERATURE), CutOffPenalty)
    fixed = register_default_penalties(FixedPenaltyManager())
    assert isinstance(fixed.select_cost_function(HumanProfile(), Datatype.STEP_HEIGHT), CutOffPenalty)


@pytest.mark.parametrize('profile', [Profile(), ExpeditionProfile()])
def test_unsupported_or_empty_profile_rejected(profile):
    with pytest.raises((TypeError, ValueError)):
        PenaltyManager().configure_from_json(profile, {})
