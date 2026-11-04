# mypy: ignore-errors
"""Unit tests for the manifold shapes of _variable_codec: the ranges it registers for DAG and
TREE, top-k rankings, spherical polygons and policy sequence item scores."""

import importlib

import optquest
import optunahub
import pytest


module = optunahub.load_local_module(package="samplers/optquest", registry_root="package/")
_variable_codec = importlib.import_module(
    f"{module.OptQuestSampler.__module__.rsplit('.', 1)[0]}._variable_codec"
)


def _params(stmt):
    return _variable_codec.build(stmt).params


def test_dag_registers_every_ordered_pair_and_the_diagonal_is_zero():
    params = _params({"cmd": "DAG", "name": "g", "nodes": ["A", "B", "C"]})
    assert set(params) == {f"g@{a},{b}" for a in "ABC" for b in "ABC"}
    for node in "ABC":
        dist = params[f"g@{node},{node}"]
        assert (dist.low, dist.high) == (0, 0)
    assert (params["g@A,B"].low, params["g@A,B"].high) == (0, 1)


@pytest.mark.parametrize(
    "top_k, highest",
    [(None, 4), (4, 4), (9, 4), (2, 3), (1, 2), (0, 2), (-3, 2)],
)
def test_sequence_range_follows_top_k(top_k, highest):
    stmt = {"cmd": "SEQUENCE", "name": "r", "components": ["a", "b", "c", "d"]}
    if top_k is not None:
        stmt["top_k"] = top_k
    params = _params(stmt)
    assert all((d.low, d.high) == (1, highest) for d in params.values())


def _polygon_ranges(vertices):
    params = _params({"cmd": "SPHERICALREGION", "name": "p", "vertices": vertices})
    theta, phi = params["p@theta"], params["p@phi"]
    return (theta.low, theta.high), (phi.low, phi.high)


def test_spherical_polygon_across_the_antimeridian_uses_the_full_longitude_range():
    _, phi = _polygon_ranges([[0, 179], [10, 179], [10, -179], [0, -179]])
    assert phi == (-180.0, 180.0)


def test_spherical_polygon_longitudes_are_folded_like_the_engine_folds_them():
    _, phi = _polygon_ranges([[0, 190], [10, 190], [10, 200], [0, 200]])
    assert phi == pytest.approx((-170.0, -160.0))


def test_spherical_polygon_latitude_range_includes_the_bulge_of_great_circle_edges():
    theta, _ = _polygon_ranges([[0, 0], [60, 0], [60, 90], [0, 90]])
    # The edge from (60, 0) to (60, 90) reaches about 67.8 degrees at longitude 45.
    assert theta[0] <= 0.0
    assert theta[1] == pytest.approx(67.79, abs=0.05)


def test_spherical_polygon_around_the_north_pole_reaches_the_pole_at_every_longitude():
    # East to west around the North Pole: the region is the cap north of latitude 60.
    theta, phi = _polygon_ranges([[60, 0], [60, -90], [60, 180], [60, 90]])
    assert theta[0] == pytest.approx(60.0, abs=1e-5) and theta[1] == 90.0
    assert phi == (-180.0, 180.0)


def test_spherical_polygon_around_the_south_pole_reaches_the_pole_at_every_longitude():
    theta, phi = _polygon_ranges([[-60, 0], [-60, 90], [-60, 180], [-60, -90]])
    assert theta[0] == -90.0 and theta[1] == pytest.approx(-60.0, abs=1e-5)
    assert phi == (-180.0, 180.0)


def test_counterclockwise_spherical_polygon_covers_the_rest_of_the_earth():
    clockwise = [[20, 10], [20, 20], [10, 20], [10, 10]]
    theta, phi = _polygon_ranges(clockwise)
    assert theta == pytest.approx((10.0, 20.07), abs=0.01) and phi == (10.0, 20.0)
    assert _polygon_ranges(clockwise[::-1]) == ((-90.0, 90.0), (-180.0, 180.0))


def test_spherical_polygon_with_a_vertex_on_the_meridian_of_a_pole_holds_no_pole():
    # The path from the reference point to the South Pole is near the meridian of (-89, 45).
    theta, phi = _polygon_ranges([[0, 0], [0, 90], [-89, 45]])
    assert theta == pytest.approx((-89.0, 0.0), abs=1e-5) and phi == (0.0, 90.0)


def _policy_sequence_bound(matrix):
    params = _params(
        {
            "cmd": "POLICYSEQUENCE",
            "name": "p",
            "items": [f"i{k}" for k in range(len(matrix))],
            "features": [f"f{k}" for k in range(len(matrix[0]))],
            "matrix": matrix,
        }
    )
    dists = list(params.values())
    assert all(d.low == -d.high for d in dists)
    return dists[0].high


def test_policy_sequence_bound_covers_a_column_with_a_small_spread():
    # Values near 0.5 with a spread of about 0.001 scale to about 450; the old bound was 10.
    bound = _policy_sequence_bound([[0.500], [0.501], [0.502], [0.503], [0.504]])
    assert bound > 350.0


def test_policy_sequence_bound_for_binary_columns_is_the_row_length():
    # Each binary column maps to -1 and +1, so every scaled row has length sqrt(2).
    bound = _policy_sequence_bound([[0, 5], [1, 7], [0, 7], [1, 5]])
    assert bound == pytest.approx(1.01 * 2**0.5)


def test_policy_sequence_bound_for_constant_columns_is_one():
    assert _policy_sequence_bound([[3.0], [3.0], [3.0]]) == 1.0


def test_dag_and_tree_options_reach_the_model():

    model = optquest.OptQuestModel()
    _variable_codec.build_model_statement(
        model,
        {
            "cmd": "DAG",
            "name": "flow",
            "nodes": ["S", "A", "B", "T"],
            "start": "S",
            "end": "T",
            "connected": True,
            "predecessors": {"B": ["A"]},
        },
    )
    _variable_codec.build_model_statement(
        model, {"cmd": "TREE", "name": "net", "nodes": ["a", "b", "c"], "root": "b"}
    )
    text = model.describe()
    assert "DAG flow S,A,B,T S;T;1;B,A" in text
    assert "TREE net a,b,c b" in text


def test_dag_and_tree_without_options_keep_the_plain_statement():

    model = optquest.OptQuestModel()
    _variable_codec.build_model_statement(
        model, {"cmd": "DAG", "name": "flow", "nodes": ["A", "B"]}
    )
    _variable_codec.build_model_statement(
        model, {"cmd": "TREE", "name": "net", "nodes": ["a", "b"]}
    )
    lines = model.describe().splitlines()
    assert "DAG flow A,B" in lines
    assert "TREE net a,b" in lines
