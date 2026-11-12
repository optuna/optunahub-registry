# mypy: ignore-errors
"""Unit tests for the manifold shapes of _variable_codec: the ranges it registers for DAG and
TREE, top-k rankings, spherical polygons and policy sequence item scores."""

import importlib
import math
import warnings

import optuna
import optunahub
import pytest


try:
    import optquest

    optquest.get_version_info()
except Exception as e:  # ImportError, or OSError when the engine library cannot be loaded
    pytest.skip(
        f"the OptQuest engine is unavailable on this platform ({e})", allow_module_level=True
    )


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


@pytest.mark.parametrize(
    "vertices",
    [
        [[10.0, 170.0], [10.0, -170.0], [-10.0, -170.0], [-10.0, 170.0]],
        [[80.0, 0.0], [80.0, 120.0], [80.0, -120.0]],
        [[-21.682, 93.379], [-21.682, 111.805], [-36.316, 111.805], [-36.316, 93.379]],
    ],
)
def test_spherical_region_distributions_cover_the_whole_sphere(vertices):
    """A tighter range would have to follow the engine's geometry exactly (edges that rise above
    their end vertices, regions that hold a pole); the engine keeps every point in the region."""
    params = _params({"cmd": "SPHERICALREGION", "name": "g", "vertices": vertices})
    assert (params["g@theta"].low, params["g@theta"].high) == (-90.0, 90.0)
    assert (params["g@phi"].low, params["g@phi"].high) == (-180.0, 180.0)


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


@pytest.mark.parametrize(
    "low, high, top",
    [
        (0.0, 360.0, 360.0),
        (-45.0, 45.0, 45.0),
        (-180.0, 180.0, 180.0),
        (90.0, 450.0, 450.0),
        (0.0, 720.0, 720.0),
        (320.0, 40.0, 400.0),
        (50.0, 10.0, 370.0),
        (400.0, 30.0, 760.0),  # |max - min| >= 360: the engine treats it as the full circle
        (10.0, 10.0, 10.0),
    ],
)
def test_arc_top_is_the_engines_arc_measured_upward_from_min(low, high, top):
    assert _variable_codec._arc_top(low, high) == top
    fold = _variable_codec._azimuth_decoder(low, top)
    for engine_value in (0.0, 45.0, 180.0, 315.0, 359.0):
        value = fold(engine_value)
        assert low <= value <= top
        if top - low >= 360.0 - 1e-9:  # every angle is on a full circle: only the turn changes
            turns = (value - engine_value) / 360.0
            assert math.isclose(turns, round(turns), abs_tol=1e-9)


@pytest.mark.parametrize(
    "stmt, engine_built, new_max",
    [
        ({"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 1.0, "step": 0.6}, False, 0.6),
        ({"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 1.0, "step": 0.6}, True, 1.2),
        ({"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 1.0, "step": 0.4}, True, 1.2),
        ({"cmd": "DISCRETE", "name": "d", "min": 0.1, "max": 0.7, "step": 0.2}, False, 0.7),
        ({"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 0.3, "step": 0.1}, True, 0.3),
        (
            {"cmd": "discretevector", "name": "v", "size": 2, "min": 0, "max": 5, "step": 2},
            False,
            4.0,
        ),
        (
            {"cmd": "DISCRETENDVARIABLE", "name": "n", "dims": [2], "min": 0, "max": 5, "step": 2},
            True,
            6.0,
        ),
    ],
)
def test_align_stepped_max_picks_one_grid_top(stmt, engine_built, new_max):
    engine_stmt, codec_stmt = _variable_codec.align_stepped_max(stmt, engine_built=engine_built)
    assert codec_stmt["max"] == new_max
    if new_max == stmt["max"]:
        assert codec_stmt is stmt
    if engine_built:
        assert engine_stmt is stmt  # the caller built the model; it is not changed


@pytest.mark.parametrize(
    "low, high, step",
    [(-1.0, -0.85, 0.1), (1.0, 1.3, 0.2), (0.1, 0.4, 0.2), (2.5, 2.95, 0.3)],
)
def test_align_stepped_max_keeps_a_one_step_grid_that_the_engine_accepts(low, high, step):
    """With two grid values, Optuna's top minus min can be an ulp shorter than the step in
    floating point, and the engine refuses a range shorter than one step. The engine gets the
    next float up instead; its grid is the same two values."""
    stmt = {"cmd": "DISCRETE", "name": "d", "min": low, "max": high, "step": step}
    engine_stmt, codec_stmt = _variable_codec.align_stepped_max(stmt, engine_built=False)
    top = codec_stmt["max"]
    assert math.isclose(top, low + step) and top <= high
    assert engine_stmt["max"] - low >= step and engine_stmt["max"] < high


@pytest.mark.parametrize(
    "low, high, step",
    [
        (0.0, 2.4, 1.0 / 3.0),
        (0.0, 2.4, 1.0 / 13.0),
        (0.0, 2.4, math.pi / 4.0),
        (0.0, 1.0, 1.0 / 15.0),
        (0.0, 3.0, 1.0 / 3.0),
        (0.0, 10.0, 1.0 / 3.0),
        (0.0, 1.0, 0.1 * 3),
        (-1.53, -0.93, 1.0 / 3.0),
        (1.0, 1.2285714285714286, 1.0 / 7.0),
        (5.0, 7.0, 0.1 * 3),
        (0.0, 1.0, 0.6),
        (0.1, 0.3, 0.2),
    ],
)
def test_the_statements_grid_top_is_optunas_own_top(low, high, step):
    """With statements=, the grid top is the `high` of FloatDistribution(low, high, step=step),
    which Optuna computes once. Optuna would move some of these tops down again if it got one
    back as `high` (2.333... to 1.999... for a step of 1/3), and a top taken to that fixed point
    would lose values of the grid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Optuna warns that it moves `high`
        expected = optuna.distributions.FloatDistribution(low, high, step=step).high
    assert _variable_codec._optuna_grid_top(low, high, step) == expected
    stmt = {"cmd": "DISCRETE", "name": "d", "min": low, "max": high, "step": step}
    engine_stmt, codec_stmt = _variable_codec.align_stepped_max(stmt, engine_built=False)
    assert codec_stmt["max"] == expected
    assert expected <= engine_stmt["max"] <= expected + 1e-12 and engine_stmt["max"] - low >= step


@pytest.mark.parametrize(
    "low, high, step, top",
    [
        (0.0, 1.0, 0.6, 1.2),  # the nearest grid value is above max
        (0.0, 1.0, 0.4, 1.2),
        (0.0, 1.0, 0.333333, 1.0),  # 0.999999 is within 1e-5 of max: the engine keeps max
        (0.0, 1.0, 0.142858, 1.0),  # 1.000006 is within 1e-5 of max
        (0.0, 100.000001, 1.0, 100.000001),
        (0.0, 4.999999, 1.0, 4.999999),
        (0.0, 10.00002, 1.0, 10.0),  # 2e-5 from max: the engine moves its bound to the grid
    ],
)
def test_the_model_grid_top_follows_the_engines_tolerance(low, high, step, top):
    """COptQuestDiscreteVariable.makeValid keeps max as the upper bound when the grid value
    nearest to it is within OptQuestDefinition.EPSILON (1e-5); otherwise the bound is that grid
    value. The sampler tests in test_sampler.py run the engine on the same grids."""
    assert _variable_codec._engine_grid_top(low, high, step) == top


@pytest.mark.parametrize(
    "low, high, step, stepped",
    [
        (0.0, 1.0, 0.25, True),
        (1e6, 1e6 + 10.0, 0.001, False),  # far from zero for its step
        (1e9, 1e9 + 10.0, 0.1, False),
        (-3e10, -3e10 + 3.0, 0.3, False),
        (1e7, 1e7 + 1000.0, 0.01, False),
        (0.0, 1e12, 1.0, True),  # whole numbers have no rounding error
        (8.0, 8.3, 0.1, True),
        (0.0, 1.0, 1.0 / 3.0, False),  # Optuna's decimal check cannot confirm the step
    ],
)
def test_stepped_registers_the_continuous_range_when_optuna_cannot_hold_the_grid(
    low, high, step, stepped
):
    dist = _variable_codec._stepped(low, high, step)
    assert (dist.low, dist.high) == (low, high)
    assert (dist.step is not None) == stepped
    if stepped:  # every grid value as the engine computes it is in the distribution
        count = int(0.5 + (high - low) / step)
        ks = sorted({0, 1, count // 3, count // 2, count - 1, count})
        assert all(dist._contains(low + k * step) for k in ks)


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 1.0, "step": -0.25},
        {"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 1.0, "step": 0.0},
        {"cmd": "DISCRETEVECTOR", "name": "v", "size": 2, "min": 0.0, "max": 1.0, "step": -1},
        {
            "cmd": "DISCRETEMATRIX",
            "name": "m",
            "rows": 1,
            "cols": 2,
            "min": 0,
            "max": 1,
            "step": -1,
        },
        {"cmd": "DISCRETENDVARIABLE", "name": "n", "dims": [2], "min": 0, "max": 1, "step": -1},
    ],
)
def test_a_step_that_is_not_positive_is_refused(stmt):
    """Optuna's distributions take only a positive step (and the 10.2 engine makes a variable
    with a negative step take a single value)."""
    with pytest.raises(ValueError, match="step must be a positive number"):
        _params(stmt)


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "DISCRETEVECTOR", "name": "v", "size": 2, "min": 0.0, "max": 2.0, "step": 0.5},
        {
            "cmd": "DISCRETEMATRIX",
            "name": "v",
            "rows": 2,
            "cols": 2,
            "min": 0,
            "max": 2,
            "step": 0.5,
        },
        {
            "cmd": "DISCRETENDVARIABLE",
            "name": "v",
            "dims": [2, 2],
            "min": 0,
            "max": 2,
            "step": 0.5,
        },
    ],
)
def test_every_element_of_a_discrete_container_has_the_grid(stmt):
    params = _params(stmt)
    assert params and all(
        dist == optuna.distributions.FloatDistribution(0.0, 2.0, step=0.5)
        for dist in params.values()
    )


INTEGER_V = {"cmd": "INTEGER", "name": "v", "min": 0, "max": 20}
DISCRETE_D = {"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 20.0, "step": 2.0}
ENUM_K = {"cmd": "ENUMERATION", "name": "k", "values": [1.0, 2.0, 3.0]}
ENUM_W = {"cmd": "ENUMERATION", "name": "w", "values": [10, 20, 30, 40]}
CONTINUOUS_X = {"cmd": "CONTINUOUS", "name": "x", "min": 0.0, "max": 10.0}
VECTOR_U = {"cmd": "DISCRETEVECTOR", "name": "u", "size": 2, "min": 0.0, "max": 10.0, "step": 1.0}
DECLARED = {
    s["name"].lower(): s for s in (INTEGER_V, DISCRETE_D, ENUM_K, ENUM_W, CONTINUOUS_X, VECTOR_U)
}


def _exclusive(name, low, high):
    return {"cmd": "EXCLUSIVE", "name": name, "min": low, "max": high}


@pytest.mark.parametrize(
    "stmt, fitted",
    [
        (_exclusive("v", 5.5, 15.5), (5.0, 16.0)),  # excludes 6 to 15 either way
        (_exclusive("V", 5.0, 16.0), (5.0, 16.0)),  # edges that are values stay
        (_exclusive("d", 5.0, 15.0), (4.0, 16.0)),  # excludes 6, 8, ..., 14 either way
        (_exclusive("k", 1.5, 2.5), (1.0, 3.0)),  # excludes 2 either way
        (_exclusive("w", 15.0, 35.0), (10.0, 40.0)),  # excludes 20 and 30 either way
        (_exclusive("x", -5.0, 3.0), (-5.0, 3.0)),  # no value is moved to -5: it is left alone
        (_exclusive("x", 2.5, 3.5), (2.5, 3.5)),
        (_exclusive("u@1", 2.5, 6.5), (2.0, 7.0)),  # an element of a vector
        (_exclusive("z", 2.5, 6.5), (2.5, 6.5)),  # a name that is not declared here
    ],
)
def test_an_exclusive_range_gets_edges_that_are_values_of_its_variable(stmt, fitted):
    """The engine moves a value inside an excluded range to the nearer edge, so it proposes the
    edges themselves. With statements=, an edge that the variable cannot take moves outward to
    the nearest value that it can take, which excludes the same values."""
    built = _variable_codec.fit_exclusive_range(stmt, DECLARED, engine_built=False)
    assert (built["min"], built["max"]) == fitted
    if fitted == (stmt["min"], stmt["max"]):
        assert built is stmt


def test_a_built_model_with_an_exclusive_edge_that_is_not_a_value_is_refused():
    """With model=, the model is built and cannot change, so the sampler names the range that
    excludes the same values."""
    with pytest.raises(ValueError, match="from 5.0 to 16.0, which excludes the same values"):
        _variable_codec.fit_exclusive_range(
            _exclusive("v", 5.5, 15.5), DECLARED, engine_built=True
        )
    unchanged = _exclusive("x", -5.0, 3.0)
    assert _variable_codec.fit_exclusive_range(unchanged, DECLARED, engine_built=True) is unchanged


@pytest.mark.parametrize(
    "stmt, instead",
    [
        (_exclusive("v", -0.5, 3.5), "Change the variable's min or max"),
        (_exclusive("k", 0.5, 1.5), "Leave those values out of the variable's values"),
        (_exclusive("x", -1.0, 3.0), "Change the variable's min or max"),
    ],
)
def test_an_exclusive_range_over_the_start_of_the_values_is_refused(stmt, instead):
    """No value is below the first one, and the engine would move 0 and 1 to -0.5: change the
    variable's min (or, for an ENUMERATION, its values) instead."""
    for engine_built in (False, True):
        with pytest.raises(ValueError, match=instead):
            _variable_codec.fit_exclusive_range(stmt, DECLARED, engine_built=engine_built)


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "CONTINUOUS", "name": "x", "min": 0.0, "max": 1.0},
        {"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 1.0, "step": 2.0},  # step > range
        {"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 1.0, "step": -0.5},
        {"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 1.0},
    ],
)
def test_align_stepped_max_leaves_other_statements_alone(stmt):
    engine_stmt, codec_stmt = _variable_codec.align_stepped_max(stmt, engine_built=False)
    assert engine_stmt is stmt and codec_stmt is stmt


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "INTEGER", "name": "v", "min": 0.5, "max": 5.5},
        {"cmd": "INTEGERVECTOR", "name": "v", "size": 2, "min": 0, "max": 28.999999999999996},
        {"cmd": "INTEGERMATRIX", "name": "v", "rows": 1, "cols": 2, "min": -7.2, "max": 3},
        {"cmd": "INTEGERNVARIABLE", "name": "v", "dims": [2], "min": 0.1, "max": 4},
    ],
)
def test_integer_bounds_must_be_whole_numbers(stmt):
    """The engine's values are min, min + 1, ...; with a fractional min, an IntDistribution
    would reject every one of them."""
    with pytest.raises(ValueError, match="whole numbers"):
        _params(stmt)


@pytest.mark.parametrize("components", [3, "3", 3.0])
def test_a_mixture_with_a_count_names_its_components_1_to_n(components):
    """The statement grammar takes a count n in place of the component names, and the engine
    then names the components 1 to n."""
    params = _params({"cmd": "MIXTURE", "name": "m", "components": components})
    assert set(params) == {"m@1", "m@2", "m@3"}


@pytest.mark.parametrize(
    "stmt, label, row",
    [
        (
            {"cmd": "TUPLE", "name": "t", "tuples": [[1.0, 2.0], [3.0, 4.0]]},
            "[1.0, 2.0]",
            [1.0, 2.0],
        ),
        (
            {"cmd": "GEOLOCATION", "name": "t", "coords": [[10.0, 20.0]]},
            "[10.0, 20.0]",
            [10.0, 20.0],
        ),
    ],
)
def test_a_decoded_row_is_a_copy(stmt, label, row):
    codec = _variable_codec.build(stmt)
    first = codec.decode({"t": label})
    first.append(99.0)
    assert codec.decode({"t": label}) == row


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "INTEGER", "name": "k", "min": 0, "max": 2**31},
        {"cmd": "INTEGERVECTOR", "name": "k", "size": 2, "min": -(2**31), "max": 1},
        {"cmd": "INTEGERMATRIX", "name": "k", "rows": 1, "cols": 2, "min": 0, "max": 2**40},
        {"cmd": "DISCRETE", "name": "k", "min": 0.0, "max": 2.0**36, "step": 16.0},
        {"cmd": "DISCRETEVECTOR", "name": "k", "size": 2, "min": 0.0, "max": 1.0, "step": 1e-10},
    ],
)
def test_a_grid_wider_than_the_engines_is_refused(stmt):
    """The engine counts the steps of a grid in a 32-bit int and proposes only the first 2**31
    values of a wider one."""
    with pytest.raises(ValueError, match=r"2\*\*31"):
        _params(stmt)


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "INTEGER", "name": "k", "min": 0, "max": 2**31 - 1},
        {"cmd": "DISCRETE", "name": "k", "min": 0.0, "max": 16.0 * (2**31 - 1), "step": 16.0},
    ],
)
def test_the_widest_grid_that_the_engine_holds_is_accepted(stmt):
    assert _params(stmt)


def test_coerce_fields_takes_every_form_and_gives_names_as_text():
    commands = optquest.load_spec()["commands"]
    coerce = _variable_codec.coerce_fields
    budget = coerce(
        {
            "cmd": "BUDGETEDSELECTION",
            "name": "b",
            "n": 4.0,
            "budget_min": "1",
            "budget_max": "10",
            "k_min": "1",
            "k_max": 2.0,
            "costs": ["3", 4],
        },
        commands,
    )
    assert budget["n"] == 4 and type(budget["n"]) is int and budget["budget_min"] == 1.0
    assert type(budget["k_min"]) is int and type(budget["k_max"]) is int
    assert budget["costs"] == [3.0, 4.0]
    dag = coerce(
        {"cmd": "DAG", "name": "g", "nodes": [1, 2], "start": 1, "end": 2, "predecessors": {2: 1}},
        commands,
    )
    assert dag["nodes"] == ["1", "2"] and (dag["start"], dag["end"]) == ("1", "2")
    assert dag["predecessors"] == {"2": "1"}
    path = coerce(
        {
            "cmd": "PATH",
            "name": "p",
            "nodes": [1, 2, 3],
            "path": {"start": 1, "end": 3, "required": [2]},
        },
        commands,
    )
    assert path["path"] == {"start": "1", "end": "3", "required": ["2"]}
    stages = coerce({"cmd": "MULTISTAGESEQUENCE", "name": "s", "stages": [[1, 2], [3]]}, commands)
    assert stages["stages"] == [["1", "2"], ["3"]]
    mixture = coerce({"cmd": "MIXTURE", "name": "m", "components": [1, 2]}, commands)
    assert mixture["components"] == ["1", "2"]


@pytest.mark.parametrize("count", [2**31, 2**40, 2.0**40, "1099511627776", -(2**31) - 1])
def test_a_mixture_count_beyond_the_engines_int_is_refused(count):
    """The count of a MIXTURE is an int field like every other count. coerce_fields() builds no
    component, so this test allocates nothing: the codec would build one name for each."""
    commands = optquest.load_spec()["commands"]
    coerce = _variable_codec.coerce_fields
    with pytest.raises(ValueError, match=r"MIXTURE 'm': components is .*2\*\*31 - 1"):
        coerce({"cmd": "MIXTURE", "name": "m", "components": count}, commands)
    kept = coerce({"cmd": "MIXTURE", "name": "m", "components": 2**31 - 1}, commands)
    assert kept["components"] == 2**31 - 1


# ---------------------------------------------------------------------------------------------
# Every codec handler: the parameters that it registers, as the statements reference gives them
# ---------------------------------------------------------------------------------------------

F = optuna.distributions.FloatDistribution
Int = optuna.distributions.IntDistribution
C = optuna.distributions.CategoricalDistribution
ROW3 = [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]

# (statement after coerce_fields, the number of parameters, {parameter: its distribution} for
# some of them, the parameters whose distribution holds one value). The names and the ranges come from the
# statements reference: "<name>@<key>" for each element, item, node or component.
REGISTERED = [
    ({"cmd": "CONTINUOUS", "name": "x", "min": -1.0, "max": 2.0}, 1, {"x": F(-1.0, 2.0)}, ()),
    ({"cmd": "INTEGER", "name": "x", "min": 0, "max": 5}, 1, {"x": Int(0, 5)}, ()),
    (
        {"cmd": "DISCRETE", "name": "x", "min": 0.0, "max": 1.0, "step": 0.25},
        1,
        {"x": F(0.0, 1.0, step=0.25)},
        (),
    ),
    ({"cmd": "BINARY", "name": "x"}, 1, {"x": Int(0, 1)}, ()),
    ({"cmd": "DESIGN", "name": "x", "num": 3}, 1, {"x": C(["1", "2", "3"])}, ()),
    (
        {"cmd": "ENUMERATION", "name": "x", "values": [1.5, 2.5, 4.0]},
        1,
        {"x": C([1.5, 2.5, 4.0])},
        (),
    ),
    (
        {"cmd": "TUPLE", "name": "x", "tuples": [[1.0, 2.0], [3.0, 4.0]]},
        1,
        {"x": C(["[1.0, 2.0]", "[3.0, 4.0]"])},
        (),
    ),
    (
        {"cmd": "GEOLOCATION", "name": "x", "coords": [[10.0, 20.0], [30.0, 40.0]]},
        1,
        {"x": C(["[10.0, 20.0]", "[30.0, 40.0]"])},
        (),
    ),
    (
        {"cmd": "VECTOR", "name": "x", "size": 3, "min": -1.0, "max": 1.0},
        3,
        {"x@1": F(-1.0, 1.0), "x@3": F(-1.0, 1.0)},
        (),
    ),
    (
        {"cmd": "MATRIX", "name": "x", "rows": 2, "cols": 3, "min": 0.0, "max": 1.0},
        6,
        {"x@1,1": F(0.0, 1.0), "x@2,3": F(0.0, 1.0)},
        (),
    ),
    (
        {"cmd": "NDVARIABLE", "name": "x", "dims": [2, 2, 2], "min": 0.0, "max": 1.0},
        8,
        {"x@1,1,1": F(0.0, 1.0), "x@2,2,2": F(0.0, 1.0)},
        (),
    ),
    (
        {"cmd": "INTEGERVECTOR", "name": "x", "size": 2, "min": 0, "max": 4},
        2,
        {"x@1": Int(0, 4), "x@2": Int(0, 4)},
        (),
    ),
    (
        {"cmd": "INTEGERMATRIX", "name": "x", "rows": 2, "cols": 2, "min": 0, "max": 3},
        4,
        {"x@1,1": Int(0, 3), "x@2,2": Int(0, 3)},
        (),
    ),
    (
        {"cmd": "INTEGERNVARIABLE", "name": "x", "dims": [2, 3], "min": 0, "max": 2},
        6,
        {"x@1,1": Int(0, 2), "x@2,3": Int(0, 2)},
        (),
    ),
    (
        {"cmd": "DISCRETEVECTOR", "name": "x", "size": 2, "min": 0.0, "max": 1.0, "step": 0.5},
        2,
        {"x@2": F(0.0, 1.0, step=0.5)},
        (),
    ),
    (
        {
            "cmd": "DISCRETEMATRIX",
            "name": "x",
            "rows": 2,
            "cols": 2,
            "min": 0.0,
            "max": 1.0,
            "step": 0.5,
        },
        4,
        {"x@2,2": F(0.0, 1.0, step=0.5)},
        (),
    ),
    (
        {
            "cmd": "DISCRETENDVARIABLE",
            "name": "x",
            "dims": [2, 2],
            "min": 0.0,
            "max": 1.0,
            "step": 0.5,
        },
        4,
        {"x@2,1": F(0.0, 1.0, step=0.5)},
        (),
    ),
    ({"cmd": "BINARYVECTOR", "name": "x", "k": 3}, 3, {"x@1": Int(0, 1), "x@3": Int(0, 1)}, ()),
    ({"cmd": "BINARYMATRIX", "name": "x", "rows": 2, "cols": 2}, 4, {"x@1,2": Int(0, 1)}, ()),
    ({"cmd": "BINARYNDVARIABLE", "name": "x", "dims": [2, 2]}, 4, {"x@2,2": Int(0, 1)}, ()),
    (
        {"cmd": "PERMUTATION", "name": "x", "items": ["a", "b", "c"]},
        3,
        {"x@a": Int(1, 3), "x@c": Int(1, 3)},
        (),
    ),
    ({"cmd": "CHOICE", "name": "x", "k": 4}, 1, {"x@choice": Int(1, 4)}, ()),
    (
        {
            "cmd": "MULTICHOICE",
            "name": "x",
            "choices": ["a", "b", "c"],
            "min_select": 1,
            "max_select": 2,
        },
        3,
        {"x@a": Int(0, 1), "x@c": Int(0, 1)},
        (),
    ),
    (
        {"cmd": "MIXTURE", "name": "x", "components": ["a", "b", "c"], "min": 0.1, "max": 0.6},
        3,
        {"x@a": F(0.1, 0.6), "x@c": F(0.1, 0.6)},
        (),
    ),
    (
        {"cmd": "SEQUENCE", "name": "x", "components": ["a", "b", "c"]},
        3,
        {"x@a": Int(1, 3), "x@c": Int(1, 3)},
        (),
    ),
    (
        {"cmd": "SEQUENCE", "name": "x", "components": ["a", "b", "c", "d"], "top_k": 2},
        4,
        {"x@a": Int(1, 3), "x@d": Int(1, 3)},
        (),
    ),
    (
        {"cmd": "MULTISTAGESEQUENCE", "name": "x", "stages": [["a", "b"], ["c", "d", "e"]]},
        5,
        {"x@a": Int(1, 2), "x@b": Int(1, 2), "x@c": Int(1, 3), "x@e": Int(1, 3)},
        (),
    ),
    (
        {
            "cmd": "POLICYSEQUENCE",
            "name": "x",
            "items": ["i1", "i2"],
            "features": ["f1", "f2"],
            "matrix": [[1, 2], [3, 4]],
        },
        2,
        {},
        (),
    ),
    (
        {
            "cmd": "TOPOLOGICALSEQUENCE",
            "name": "x",
            "tasks": ["a", "b", "c"],
            "predecessors": {"c": ["a"]},
        },
        3,
        {"x@a": Int(1, 3), "x@c": Int(1, 3)},
        (),
    ),
    (
        {"cmd": "ASSIGNMENT", "name": "x", "m": 3, "n": 2},
        3,
        {"x@1": Int(0, 2), "x@3": Int(0, 2)},
        (),
    ),
    (
        {"cmd": "ITEMSELECTION", "name": "x", "n": 4, "k": 2},
        4,
        {"x@1": Int(0, 1), "x@4": Int(0, 1)},
        (),
    ),
    (
        {
            "cmd": "BUDGETEDSELECTION",
            "name": "x",
            "n": 3,
            "budget_max": 10.0,
            "costs": [1.0, 2.0, 3.0],
        },
        3,
        {"x@1": Int(0, 1), "x@3": Int(0, 1)},
        (),
    ),
    # Item 1 is always in group 1; its distribution is the range of every item, which holds that
    # value.
    (
        {"cmd": "PARTITION", "name": "x", "m": 4, "k": 2},
        4,
        {"x@1": Int(1, 2), "x@4": Int(1, 2)},
        (),
    ),
    (
        {"cmd": "DAG", "name": "x", "nodes": ["a", "b"]},
        4,
        {"x@a,b": Int(0, 1), "x@b,a": Int(0, 1), "x@a,a": Int(0, 0)},
        ("x@a,a", "x@b,b"),
    ),
    (
        {"cmd": "TREE", "name": "x", "nodes": ["a", "b", "c"]},
        9,
        {"x@a,c": Int(0, 1), "x@c,c": Int(0, 0)},
        ("x@a,a", "x@b,b", "x@c,c"),
    ),
    (
        {"cmd": "PATH", "name": "x", "nodes": ["a", "b", "c"]},
        3,
        {"x@a": Int(0, 3), "x@c": Int(0, 3)},
        (),
    ),
    (
        {"cmd": "ANGLE", "name": "x", "min_deg": 0.0, "max_deg": 90.0},
        1,
        {"x@theta": F(0.0, 90.0)},
        (),
    ),
    (
        {"cmd": "ANGLE", "name": "x", "min_deg": 320.0, "max_deg": 40.0},
        1,
        {"x@theta": F(320.0, 400.0)},
        (),
    ),
    (
        {"cmd": "ANGLE", "name": "x", "min_deg": 360.0, "max_deg": 0.0},
        1,
        {"x@theta": F(360.0, 720.0)},
        (),
    ),
    (
        {"cmd": "ANGLE", "name": "x", "min_deg": -180.0, "max_deg": 180.0},
        1,
        {"x@theta": F(-180.0, 180.0)},
        (),
    ),
    (
        {
            "cmd": "SPHERICALANGLE",
            "name": "x",
            "min_theta_deg": -30.0,
            "max_theta_deg": 60.0,
            "min_phi_deg": 0.0,
            "max_phi_deg": 90.0,
        },
        2,
        {"x@theta": F(-30.0, 60.0), "x@phi": F(0.0, 90.0)},
        (),
    ),
    (
        {"cmd": "REGION", "name": "x", "vertices": [[0, 0], [4, 0], [0, 3]]},
        2,
        {"x@x": F(0.0, 4.0), "x@y": F(0.0, 3.0)},
        (),
    ),
    (
        {
            "cmd": "SPHERICALREGION",
            "name": "x",
            "vertices": [[10, 10], [10, 40], [-10, 40], [-10, 10]],
        },
        2,
        {"x@theta": F(-90.0, 90.0), "x@phi": F(-180.0, 180.0)},
        (),
    ),
    (
        {"cmd": "LOWRANK", "name": "x", "rows": 2, "cols": 3, "rank": 1, "min": 0.0, "max": 1.0},
        6,
        {"x@1,1": F(0.0, 1.0), "x@2,3": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "LOWRANKCORRECTION",
            "name": "x",
            "initial": ROW3,
            "rank": 1,
            "min": 0.0,
            "max": 1.0,
        },
        6,
        {"x@2,3": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "LOWRANKINCREMENTAL",
            "name": "x",
            "initial": ROW3,
            "rank": 1,
            "min": 0.0,
            "max": 1.0,
        },
        6,
        {"x@1,2": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "TIMESERIESVECTOR",
            "name": "x",
            "long_scale": 2,
            "short_scale": 3,
            "rank": 1,
            "min": 0.0,
            "max": 1.0,
        },
        6,
        {"x@1": F(0.0, 1.0), "x@6": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "TIMESERIESCORRECTIONVECTOR",
            "name": "x",
            "initial": [0.5] * 6,
            "long_scale": 2,
            "short_scale": 3,
            "rank": 1,
            "min": 0.0,
            "max": 1.0,
        },
        6,
        {"x@6": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "TIMESERIESINCREMENTALVECTOR",
            "name": "x",
            "initial": [0.5] * 6,
            "long_scale": 2,
            "short_scale": 3,
            "rank": 1,
            "min": 0.0,
            "max": 1.0,
        },
        6,
        {"x@1": F(0.0, 1.0)},
        (),
    ),
    (
        {"cmd": "BSPLINEVECTOR", "name": "x", "length": 5, "num_knots": 3, "min": 0.0, "max": 1.0},
        5,
        {"x@1": F(0.0, 1.0), "x@5": F(0.0, 1.0)},
        (),
    ),
    (
        {"cmd": "SPLINEVECTOR", "name": "x", "length": 5, "num_knots": 3, "min": 0.0, "max": 1.0},
        5,
        {"x@1": F(0.0, 1.0), "x@5": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "BSPLINECORRECTIONVECTOR",
            "name": "x",
            "initial": [0.5] * 5,
            "num_knots": 3,
            "min": 0.0,
            "max": 1.0,
        },
        5,
        {"x@5": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "SPLINECORRECTIONVECTOR",
            "name": "x",
            "initial": [0.5] * 5,
            "num_knots": 3,
            "min": 0.0,
            "max": 1.0,
        },
        5,
        {"x@1": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "BSPLINEMATRIX",
            "name": "x",
            "rows": 2,
            "cols": 3,
            "knot_rows": 2,
            "knot_cols": 2,
            "min": 0.0,
            "max": 1.0,
        },
        6,
        {"x@2,3": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "SPLINEMATRIX",
            "name": "x",
            "rows": 2,
            "cols": 3,
            "knot_rows": 2,
            "knot_cols": 2,
            "min": 0.0,
            "max": 1.0,
        },
        6,
        {"x@1,1": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "BSPLINECORRECTIONMATRIX",
            "name": "x",
            "initial": ROW3,
            "knot_rows": 2,
            "knot_cols": 2,
            "min": 0.0,
            "max": 1.0,
        },
        6,
        {"x@2,3": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "SPLINECORRECTIONMATRIX",
            "name": "x",
            "initial": ROW3,
            "knot_rows": 2,
            "knot_cols": 2,
            "min": 0.0,
            "max": 1.0,
        },
        6,
        {"x@1,3": F(0.0, 1.0)},
        (),
    ),
    (
        {"cmd": "FOURIERVECTOR", "name": "x", "length": 4, "num_freq": 2, "min": 0.0, "max": 1.0},
        4,
        {"x@1": F(0.0, 1.0), "x@4": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "FOURIERMATRIX",
            "name": "x",
            "rows": 2,
            "cols": 2,
            "freq_rows": 1,
            "freq_cols": 1,
            "min": 0.0,
            "max": 1.0,
        },
        4,
        {"x@2,2": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "FOURIERCORRECTIONVECTOR",
            "name": "x",
            "initial": [0.5] * 4,
            "num_freq": 2,
            "min": 0.0,
            "max": 1.0,
        },
        4,
        {"x@4": F(0.0, 1.0)},
        (),
    ),
    (
        {
            "cmd": "FOURIERCORRECTIONMATRIX",
            "name": "x",
            "initial": [[0.5, 0.5], [0.5, 0.5]],
            "freq_rows": 1,
            "freq_cols": 1,
            "min": 0.0,
            "max": 1.0,
        },
        4,
        {"x@1,2": F(0.0, 1.0)},
        (),
    ),
]
REGISTERED_COMMANDS = {stmt["cmd"] for stmt, *_ in REGISTERED}


def test_the_table_covers_every_variable_command():
    assert REGISTERED_COMMANDS == set(_variable_codec._HANDLERS) - {"SELECTION"}


@pytest.mark.parametrize(
    "stmt, count, expected, single",
    REGISTERED,
    ids=[f"{s['cmd']}-{i}" for i, (s, *_) in enumerate(REGISTERED)],
)
def test_each_command_registers_the_parameters_of_the_reference(stmt, count, expected, single):
    params = _params(dict(stmt))
    assert len(params) == count
    for name, dist in expected.items():
        assert params[name] == dist, name
    assert {name for name, dist in params.items() if dist.single()} == set(single)


def test_selection_registers_one_flag():
    params = _params({"cmd": "SELECTION", "name": "gate", "vars": ["x"]})
    assert params == {"gate": C([False, True])}


def test_names_given_as_a_set_or_a_view_become_a_list_of_text():
    """emit_model() writes the predecessors of a task in the order that it iterates them (a
    mapping by its keys), and a bytes value as one name."""
    commands = optquest.load_spec()["commands"]
    stmt = _variable_codec.coerce_fields(
        {
            "cmd": "TOPOLOGICALSEQUENCE",
            "name": "x",
            "tasks": ("a", "b", "c"),
            "predecessors": {"b": {"a"}, "c": frozenset({"b"})},
        },
        commands,
    )
    assert stmt["tasks"] == ["a", "b", "c"]
    assert stmt["predecessors"] == {"b": ["a"], "c": ["b"]}
    nodes = {"p": 1, "q": 2}.keys()
    assert _variable_codec.coerce_fields({"cmd": "DAG", "name": "g", "nodes": nodes}, commands)[
        "nodes"
    ] == ["p", "q"]
    assert _variable_codec._names(b"ab") == "b'ab'"
    assert _variable_codec._names({"a": 1}) == ["a"]


def test_overlapping_excluded_ranges_are_merged_and_touching_ones_are_kept():
    def ranges(*pairs):
        return [_exclusive("v", low, high) for low, high in pairs]

    def bounds(stmts):
        return [(s["min"], s["max"]) for s in stmts]

    merge = _variable_codec.merge_exclusive_ranges
    assert bounds(merge(ranges((6.0, 10.0), (3.0, 7.0)), {}, engine_built=False)) == [(3.0, 10.0)]
    assert bounds(merge(ranges((3.0, 7.0), (7.0, 10.0)), {}, engine_built=False)) == [
        (3.0, 7.0),
        (7.0, 10.0),
    ]
    assert bounds(merge(ranges((1.0, 2.0), (3.0, 4.0), (1.5, 3.5)), {}, engine_built=False)) == [
        (1.0, 4.0)
    ]
    with pytest.raises(ValueError, match="overlap"):
        merge(ranges((3.0, 7.0), (6.0, 10.0)), {}, engine_built=True)


def test_an_excluded_range_far_from_zero_keeps_its_grid_edges():
    """The edges are grid values; far from zero, (x - min) / step is a whole number only to
    within the rounding of x, and the check allows for it."""
    declared = {"x": {"cmd": "DISCRETE", "name": "x", "min": 1e6, "max": 1e6 + 1.0, "step": 1e-3}}
    for low, high in ((1000000.1, 1000000.2), (1000000.5, 1000000.6)):
        stmt = _exclusive("x", low, high)
        assert _variable_codec.fit_exclusive_range(stmt, declared, engine_built=True) is stmt
        assert _variable_codec.fit_exclusive_range(stmt, declared, engine_built=False) is stmt


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "CONTINUOUS", "name": "x", "min": 0.0, "max": 1e16},
        {"cmd": "INTEGER", "name": "x", "min": 0, "max": 2**1024},
        {"cmd": "DISCRETE", "name": "x", "min": 0.0, "max": 1e30, "step": 1.0},
        {"cmd": "ENUMERATION", "name": "x", "values": [1.0, 2.0, 1e22]},
        {"cmd": "CONTINUOUS", "name": "x", "min": float("nan"), "max": 1.0},
        {"cmd": "ANGLE", "name": "x", "min_deg": float("inf"), "max_deg": float("inf")},
        {"cmd": "VECTOR", "name": "x", "size": 2, "min": -1e16, "max": 0.0},
    ],
)
def test_a_bound_that_the_engine_does_not_take_is_refused_with_its_statement(stmt):
    """The engine takes a bound from -2**50 to 2**50 and refuses the model otherwise."""
    with pytest.raises(ValueError, match=rf"{stmt['cmd']} 'x'.*2\*\*50"):
        _params(stmt)


@pytest.mark.parametrize(
    "low, high, step",
    [
        (10000.0, 10000.0001, 0.0001),
        (1000.0, 1000.00001, 1e-5),
        (123456.789, 123456.79, 0.001),
        (1e7, 1e7 + 0.1, 0.1),
        # A min much larger than the max: the shortfall is thousands of ulps of the max.
        (-1.001, -0.001, 1.0),
        (-2.3, 0.001, 2.301),
        (-99.1, 0.13, 99.23),
        (-499.9, 0.2, 500.1),
        (-649009.7, 0.57, 649010.27),
        (-0.009999, 1e-06, 0.01),
    ],
)
def test_a_two_value_grid_far_from_zero_gets_an_engine_max_of_one_step(low, high, step):
    stmt = {"cmd": "DISCRETE", "name": "x", "min": low, "max": high, "step": step}
    engine_stmt, codec_stmt = _variable_codec.align_stepped_max(stmt, engine_built=False)
    assert engine_stmt["max"] - engine_stmt["min"] >= step
    assert codec_stmt["max"] == high
    # The engine's max is the same value for the engine, and for the decoder of the top value.
    assert abs(engine_stmt["max"] - high) <= 1e-9 * max(1.0, abs(low), abs(high))


# ---------------------------------------------------------------------------------------------
# Fifth review
# ---------------------------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(math, "ulp"), reason="math.ulp is new in Python 3.9")
def test_the_ulp_of_the_codec_is_the_one_of_math_ulp():
    """The codec computes ulps itself, because Python 3.8 has no math.ulp."""
    import random

    rng = random.Random(0)
    values = [0.0, 5e-324, 1e-310, 2.2250738585072014e-308, 0.75, 1.0, 1000.6, 2.0**50, 1e308]
    values += [1.7976931348623157e308, -5.5, float("inf")]
    values += [rng.uniform(-1.0, 1.0) * 10.0 ** rng.randint(-300, 300) for _ in range(2000)]
    for value in values:
        assert _variable_codec._ulp(value) == math.ulp(value), value
    assert math.isnan(_variable_codec._ulp(float("nan")))


@pytest.mark.parametrize(
    "var, low, high",
    [
        ({"cmd": "DISCRETE", "name": "x", "min": 0.0, "max": 1.3e-4, "step": 1e-5}, 3e-5, 6e-5),
        (
            {"cmd": "DISCRETE", "name": "x", "min": 1000.0, "max": 1000.00013, "step": 1e-5},
            1000.00003,
            1000.00006,
        ),
        ({"cmd": "DISCRETE", "name": "x", "min": 0.0, "max": 1e-4, "step": 5e-6}, 1e-5, 3e-5),
        ({"cmd": "ENUMERATION", "name": "x", "values": [1.0, 1.000005, 1.5, 2.0]}, 1.0, 2.0),
    ],
)
def test_a_range_that_excludes_a_value_within_1e_5_of_an_edge_is_refused(var, low, high):
    """The engine takes a value within 1e-5 of an edge for the edge, and proposes it."""
    declared = {"x": var}
    for engine_built in (False, True):
        with pytest.raises(ValueError, match="within 1e-5 of an edge"):
            _variable_codec.fit_exclusive_range(
                _exclusive("x", low, high), declared, engine_built=engine_built
            )


def test_a_range_on_a_step_above_1e_5_is_kept():
    declared = {"x": {"cmd": "DISCRETE", "name": "x", "min": 0.0, "max": 1.43e-4, "step": 1.1e-5}}
    stmt = _exclusive("x", 3.3e-5, 6.6e-5)
    assert _variable_codec.fit_exclusive_range(stmt, declared, engine_built=True) is stmt


# ---------------------------------------------------------------------------------------------
# Sixth review
# ---------------------------------------------------------------------------------------------

INF = float("inf")
# (variable, an edge inside its values that is one of them)
FAR_EDGE_VARS = [
    ({"cmd": "INTEGER", "name": "x", "min": 0, "max": 10}, 5.0),
    ({"cmd": "DISCRETE", "name": "x", "min": 0.0, "max": 10.0, "step": 0.5}, 5.0),
    ({"cmd": "DISCRETE", "name": "x", "min": 0.0, "max": 1e-3, "step": 1e-4}, 5e-4),
    ({"cmd": "TUPLE", "name": "x", "tuples": [[float(i), 0.0] for i in range(1, 11)]}, 5.0),
    ({"cmd": "GEOLOCATION", "name": "x", "coords": [[float(i), 0.0] for i in range(1, 11)]}, 5.0),
    ({"cmd": "CONTINUOUS", "name": "x", "min": 0.0, "max": 10.0}, 5.0),
    ({"cmd": "ENUMERATION", "name": "x", "values": [0.0, 2.0, 4.0, 5.0, 8.0, 10.0]}, 5.0),
]


@pytest.mark.parametrize("var, edge", FAR_EDGE_VARS, ids=[v["cmd"] for v, _ in FAR_EDGE_VARS])
@pytest.mark.parametrize("far", [INF, 1.7976931348623157e308, 1e308, 1e300])
def test_an_excluded_range_with_an_edge_far_outside_the_values_is_taken(var, edge, far):
    """A built model can hold an infinite edge, and statements= one of 1e308: every value inside
    such a range is nearer to the other edge. The step number of the far edge is beyond a float,
    or beyond an int."""
    declared = {"x": var}
    for stmt in (_exclusive("x", edge, far), _exclusive("x", -far, edge)):
        for engine_built in (False, True):
            assert (
                _variable_codec.fit_exclusive_range(stmt, declared, engine_built=engine_built)
                is stmt
            )
    if var["cmd"] != "CONTINUOUS":
        # An edge that is not a value still moves, or is refused, next to a far edge.
        odd = _exclusive("x", edge * 0.99, far)
        fitted = _variable_codec.fit_exclusive_range(odd, declared, engine_built=False)
        assert fitted["min"] < odd["min"] and fitted["max"] == far
        with pytest.raises(ValueError, match="Declare the range from"):
            _variable_codec.fit_exclusive_range(odd, declared, engine_built=True)
    for engine_built in (False, True):
        with pytest.raises(ValueError, match="EXCLUSIVE 'x' from"):
            _variable_codec.fit_exclusive_range(
                _exclusive("x", -far, far), declared, engine_built=engine_built
            )


def test_excluded_ranges_that_share_an_infinite_edge_overlap():
    declared = {"x": FAR_EDGE_VARS[0][0]}
    for pair in ([(5.0, INF), (7.0, 9.0)], [(-INF, 5.0), (3.0, 8.0)], [(2.0, INF), (4.0, INF)]):
        stmts = [_exclusive("x", a, b) for a, b in pair]
        with pytest.raises(ValueError, match="overlap"):
            _variable_codec.merge_exclusive_ranges(stmts, declared, engine_built=True)
    kept = [_exclusive("x", -INF, 3.0), _exclusive("x", 3.0, 5.0), _exclusive("x", 7.0, INF)]
    merged = _variable_codec.merge_exclusive_ranges(kept, declared, engine_built=True)
    assert [(m["min"], m["max"]) for m in merged] == [(-INF, 3.0), (3.0, 5.0), (7.0, INF)]


TOPO_ABC = {"cmd": "TOPOLOGICALSEQUENCE", "tasks": ["a", "b", "c"]}
TOPO_JOBS = {"cmd": "TOPOLOGICALSEQUENCE", "tasks": ["Design", "Build"]}
GRAPH_ABC = {"nodes": ["a", "b", "c"]}


@pytest.mark.parametrize(
    "stmt, name",
    [
        ({**TOPO_ABC, "predecessors": {"b": ["aa"]}}, "aa"),
        ({**TOPO_ABC, "predecessors": {"bb": ["a"]}}, "bb"),
        ({**TOPO_JOBS, "predecessors": {"Build": ["design"]}}, "design"),
        ({"cmd": "DAG", **GRAPH_ABC, "predecessors": {"b": ["a", "cc"]}}, "cc"),
        ({"cmd": "DAG", **GRAPH_ABC, "predecessors": {"b": "A"}}, "A"),
        ({"cmd": "TREE", **GRAPH_ABC, "predecessors": {"c": {"bb"}}}, "bb"),
        ({"cmd": "DAG", "nodes": [1, 2, 3], "predecessors": {2: [4]}}, "4"),
    ],
)
def test_a_predecessor_that_is_not_a_node_of_the_statement_is_refused(stmt, name):
    """The engine ignores a name in `predecessors` that is not one of the nodes or tasks, so a
    misspelled name would drop its order from the model with no error."""
    commands = optquest.load_spec()["commands"]
    stmt = {"name": "g", **stmt}
    optquest.emit_model([stmt])  # emit_model() takes it
    with pytest.raises(ValueError, match=rf"{stmt['cmd']} 'g': predecessors names '{name}'"):
        _variable_codec.coerce_fields(stmt, commands)


@pytest.mark.parametrize("cmd", ["DAG", "TREE", "TOPOLOGICALSEQUENCE"])
@pytest.mark.parametrize(
    "before, kept",
    [
        ("", None),
        (None, None),
        ([], []),
        ([None], []),
        (["a", ""], ["a"]),
        (["a", None], ["a"]),
        ("a", "a"),
    ],
)
def test_an_empty_predecessor_is_no_predecessor(cmd, before, kept):
    """emit_model() and the typed methods take an empty name or None among the predecessors of
    a node for no predecessor (a table read from a file with empty cells), so it is not a name
    that the statement does not have."""
    commands = optquest.load_spec()["commands"]
    field = "tasks" if cmd == "TOPOLOGICALSEQUENCE" else "nodes"
    stmt = {"cmd": cmd, "name": "g", field: ["a", "b"], "predecessors": {"b": before}}
    assert _variable_codec.coerce_fields(stmt, commands)["predecessors"] == {"b": kept}
    # The node itself is a name: an empty one is refused, as the typed method refuses it.
    stmt = {"cmd": cmd, "name": "g", field: ["a", "b"], "predecessors": {"": ["a"]}}
    with pytest.raises(ValueError, match=rf"{cmd} 'g': predecessors names ''"):
        _variable_codec.coerce_fields(stmt, commands)


@pytest.mark.parametrize(
    "stmt, field",
    [
        ({"cmd": "DAG", "start": ["a"]}, "start"),
        ({"cmd": "DAG", "end": {"c"}}, "end"),
        ({"cmd": "TREE", "root": ["a"]}, "root"),
        ({"cmd": "PATH", "path": {"start": ["a"]}}, "the start of its path"),
        ({"cmd": "PATH", "path": {"start": "a", "end": ["c"]}}, "the end of its path"),
    ],
)
def test_a_collection_for_the_one_node_of_a_field_is_refused(stmt, field):
    """emit_model() writes the list as text ("['a']"), and the typed method raised an
    AttributeError that named no statement."""
    commands = optquest.load_spec()["commands"]
    stmt = {"name": "g", "nodes": ["a", "b", "c"], **stmt}
    optquest.emit_model([stmt])  # emit_model() takes it
    with pytest.raises(ValueError, match=rf"{stmt['cmd']} 'g': {field} is .*one node"):
        _variable_codec.coerce_fields(stmt, commands)


def test_the_one_node_of_a_field_is_text():
    commands = optquest.load_spec()["commands"]
    dag = {"cmd": "DAG", "name": "g", "nodes": [1, 2, 3], "start": 1, "end": 3}
    coerced = _variable_codec.coerce_fields(dag, commands)
    assert (coerced["start"], coerced["end"]) == ("1", "3")
    path = {"cmd": "PATH", "name": "p", "nodes": [1, 2, 3], "path": {"start": 1, "required": [2]}}
    assert _variable_codec.coerce_fields(path, commands)["path"] == {
        "start": "1",
        "required": ["2"],
    }
    # An empty collection is no node, as None is: the statement builds the model without it.
    empty = {"cmd": "DAG", "name": "g", "nodes": ["a", "b"], "start": [], "end": ()}
    optquest.emit_model([empty])
    coerced = _variable_codec.coerce_fields(empty, commands)
    assert not coerced["start"] and not coerced["end"]
    assert (
        _variable_codec.build(coerced).params
        == _variable_codec.build({"cmd": "DAG", "name": "g", "nodes": ["a", "b"]}).params
    )


@pytest.mark.parametrize(
    "path, field",
    [
        ({"start_node": "a", "end_node": "d", "required_nodes": ["b"]}, "start_node"),
        ({"start": "a", "End": "d"}, "End"),
        ({"start": "a", "end": "d", "require": ["b"]}, "require"),
    ],
)
def test_a_field_that_the_path_of_a_path_does_not_have_is_refused(path, field):
    """emit_model() reads start, end and required from `path` and drops every other key, so
    the names of the typed method's arguments would build a PATH with no start and no end."""
    commands = optquest.load_spec()["commands"]
    stmt = {"cmd": "PATH", "name": "p", "nodes": ["a", "b", "c", "d"], "path": path}
    optquest.emit_model([stmt])  # emit_model() takes it
    with pytest.raises(ValueError, match=rf"PATH 'p': path has no field '{field}'.*start, end"):
        _variable_codec.coerce_fields(stmt, commands)
    kept = {"start": "a", "end": "d", "required": ["b"]}
    assert _variable_codec.coerce_fields({**stmt, "path": kept}, commands)["path"] == kept
