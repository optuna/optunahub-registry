# mypy: ignore-errors

# MIT License
#
# Copyright (c) 2026 OptTek Systems, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Shape-strategy codec bridging OptQuest's flat "<name>@<key>" component convention and
Optuna's per-parameter search space.

OptQuest exposes every non-scalar variable (and every shaped output) as a flat set of scalar
sub-values named "<name>@<key>"; there is no bulk get/set in the underlying engine (OQGetValue
/OQSetValue are scalar-only). Rather than one hand-written parser for each of the 58
decision-variable commands, this module implements the small number of distinct *shapes* those commands
take and dispatches every command to one of them.

Input to `build()` is an emitter statement dict: the same {"cmd": "DAG", "name": ..., ...} shape
that `optquest.emit_statement()` reads and `optquest.parse_model()` returns. Field names below
match that grammar exactly (see `_EMITTER_SPEC["commands"]` in optquest.py), so statements built
declaratively and statements recovered from a `describe()` reverse-parse work identically here.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
import decimal
import itertools
import math
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence

import numpy
import optuna


def _identity(v):
    return v


class VariableCodec:
    """One OptQuest variable's Optuna bridge:

    - `params`: which sub-parameters to register in the search space (name -> Distribution).
    - `to_engine`/`from_engine`: per-param value converters between the engine's raw float
      (the type that `OQGetValue`/`OQSetValue` use) and the Optuna-facing value (what the
      registered Distribution holds). Most shapes are identity (the engine value
      is the Optuna value), but the scalar-index shape (DESIGN/TUPLE/GEOLOCATION, and
      SELECTION's own near-identical variant) is not: the engine stores an index
      while Optuna holds the category *label*. A wrong direction here does not raise:
      OQGetValue/OQSetValue on a mismatched name return NaN, so an incorrect converter
      would pass NaN into a simulation without error.
    - `decode`: turn a dict of this variable's *Optuna-facing* values (i.e. already passed
      through `from_engine`, matching what ends up in `trial.params`) into one natural Python
      value. suggest_all() and wrap() use it; the plain suggest/write-back path does not.
      The shape depends on the command:

      - One number: CONTINUOUS, INTEGER, DISCRETE, BINARY, ENUMERATION, ANGLE, CHOICE.
      - The declared entry that the engine picked: DESIGN (an int from 1 to num), TUPLE (the
        row, a list) and GEOLOCATION (the coordinate pair, a list).
      - A list of the gated variable names when the gate is on, or an empty list: SELECTION.
      - A flat list with one value for each index from 1 to the count: ASSIGNMENT,
        ITEMSELECTION, BUDGETEDSELECTION, PARTITION.
      - A flat list for every command whose keyword ends in VECTOR (a TIMESERIESVECTOR has
        long_scale x short_scale values). A list of rows for every command whose keyword ends
        in MATRIX, and for LOWRANK, LOWRANKCORRECTION and LOWRANKINCREMENTAL. A list nested
        one level for each dimension for NDVARIABLE, INTEGERNVARIABLE, DISCRETENDVARIABLE and
        BINARYNDVARIABLE.
      - A dict keyed by the declared item name: PERMUTATION, SEQUENCE, MULTISTAGESEQUENCE,
        POLICYSEQUENCE, MIXTURE, MULTICHOICE, TOPOLOGICALSEQUENCE, PATH.
      - A dict with fixed keys: "theta" and "phi" for SPHERICALANGLE and SPHERICALREGION, "x"
        and "y" for REGION.
      - A dict keyed by the (from, to) tuple of every ordered pair of nodes: DAG, TREE.
    """

    __slots__ = ("params", "to_engine", "from_engine", "decode")

    def __init__(
        self,
        params: Mapping[str, "optuna.distributions.BaseDistribution"],
        decode: Callable[[Mapping[str, Any]], Any],
        *,
        to_engine: Mapping[str, Callable[[Any], float]] | None = None,
        from_engine: Mapping[str, Callable[[float], Any]] | None = None,
    ):
        self.params = dict(params)
        self.to_engine = (
            dict(to_engine) if to_engine is not None else {k: _identity for k in self.params}
        )
        self.from_engine = (
            dict(from_engine) if from_engine is not None else {k: _identity for k in self.params}
        )
        self.decode = decode


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------


# The engine takes a bound from -2**50 to 2**50 (Netlip.BIG, the largest number in its linear
# programs). It refuses the model, when it initializes it, for a variable with a larger bound.
ENGINE_MAX_BOUND = 2**50


def _ulp(x: float) -> float:
    """The distance from abs(x) to the next larger float, as math.ulp(x) gives it; Python 3.8
    has no math.ulp."""
    x = abs(float(x))
    if not math.isfinite(x):
        return x
    if x == 0.0:
        return 5e-324
    exponent = math.frexp(x)[1]  # x = m * 2**exponent, with 0.5 <= m < 1
    return math.ldexp(1.0, max(exponent - 53, -1074))


def check_engine_bounds(*values) -> None:
    """Raise ValueError for a bound that the engine does not take: one that is not a finite
    number, or one whose magnitude is above 2**50. A Python int is compared as it is, because
    float() of a very large int raises OverflowError."""
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            ok = abs(value) <= ENGINE_MAX_BOUND
        else:
            number = float(value)
            ok = math.isfinite(number) and abs(number) <= ENGINE_MAX_BOUND
        if not ok:
            raise ValueError(
                f"the bound {value!r} is not one that the engine takes: a bound must be a finite "
                f"number from -2**50 to 2**50 (1125899906842624)."
            )


def spans_a_step(low: float, high: float, step: float) -> bool:
    """Whether the range from `low` to `high` holds at least one step. In binary floating point,
    `high` - `low` can be a few ulps short of a whole number of steps (0.3 - 0.1 is
    0.19999999999999998), and the shortfall grows with the magnitude of the bounds. It is never
    taken for more than half a step: near 2**50, a few ulps are as large as a step of 1, and a
    range of one value holds no step."""
    shortfall = 8 * _ulp(max(abs(low), abs(high), step))
    return high - low >= step - min(shortfall, step / 2.0)


def _float(
    low: float, high: float, *, step: float | None = None
) -> optuna.distributions.FloatDistribution:
    check_engine_bounds(low, high)
    return optuna.distributions.FloatDistribution(float(low), float(high), step=step)


def _int(low: int, high: int) -> optuna.distributions.IntDistribution:
    check_engine_bounds(low, high)
    return optuna.distributions.IntDistribution(int(low), int(high))


# The engine counts the steps of an INTEGER or DISCRETE grid in a 32-bit int: it takes the
# number of steps as int(0.5 + (max - min) / step), and with more than this many it keeps the
# first 2**31 - 1 steps and proposes no value above them.
ENGINE_MAX_GRID_STEPS = 2**31 - 1


def fits_engine_grid(low, high, step) -> bool:
    """Whether the engine can hold the grid `low`, `low` + `step`, ..., up to `high`. A number
    of steps beyond the range of a float (a step of 1e-310) is not a grid that it holds."""
    steps = 0.5 + (float(high) - float(low)) / float(step)
    return math.isfinite(steps) and int(steps) <= ENGINE_MAX_GRID_STEPS


def _check_engine_grid(stmt, step: float) -> None:
    if not fits_engine_grid(stmt["min"], stmt["max"], step):
        raise ValueError(
            f"{stmt['cmd']} {stmt['name']!r}: the grid from {stmt['min']!r} to {stmt['max']!r} "
            f"with step {step!r} has more than 2**31 - 1 steps, and the engine proposes only "
            f"its first 2**31 values. Narrow the range, take a larger step, or declare a "
            f"CONTINUOUS variable (with bounds from -2**50 to 2**50) and round its value."
        )


def _whole_bounds(stmt) -> tuple[int, int]:
    """`min` and `max` of an INTEGER-type statement as ints. The engine's values are `min`,
    `min` + 1, ..., so with a `min` that is not a whole number they are not whole numbers, and
    an Optuna IntDistribution cannot hold them: Optuna would reject every engine value. The
    range must also fit the engine's grid (see ENGINE_MAX_GRID_STEPS)."""
    check_engine_bounds(stmt["min"], stmt["max"])
    low, high = float(stmt["min"]), float(stmt["max"])
    if not (low.is_integer() and high.is_integer()):
        raise ValueError(
            f"{stmt['cmd']} {stmt['name']!r}: min and max must be whole numbers, got "
            f"{stmt['min']!r} and {stmt['max']!r}. The engine's values are min, min + 1, ..., "
            f"which an Optuna IntDistribution cannot hold when min is not a whole number. Use "
            f"DISCRETE with step 1 for that grid."
        )
    _check_engine_grid(stmt, 1)
    return int(low), int(high)


# Optuna's stepped FloatDistribution holds a value only when (value - low) / step is within 1e-8
# of a whole number. The engine computes its grid values as low + k * step in binary floating
# point, and the rounding error of that quotient grows with max(|low|, |high|) / step: up to
# this ratio it stays below 1e-8. Whole numbers below 2**53 have no rounding error.
_STEPPED_RATIO_MAX = 2.0**23


def _stepped(low, high, step) -> optuna.distributions.FloatDistribution:
    """The distribution for the engine's grid `low`, `low` + `step`, ... whose top is `high`.

    It is stepped only when Optuna can hold every value of the grid. Optuna checks in decimal
    arithmetic, from the repr of each number, that `high` - `low` is a whole multiple of `step`.
    When it is not, Optuna moves `high` down (a stepped distribution would then reject the
    engine's top value), and a trial.suggest_float() call with that `high` would move it down
    again for a step such as 1/3. And far from zero, relative to the step (a step of 0.001 at
    1e6), the engine's values are too far from the grid in binary floating point for Optuna's
    tolerance. In those cases this returns the continuous distribution from `low` to `high`:
    every value of the engine's grid is in it."""
    check_engine_bounds(low, high)  # before the decimal arithmetic, which a huge bound breaks
    low, high = float(low), float(high)
    if step is None:
        return _float(low, high)
    step = float(step)
    d_low, d_high, d_step = (decimal.Decimal(str(v)) for v in (low, high, step))
    size = max(abs(low), abs(high))
    exact = size <= 2.0**53 and all(v.is_integer() for v in (low, high, step))
    if (
        d_step > 0
        and (d_high - d_low) % d_step == 0
        and (exact or size / step <= _STEPPED_RATIO_MAX)
    ):
        return _float(low, high, step=step)
    return _float(low, high)


def narrow_grid_problem(low, high):
    """What is wrong with a grid from `low` to `high` that the engine takes for one value, or
    None. The engine takes two numbers within its epsilon of 1e-5 for one number, and it tests
    the bounds of a variable that way (OptQuestDefinition.EPSILON_EQ; the test here is the
    engine's, in the same arithmetic). It then proposes `high` and `low` only, and next to
    another variable it stays on `high`. A grid whose `low` is its `high` has one value, and is
    not narrow."""
    low, high = float(low), float(high)
    if low < high and high - _ENGINE_EPSILON < low:
        return (
            f"the engine takes the ends of the grid, {low!r} and {high!r}, for one value, "
            f"because they are within its tolerance of 1e-5 of each other ({high!r} - 1e-5 < "
            f"{low!r} in floating point): it would propose no value between them. Give the "
            f"values in a larger unit, so that the grid is wider than 1e-5, or use a variable "
            f"without a step."
        )
    return None


def _grid_step(stmt) -> float:
    """The `step` of a DISCRETE-type statement. It must be a positive number: Optuna's
    distributions take only a positive step. The grid must also fit the engine's grid (see
    ENGINE_MAX_GRID_STEPS), and be one that the engine does not take for one value (see
    narrow_grid_problem)."""
    step = float(stmt["step"])
    if not step > 0.0:  # also refuses NaN
        raise ValueError(
            f"{stmt['cmd']} {stmt['name']!r}: step must be a positive number, got "
            f"{stmt['step']!r}. Optuna's distributions take only a positive step."
        )
    check_engine_bounds(stmt["min"], stmt["max"])
    _check_engine_grid(stmt, step)
    problem = narrow_grid_problem(stmt["min"], stmt["max"])
    if problem is not None:
        raise ValueError(f"{stmt['cmd']} {stmt['name']!r}: {problem}")
    return step


def _broadcast(value, n: int, default: float) -> list[float]:
    """MIXTURE-style optional fields accept either a single number (apply to every component)
    or a per-component list (`int_or_csv_str`/`float_or_csv_float` in the emitter grammar)."""
    if value is None:
        return [default] * n
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"expected {n} values, got {len(value)}: {value!r}")
        return [float(v) for v in value]
    return [float(value)] * n


def _bbox(points: Sequence[Sequence[float]], axis: int) -> tuple[float, float]:
    values = [p[axis] for p in points]
    return min(values), max(values)


# ---------------------------------------------------------------------------------------------
# shape strategies
# ---------------------------------------------------------------------------------------------


def _scalar_bare(name: str, dist: "optuna.distributions.BaseDistribution") -> VariableCodec:
    """A single Optuna param under the variable's own name, value passed through unchanged."""
    return VariableCodec({name: dist}, lambda raw: raw[name])


def _fixed_subkeys(
    name: str, fields: Mapping[str, "optuna.distributions.BaseDistribution"]
) -> VariableCodec:
    """A small, fixed set of named sub-keys, e.g. ANGLE's "<name>@theta", CHOICE's
    "<name>@choice", REGION's "<name>@x"/"<name>@y"."""
    params = {f"{name}@{label}": dist for label, dist in fields.items()}
    if len(fields) == 1:
        (label,) = fields.keys()

        def decode(raw):
            return raw[f"{name}@{label}"]

    else:

        def decode(raw):
            return {label: raw[f"{name}@{label}"] for label in fields}

    return VariableCodec(params, decode)


def _index_keyed(
    name: str, count: int, dist_factory: Callable[[int], "optuna.distributions.BaseDistribution"]
) -> VariableCodec:
    """1-based-index sub-keys "<name>@1" to "<name>@<count>": ASSIGNMENT, PARTITION,
    ITEMSELECTION, BUDGETEDSELECTION."""
    params = {f"{name}@{i}": dist_factory(i) for i in range(1, count + 1)}

    def decode(raw):
        return [raw[f"{name}@{i}"] for i in range(1, count + 1)]

    return VariableCodec(params, decode)


def _name_keyed(
    name: str,
    items: Sequence[str],
    dist_factory: Callable[[str], "optuna.distributions.BaseDistribution"],
) -> VariableCodec:
    """Item-name sub-keys "<name>@<item>": PERMUTATION, SEQUENCE, MULTISTAGESEQUENCE,
    POLICYSEQUENCE, MIXTURE, MULTICHOICE, TOPOLOGICALSEQUENCE, PATH. Decodes to a dict keyed
    by item name."""
    params = {f"{name}@{item}": dist_factory(item) for item in items}

    def decode(raw):
        return {item: raw[f"{name}@{item}"] for item in items}

    return VariableCodec(params, decode)


def _grid(
    name: str,
    dims: Sequence[int],
    dist_factory: Callable[[], "optuna.distributions.BaseDistribution"],
) -> VariableCodec:
    """1-D "<name>@i", 2-D "<name>@row,col", or N-D "<name>@i1,i2,..." sub-keys: every
    vector, matrix and ND command and the entire `smooth` family, which all share this one
    namespace."""
    dims = tuple(dims)
    indices = list(itertools.product(*(range(1, d + 1) for d in dims)))
    params = {f"{name}@" + ",".join(map(str, idx)): dist_factory() for idx in indices}

    def decode(raw):
        def build(prefix, remaining_dims):
            if not remaining_dims:
                key = f"{name}@" + ",".join(map(str, prefix))
                return raw[key]
            return [
                build(prefix + (i,), remaining_dims[1:]) for i in range(1, remaining_dims[0] + 1)
            ]

        return build((), dims)

    return VariableCodec(params, decode)


def _all_pairs(
    name: str,
    nodes: Sequence[str],
    dist_factory: Callable[[], "optuna.distributions.BaseDistribution"],
) -> VariableCodec:
    """ "<name>@<from>,<to>" for every ordered pair of nodes, including a node paired with
    itself: DAG, TREE. These are the names the engine reports: the full n by n adjacency
    matrix. The diagonal is always 0, so its Distribution holds only 0. DAG and TREE use the
    same names. For a TREE, in every solution that the engine generates, the edges form a spanning
    tree, so this codec does not need to encode that rule."""
    pairs = [(a, b) for a in nodes for b in nodes]
    params = {f"{name}@{a},{b}": (_int(0, 0) if a == b else dist_factory()) for a, b in pairs}

    def decode(raw):
        return {(a, b): raw[f"{name}@{a},{b}"] for a, b in pairs}

    return VariableCodec(params, decode)


def _scalar_index(name: str, choices: Sequence) -> VariableCodec:
    """A single CategoricalDistribution picking one whole entry from a declared list; the raw
    engine value is a 1-based index into `choices`: DESIGN, TUPLE, GEOLOCATION. (SELECTION is
    the same idea but has its own handler, _selection below, because it has a 4th "nothing
    selected" state at index 0 that these three don't have. ENUMERATION looks similar but is
    not this shape either: its raw engine value is the value, not an index.
    set_value(name, 2) on values [1.5, 2.5, 3.5] sets the out-of-domain literal 2.0 rather than
    selecting entry 2. See _enumeration below.)

    The engine uses indices (floats); Optuna's CategoricalDistribution uses the string
    labels below. to_engine/from_engine do that translation explicitly."""
    choices = list(choices)
    labels = [str(c) for c in choices]
    dist = optuna.distributions.CategoricalDistribution(labels)

    def from_engine(raw_index):
        idx = int(round(raw_index)) - 1
        if not (0 <= idx < len(labels)):
            raise ValueError(
                f"{name!r}: engine returned index {raw_index!r}, out of range for "
                f"{len(labels)} declared choice(s)"
            )
        return labels[idx]

    def to_engine(label):
        return float(labels.index(label) + 1)

    def decode(raw):
        # A copy of a TUPLE row or a GEOLOCATION pair: sim() can change the list that it gets,
        # and a later trial must still get the declared row.
        choice = choices[labels.index(raw[name])]
        return list(choice) if isinstance(choice, list) else choice

    return VariableCodec(
        {name: dist}, decode, to_engine={name: to_engine}, from_engine={name: from_engine}
    )


# ---------------------------------------------------------------------------------------------
# per-command handlers
# ---------------------------------------------------------------------------------------------


def _continuous(stmt):
    return _scalar_bare(stmt["name"], _float(stmt["min"], stmt["max"]))


def _integer(stmt):
    return _scalar_bare(stmt["name"], _int(*_whole_bounds(stmt)))


def _discrete(stmt):
    return _scalar_bare(stmt["name"], _stepped(stmt["min"], stmt["max"], _grid_step(stmt)))


def _binary(stmt):
    return _scalar_bare(stmt["name"], _int(0, 1))


def _design(stmt):
    return _scalar_index(stmt["name"], list(range(1, int(stmt["num"]) + 1)))


def _enumeration(stmt):
    # The raw engine value is one of the declared values and not an index (see
    # _scalar_index's docstring), so it must not go through the index-based shape.
    name = stmt["name"]
    check_engine_bounds(*stmt["values"])
    values = [float(v) for v in stmt["values"]]
    dist = optuna.distributions.CategoricalDistribution(values)
    return VariableCodec({name: dist}, lambda raw: raw[name])


def _element_distribution(stmt):
    """The distribution of each element of a VECTOR/MATRIX (continuous) or a DISCRETEVECTOR/
    DISCRETEMATRIX (stepped) statement. Every element has the same one."""
    step = _grid_step(stmt) if stmt.get("step") is not None else None
    return _stepped(stmt["min"], stmt["max"], step)


def _vector_like(count_field: str):
    def handler(stmt):
        dist = _element_distribution(stmt)
        return _grid(stmt["name"], (int(stmt[count_field]),), lambda: dist)

    return handler


def _matrix_like(rows_field: str, cols_field: str):
    def handler(stmt):
        dist = _element_distribution(stmt)
        rows, cols = int(stmt[rows_field]), int(stmt[cols_field])
        return _grid(stmt["name"], (rows, cols), lambda: dist)

    return handler


def _binary_vector(stmt):
    return _grid(stmt["name"], (int(stmt["k"]),), lambda: _int(0, 1))


def _binary_matrix(stmt):
    return _grid(stmt["name"], (int(stmt["rows"]), int(stmt["cols"])), lambda: _int(0, 1))


def _integer_vector(stmt):
    bounds = _whole_bounds(stmt)
    return _grid(stmt["name"], (int(stmt["size"]),), lambda: _int(*bounds))


def _integer_matrix(stmt):
    bounds = _whole_bounds(stmt)
    return _grid(stmt["name"], (int(stmt["rows"]), int(stmt["cols"])), lambda: _int(*bounds))


def _nd(dist_factory_kind: str):
    def handler(stmt):
        dims = [int(d) for d in stmt["dims"]]
        if dist_factory_kind == "float":
            dist = _float(stmt["min"], stmt["max"])
        elif dist_factory_kind == "int":
            dist = _int(*_whole_bounds(stmt))
        elif dist_factory_kind == "discrete":
            dist = _stepped(stmt["min"], stmt["max"], _grid_step(stmt))
        else:
            dist = _int(0, 1)  # binary
        return _grid(stmt["name"], dims, lambda: dist)

    return handler


def _permutation(stmt):
    items = list(stmt["items"])
    n = len(items)
    return _name_keyed(stmt["name"], items, lambda item: _int(1, n))


def _tuple_var(stmt):
    # Real rows (list-of-lists), and not pre-joined strings: decode() returns a copy of the
    # selected entry of _scalar_index's `choices`, so a join here would make decode() return a
    # comma-joined string instead of the tuple. `labels` (the Optuna-facing
    # CategoricalDistribution side) is still str(choices)-derived inside _scalar_index, so this
    # does not change the choice labels that Optuna tracks for the distribution.
    return _scalar_index(stmt["name"], [list(row) for row in stmt["tuples"]])


def _geolocation(stmt):
    return _scalar_index(stmt["name"], [list(coord) for coord in stmt["coords"]])


def _selection(stmt):
    """SELECTION is a binary gate over a group, not a choice among its members.

    The engine reports the gate as 0.0 (the gated variables are forced to their lower bounds
    for that solution) or 1.0 (they participate). The Optuna parameter is a
    CategoricalDistribution([False, True]); `decode` returns the list of gated variable names
    when the gate is on and an empty list when it is off, the natural Python value for "which
    variables are active".
    """
    name = stmt["name"]
    sub_vars = list(stmt["vars"])
    dist = optuna.distributions.CategoricalDistribution([False, True])

    def from_engine(raw_flag):
        flag = int(round(float(raw_flag)))
        if flag not in (0, 1):
            raise ValueError(
                f"{name!r}: SELECTION is a binary gate, but the engine returned "
                f"{raw_flag!r}. Expected 0 (the group is forced to its lower bounds) or 1 "
                f"(the group participates); it gates {sub_vars!r}."
            )
        return bool(flag)

    def to_engine(value):
        return 1.0 if value else 0.0

    return VariableCodec(
        {name: dist},
        lambda raw: list(sub_vars) if raw[name] else [],
        to_engine={name: to_engine},
        from_engine={name: from_engine},
    )


# The engine treats an arc as the full circle when |max - min| is at least this (in degrees).
# This codec uses the same test, so that the arc it registers is the engine's arc.
_FULL_CIRCLE_DEG = 360.0 - 1e-9


def _arc_top(low: float, high: float) -> float:
    """The upper bound of the Optuna distribution for the arc that the engine declares from `low`
    to `high`. The arc goes from `low` in the direction of increasing angle, so the bound is
    `high` when `high` is above `low`. When `low` is greater than `high` (an arc such as 320 to
    40, which crosses 0), Optuna cannot take FloatDistribution(320, 40), and the bound is
    `high` plus 360 (here 400), so that 10 degrees is the value 370. A full circle that ends
    below its start (`low` - `high` >= 360) is `low` to `low` + 360.
    """
    if low <= high:
        return high
    if low - high >= _FULL_CIRCLE_DEG:
        return low + 360.0
    return high + 360.0


def _azimuth_decoder(low: float, top: float) -> Callable[[float], float]:
    """Move an angle that the engine reports in [0, 360) onto the registered range [low, top].

    The engine reports an angle as its equivalent in [0, 360), not in the range that you
    declare: an ANGLE declared from -45 to 45 reports 315 for -45, and so does an ANGLE declared
    from -180 to 180. If this codec gave Optuna the value 315, Optuna would reject it as outside
    the distribution, put a random value in its place, and send that value back to the engine.
    This function adds or subtracts 360 so that the value is in [low, low + 360), which holds
    every angle of the arc once. `top` is the upper bound from _arc_top().
    """

    def decode(value):
        wrapped = low + math.fmod(float(value) - low, 360.0)
        if wrapped < low:
            wrapped += 360.0
        # Clamp for the same floating-point reason the log decoders clamp: at a bound, fmod
        # can give a value one ulp outside, and the distribution's bounds are inclusive.
        return min(top, max(low, wrapped))

    return decode


def _azimuth_encoder(value) -> float:
    """An angle of the registered range as the engine reports it, in [0, 360): the inverse of
    _azimuth_decoder. The sampler gives the engine back the angle of each trial, and a
    constraint or an expression on the angle is judged on that value, so it must be the value
    that the engine proposed (315 for the -45 of an angle declared from -180 to 180)."""
    deg = math.fmod(float(value), 360.0)
    if deg < 0.0:
        deg += 360.0
    if deg >= 360.0:  # a tiny negative angle rounds to 360.0 above
        deg = 0.0
    return deg + 0.0  # not -0.0


def _angle(stmt):
    name = stmt["name"]
    low, high = float(stmt["min_deg"]), float(stmt["max_deg"])
    top = _arc_top(low, high)
    codec = _fixed_subkeys(name, {"theta": _float(low, top)})
    # The engine reports the ANGLE value in [0, 360). (The SPHERICALANGLE theta is an
    # elevation, and the engine reports it as it is.)
    codec.from_engine[f"{name}@theta"] = _azimuth_decoder(low, top)
    codec.to_engine[f"{name}@theta"] = _azimuth_encoder
    return codec


def _spherical_angle(stmt):
    name = stmt["name"]
    phi_low, phi_high = float(stmt["min_phi_deg"]), float(stmt["max_phi_deg"])
    phi_top = _arc_top(phi_low, phi_high)
    codec = _fixed_subkeys(
        name,
        {
            "theta": _float(stmt["min_theta_deg"], stmt["max_theta_deg"]),
            "phi": _float(phi_low, phi_top),
        },
    )
    # phi is the azimuth, and the engine reports it in [0, 360). theta is the elevation, and
    # the engine reports it as it is.
    codec.from_engine[f"{name}@phi"] = _azimuth_decoder(phi_low, phi_top)
    codec.to_engine[f"{name}@phi"] = _azimuth_encoder
    return codec


# Commands whose values are on the grid min + k * step, with the fields min, max and step.
STEPPED_CMDS = frozenset({"DISCRETE", "DISCRETEVECTOR", "DISCRETEMATRIX", "DISCRETENDVARIABLE"})


def _optuna_grid_top(low: float, high: float, step: float) -> float:
    """The largest value of the grid `low` + k * `step` that is not above `high`, computed as
    FloatDistribution(low, high, step=step) computes its `high`: once, in decimal arithmetic,
    from the repr of each number. For some steps (such as 1/3) Optuna would move this top down
    again if it got it back as `high`; _stepped then registers the continuous range of the grid,
    so that no value of the grid is lost."""
    d_low, d_high, d_step = (decimal.Decimal(str(float(v))) for v in (low, high, step))
    d_range = d_high - d_low
    if d_range % d_step == 0:
        return float(high)
    return float((d_range // d_step) * d_step + d_low)


# The engine's tolerance for comparing two values (OptQuestDefinition.EPSILON).
_ENGINE_EPSILON = 1e-5


def _engine_grid_top(low: float, high: float, step: float) -> float:
    """The largest value that the engine uses for a DISCRETE variable declared from `low` to
    `high` (COptQuestDiscreteVariable.makeValid in the engine). The engine computes the grid
    value nearest to `high`, `low` + n * `step` with n = int(0.5 + (`high` - `low`) / `step`).
    When that value is within 1e-5 of `high`, the engine keeps `high` as its upper bound, and
    gives `high` itself for the top of the grid. Otherwise the upper bound is that grid value,
    which can be above `high`: from 0 to 1 with step 0.6, the grid is 0, 0.6, 1.2. That top is
    returned as computed in decimal arithmetic, as Optuna computes a top; the engine's value
    differs from it by a rounding error, which _FitToDistribution removes."""
    count = int(0.5 + (high - low) / step)
    nearest = low + count * step
    if high - _ENGINE_EPSILON < nearest < high + _ENGINE_EPSILON:
        return high
    return float(count * decimal.Decimal(str(step)) + decimal.Decimal(str(low)))


def _next_up(x: float) -> float:
    return float(numpy.nextafter(x, numpy.inf))


def align_stepped_max(stmt: Mapping[str, Any], *, engine_built: bool):
    """Make the engine and Optuna agree on the largest value of a DISCRETE-type grid. Returns
    (the statement to build the engine model from, the statement to build the codec from).

    When `max` - `min` is not a whole multiple of `step`, the engine and Optuna disagree about
    the largest value: the engine takes the grid value nearest to `max`, which can be above
    `max`, and Optuna takes the largest grid value that is not above `max`. If they disagree,
    Optuna rejects every engine value above its top and samples a random value in its place.

    - engine_built=False (statements=): the sampler builds the engine model, so the codec gets
      Optuna's top as `max`, and so does the engine (see engine_grid_max).
    - engine_built=True (model=): the caller built the engine model, and its grid cannot be
      changed. The codec gets the engine's top as `max`, so that the registered distribution is
      the engine's grid (see _stepped for the grids that Optuna cannot hold as a stepped
      distribution).

    Other statements, and a statement whose fields are not a grid of two or more values (the
    engine or the codec reports that one), are returned as they are."""
    if str(stmt.get("cmd", "")).upper() not in STEPPED_CMDS:
        return stmt, stmt
    try:
        low, high, step = float(stmt["min"]), float(stmt["max"]), float(stmt["step"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return stmt, stmt
    if step > 0.0 and math.isfinite(step) and not fits_engine_grid(low, high, step):
        return stmt, stmt  # a grid that the engine cannot hold: the codec reports it
    # A range that is a few ulps shorter than one step in binary floating point is still a
    # grid of two values: 0.3 - 0.1 is 0.19999999999999998, and Optuna's grid from 0.1 to 0.3
    # with step 0.2 is 0.1 and 0.3 (see spans_a_step). A bound that the engine does not take is
    # the codec's to report.
    if (
        not all(math.isfinite(v) for v in (low, high, step))
        or max(abs(low), abs(high)) > ENGINE_MAX_BOUND
        or not step > 0.0
        or not spans_a_step(low, high, step)
    ):
        return stmt, stmt
    if engine_built:
        top = _engine_grid_top(low, high, step)
        return stmt, (stmt if top == high else {**stmt, "max": top})
    top = _optuna_grid_top(low, high, step)
    if not spans_a_step(low, top, step):
        return stmt, stmt  # less than one step in decimal: the engine reports the statement
    engine_max = engine_grid_max(low, top, step)
    codec_stmt = stmt if top == high else {**stmt, "max": top}
    engine_stmt = stmt if engine_max == high else {**stmt, "max": engine_max}
    return engine_stmt, codec_stmt


def engine_grid_max(low: float, top: float, step: float) -> float:
    """The `max` to give the engine for a grid whose top value is `top`.

    The engine refuses a variable whose range is shorter than one step. With a grid of two
    values, the top minus `low` can be a few ulps shorter than `step` in binary floating point
    (0.3 - 0.1 is 0.19999999999999998 for a step of 0.2), so the engine gets the next float up
    from `top` that gives a range of one step. Its grid is the same two values: it keeps that
    `max` as its upper bound (it is within 1e-5 of the grid value) and gives it for the top
    value, which _FitToDistribution moves back onto `top`.

    The shortfall is a few ulps of the larger of the two bounds. When that is `low` (from
    -499.9 to 0.2 with a step of 500.1), it is thousands of ulps of `top`, and the search goes
    on from `low` + `step` by an ulp of `low`."""
    engine_max = float(top)
    for _ in range(64):
        if engine_max - low >= step:
            return engine_max
        engine_max = _next_up(engine_max)
    engine_max = max(float(top), low + step)
    ulp = _ulp(max(abs(low), abs(engine_max)))
    for _ in range(64):
        if engine_max - low >= step:
            break
        engine_max += ulp
    return engine_max


# ---------------------------------------------------------------------------------------------
# EXCLUSIVE ranges
# ---------------------------------------------------------------------------------------------

# The variable types whose values the check of an EXCLUSIVE range knows. An EXCLUSIVE range
# names a scalar variable, or one element ("<name>@<key>") of a vector, matrix or ND variable.
# The engine refuses an exclusive range on a BINARY or DESIGN variable, so the check leaves
# those to the engine, whose error then names the statement as it was given.
_INTERVAL_CMDS = frozenset({"CONTINUOUS", "VECTOR", "MATRIX", "NDVARIABLE"})
_INTEGER_CMDS = frozenset({"INTEGER", "INTEGERVECTOR", "INTEGERMATRIX", "INTEGERNVARIABLE"})
_SCALAR_CMDS = frozenset({"CONTINUOUS", "INTEGER", "DISCRETE", "BINARY", "DESIGN", "ENUMERATION"})


class _Values:
    """The values that a variable can take: every number from `low` to `top` (no `step`), the
    grid `low` + k * `step` and `top`, or the sorted list `points`. The grid values are computed
    as the engine computes them, in binary floating point. `top` is a value of its own because
    the engine can keep a `max` that is within 1e-5 of the grid value nearest to it (see
    _engine_grid_top); a grid value above `top` is then not a value."""

    def __init__(self, low=None, top=None, step=None, points=None):
        self.low, self.top, self.step, self.points = low, top, step, points
        if points is not None:
            # The smallest distance between two of the points (see edge_tol).
            gaps = [after - before for before, after in zip(points, points[1:]) if after > before]
            self.spacing = min(gaps) if gaps else math.inf
            return
        # The rounding error of a value: 1e-9 of the largest bound (1e-9 for bounds below 1).
        # Far from zero that is more than the distance between two values (10 at 1e10), and a
        # number between two values, or beyond the last one, would count as a value. So it is
        # held to the rounding error of numbers that are as far apart as the values are.
        magnitude = max(1.0, abs(low), abs(top))
        self.tol = 1e-9 * magnitude
        if step is None:
            self.tol = min(self.tol, max(1e-9 * abs(top - low), _ulp(magnitude)))
            return
        # The tolerance of a step number k = (x - low) / step. A grid value x far from zero
        # has a rounding error of a few ulps of x, which is a larger part of a small step:
        # at 1000.6 with a step of 1e-5, k is about 4e-8 from a whole number. It is never more
        # than an eighth of a step: near 2**50, 64 ulps are several steps of 1.
        ulps = 64 * _ulp(magnitude)
        self.ktol = min(max(1e-9, ulps / step), 0.125)
        self.tol = min(self.tol, self.ktol * abs(step))
        last = int(0.5 + (top - low) / step)
        self.last = last if low + last * step <= top + self.tol else last - 1

    def edge_tol(self, x):
        """The rounding error of an edge of an EXCLUSIVE range next to the value `x` (see
        _edge_tol), held to the rounding error of a value, so that an edge between two values
        is not one of them."""
        if self.points is None:
            return min(_edge_tol(x), self.tol)
        between = max(1e-9 * self.spacing, 64 * _ulp(max(1.0, abs(x))))
        return min(_edge_tol(x), between, self.spacing / 8.0)

    def _grid(self, k):
        return self.low + k * self.step

    def _step_number(self, x):
        """(x - low) / step, held to just outside the step numbers of the grid: floor() and
        ceil() of it are then ints for a number far outside the values too (an infinite edge of
        an excluded range, or an edge of 1e308 on a small step)."""
        beyond = 1.0 + self.ktol
        return min(max((x - self.low) / self.step, -beyond), self.last + beyond)

    def contains(self, x):
        if self.points is not None:
            return x in self.points
        if not self.low - self.tol <= x <= self.top + self.tol:
            return False
        if self.step is None or abs(x - self.top) <= self.tol:
            return True
        k = (x - self.low) / self.step
        return abs(k - round(k)) <= self.ktol and 0 <= round(k) <= self.last

    def at_edge(self, x, *, lower):
        """Whether the edge `x` of an excluded range is a value: the lower edge when `lower`,
        or the upper edge. On a grid, a number within a rounding error of a value is that value.
        In a list of points, Optuna holds the points as the choices of a CategoricalDistribution,
        which takes no other number, and the side counts. An edge one rounding error outside a
        point (a lower edge just above it, an upper edge just below it) leaves the point outside
        the range, and the engine proposes the point. An edge one rounding error on the other
        side (0.3 below the 0.30000000000000004 of 0.1 * 3) has the point inside the range: the
        engine moves it to the edge and proposes the edge, so that edge is not a value."""
        if self.points is None:
            return self.contains(x)
        if lower:
            return any(0.0 <= x - p <= self.edge_tol(p) for p in self.points)
        return any(0.0 <= p - x <= self.edge_tol(p) for p in self.points)

    def nearest(self, x):
        """The value nearest to `x`."""
        if self.points is not None:
            return min(self.points, key=lambda p: abs(p - x))
        if self.step is None:
            return min(self.top, max(self.low, x))
        k = min(self.last, max(0, round((x - self.low) / self.step)))
        value = self._grid(k)
        # The grid value at `top` is `top`, which it can miss by a rounding error.
        if abs(value - self.top) <= self.tol or abs(self.top - x) < abs(value - x):
            return self.top
        return value

    def below(self, x):
        """The largest value that is not above `x`, or None."""
        if self.points is not None:
            found = [p for p in self.points if p <= x]
            return found[-1] if found else None
        if x < self.low:
            return None
        if x >= self.top:
            return self.top
        if self.step is None:
            return x
        return self._grid(min(self.last, math.floor((x - self.low) / self.step + self.ktol)))

    def above(self, x):
        """The smallest value that is not below `x`, or None."""
        if self.points is not None:
            found = [p for p in self.points if p >= x]
            return found[0] if found else None
        if x > self.top:
            return None
        if x <= self.low:
            return self.low
        if self.step is None:
            return x
        k = math.ceil((x - self.low) / self.step - self.ktol)
        return self._grid(k) if k <= self.last else self.top

    def inside_near_an_edge(self, a, b, eps):
        """A value v with a < v < b that is within `eps` of a or b, or None. Every number from
        `low` to `top` (no `step`) gives None: the engine moves such a value to the edge."""
        if self.points is not None:
            inside = [p for p in self.points if a < p < b]
        elif self.step is None:
            return None
        else:
            inside = []
            k = max(0, math.floor(self._step_number(a) + self.ktol) + 1)
            if k <= self.last and a < self._grid(k) < b:
                inside.append(self._grid(k))
            k = min(self.last, math.ceil(self._step_number(b) - self.ktol) - 1)
            if k >= 0 and a < self._grid(k) < b:
                inside.append(self._grid(k))
            if a < self.top < b:
                inside.append(self.top)
        # A value within a rounding error of an edge is the edge (the grid value
        # 1.5000000000000002 and the edge 1.5), not a value that the range excludes.
        inside = [v for v in inside if a + self.edge_tol(v) < v < b - self.edge_tol(v)]
        limit = eps * (1.0 + 1e-6)  # a distance of eps, give or take its rounding
        for v in inside:
            if v - a <= limit or b - v <= limit:
                return v
        return None

    def any_between(self, a, b, *, from_a):
        """Whether a value v has a < v < b, or a <= v < b when `from_a`."""
        if self.points is not None:
            return any((a <= p if from_a else a < p) and p < b for p in self.points)
        if self.step is None:
            if from_a or a < self.low:
                v = max(a, self.low)
                return v <= self.top and v < b
            return a < self.top and a < b
        k_a = self._step_number(a)
        k = max(0, math.ceil(k_a - self.ktol) if from_a else math.floor(k_a + self.ktol) + 1)
        if k <= self.last and self._grid(k) < b:
            return True
        return (a <= self.top if from_a else a < self.top) and self.top < b


#: The commands whose elements have bounds that the engine keeps fixed, so it refuses an
#: EXCLUSIVE range on one of them.
_FIXED_ELEMENT_CMDS = frozenset(
    {
        "BINARYVECTOR",
        "BINARYMATRIX",
        "BINARYNDVARIABLE",
        "PERMUTATION",
        "ITEMSELECTION",
        "BUDGETEDSELECTION",
    }
)

#: The commands whose value is the 1-based index of a declared row.
_ROW_INDEX_CMDS = frozenset({"TUPLE", "GEOLOCATION"})


def _instead(var: Mapping[str, Any]) -> str:
    """What to do instead of excluding the first or the last values of `var`."""
    cmd = str(var["cmd"]).upper()
    if cmd == "TUPLE":
        return "Leave those rows out of the variable's tuples instead."
    if cmd == "GEOLOCATION":
        return "Leave those rows out of the variable's coords instead."
    if cmd == "ENUMERATION":
        return "Leave those values out of the variable's values instead."
    return (
        "Change the variable's min or max instead of excluding the start or the end of its values."
    )


def _instead_near(var: Mapping[str, Any]) -> str:
    """What to do instead of a range that excludes a value within 1e-5 of an edge."""
    if str(var["cmd"]).upper() == "ENUMERATION":
        return (
            "The engine cannot tell values within 1e-5 of each other apart: leave one of them "
            "out of the variable's values instead."
        )
    return "Declare the variable with a step larger than 1e-5, or leave this range out."


def _values_of(var: Mapping[str, Any]):
    """The values of the variable that a codec statement declares, or None for a variable type
    that the check of an EXCLUSIVE range does not know."""
    cmd = str(var["cmd"]).upper()
    try:
        if cmd in _INTERVAL_CMDS:
            return _Values(low=float(var["min"]), top=float(var["max"]))
        if cmd in _INTEGER_CMDS:
            return _Values(low=float(var["min"]), top=float(var["max"]), step=1.0)
        if cmd in STEPPED_CMDS:
            return _Values(low=float(var["min"]), top=float(var["max"]), step=float(var["step"]))
        if cmd == "ENUMERATION":
            return _Values(points=sorted(float(v) for v in var["values"]))
        if cmd in _ROW_INDEX_CMDS:
            # The engine excludes the 1-based index of the selected row.
            rows = var["tuples"] if cmd == "TUPLE" else var["coords"]
            return _Values(low=1.0, top=float(len(rows)), step=1.0)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return None


def _halfway(low: float, high: float) -> float:
    """The number halfway from `low` to `high`. With one infinite edge it is that edge: every
    value is nearer to the other one. (The difference of two edges such as -1e308 and 1e308 is
    beyond a float.)"""
    middle = low + (high - low) / 2.0
    return middle if math.isfinite(middle) else low / 2.0 + high / 2.0


def fit_exclusive_range(
    stmt: Mapping[str, Any], declared: Mapping[str, Mapping[str, Any]], *, engine_built: bool
):
    """The EXCLUSIVE statement to build: one whose edges are values of its variable.

    The engine excludes the open range from `min` to `max`, and it moves a value that falls
    inside the range to the nearer edge (COptQuestVariable.mapToExclusiveRanges), after it has
    moved the value onto the variable's bounds and grid. So the engine proposes the edge itself.
    An edge that the variable cannot take (5.5 for an INTEGER, 2.5 for an ENUMERATION of 1, 2
    and 3, a number outside the bounds) is a value that Optuna discards, and with ENUMERATION
    the study then stops with a ValueError.

    `declared` maps the lowercase name of each variable declared so far to its codec statement.
    With statements= (engine_built=False), an edge that the variable cannot take moves outward
    to the nearest value that it can take: from 5.5 and 15.5 to 5 and 16 for an INTEGER. The
    range then excludes the same values. With model= (engine_built=True), the caller built the
    model, so the sampler raises ValueError and gives that range. An edge that no value can
    reach stays as it is. An edge that a value can reach, that is not a value, and that has no
    value beyond it raises ValueError in both cases: a range from -0.5 to 3.5 on an INTEGER
    from 0, where the engine would move 0 to -0.5. (From -5 to 3, the engine moves 0 to 3, and
    the range stays as it is.)

    For an ENUMERATION, the engine moves a value onto an edge that is not a value for some
    ranges only: from 8.54 to 11.4 on -5.7, 8.2 and 9.9 it proposes only the declared values,
    but from 1.5 to 2.5 on 1, 2 and 3 it proposes 2.5, and from 1.5 to 3.5 on 1, 2, 3 and 4 it
    proposes 1.5, and also 2, inside the range. The rule here applies to every range of an
    ENUMERATION, so it also moves (statements=) or refuses (model=) a range that the engine
    would have kept to the declared values."""
    name = str(stmt["name"])
    var = declared.get(name.lower())
    if var is None and "@" in name:
        base = declared.get(name.split("@", 1)[0].lower())
        if base is not None and str(base["cmd"]).upper() in _FIXED_ELEMENT_CMDS:
            # The engine takes such a range and refuses the model when it initializes it
            # ("Variable bounds cannot be changed"), with an error that names no statement.
            raise ValueError(
                f"EXCLUSIVE {name!r}: the engine refuses an exclusive range on an element of a "
                f"{str(base['cmd']).upper()} variable ({base['name']!r})."
            )
        if base is not None and str(base["cmd"]).upper() not in _SCALAR_CMDS:
            var = base  # an element of a vector, matrix or ND variable
    values = None if var is None else _values_of(var)
    if values is None:
        return stmt
    try:
        low, high = float(stmt["min"]), float(stmt["max"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return stmt
    if not low < high:
        return stmt
    where = f"EXCLUSIVE {name!r} from {low!r} to {high!r} ({var['cmd']} {var['name']!r})"
    if math.isinf(low) and math.isinf(high):
        # A built model can hold such a range; the engine then proposes inf.
        raise ValueError(f"{where}: the range excludes every value of the variable.")

    def reached(edge_low, edge_high):
        """The edges that the engine can move a value to and that are not values."""
        middle = _halfway(edge_low, edge_high)
        edges = []
        low_is_value = values.at_edge(edge_low, lower=True)
        high_is_value = values.at_edge(edge_high, lower=False)
        if not low_is_value and values.any_between(edge_low, middle, from_a=False):
            edges.append(edge_low)
        if not high_is_value and values.any_between(middle, edge_high, from_a=True):
            edges.append(edge_high)
        return edges

    fit_low = low if values.at_edge(low, lower=True) else values.below(low)
    fit_high = high if values.at_edge(high, lower=False) else values.above(high)
    fit_low = low if fit_low is None else fit_low
    fit_high = high if fit_high is None else fit_high
    stuck = reached(fit_low, fit_high)
    if stuck:
        raise ValueError(
            f"{where}: the engine moves a value inside the range to the nearer edge, and "
            f"{stuck[0]!r} is not a value of the variable, with no value beyond it. "
            f"{_instead(var)}"
        )

    def check_near_edges(edge_low, edge_high):
        """The engine takes a value within 1e-5 (its epsilon) of an edge for the edge and keeps
        it, also when the value is inside the range (Solution.mapToFeasibleWithoutMIP and the
        enumeration of small domains): on a grid with a step of 1e-5 or less, it proposes the
        neighbour of an edge, which the range excludes."""
        near = values.inside_near_an_edge(edge_low, edge_high, _ENGINE_EPSILON)
        if near is not None:
            raise ValueError(
                f"{where}: the engine takes a value within 1e-5 of an edge for that edge, so it "
                f"would propose {near!r}, which the range excludes. {_instead_near(var)}"
            )

    if (fit_low, fit_high) == (low, high):
        check_near_edges(low, high)
        return stmt
    if engine_built:
        if not reached(low, high):
            check_near_edges(low, high)
            return stmt  # no value is moved to the edge that the variable cannot take
        raise ValueError(
            f"{where}: the engine moves a value inside the range to the nearer edge, and "
            f"{reached(low, high)[0]!r} is not a value of the variable, so Optuna would discard "
            f"it. Declare the range from {fit_low!r} to {fit_high!r}, which excludes the same "
            f"values."
        )
    check_near_edges(fit_low, fit_high)
    return {**stmt, "min": fit_low, "max": fit_high}


def _edge_tol(x: float) -> float:
    """The rounding error of an edge of an EXCLUSIVE range: two edges closer than this are one
    edge (an edge moved onto the grid value 0.6000000000000001 and an edge of 0.6). An infinite
    edge has none."""
    return 1e-9 * max(1.0, abs(x)) if math.isfinite(x) else 0.0


def _edge_tol_of(name: str, declared: Mapping[str, Mapping[str, Any]]):
    """_edge_tol for the edges of the ranges of the variable `name`, held to the rounding error
    of its values (see _Values.edge_tol) for a variable that fit_exclusive_range knows."""
    var = declared.get(name.lower())
    if var is None and "@" in name:
        base = declared.get(name.split("@", 1)[0].lower())
        if base is not None and str(base["cmd"]).upper() not in _SCALAR_CMDS:
            var = base
    values = None if var is None else _values_of(var)
    return _edge_tol if values is None else values.edge_tol


def merge_exclusive_ranges(
    stmts: Sequence[Mapping[str, Any]],
    declared: Mapping[str, Mapping[str, Any]],
    *,
    engine_built: bool,
) -> list:
    """The EXCLUSIVE statements of one variable, after fit_exclusive_range, to build.

    The engine moves a value inside one range to its nearer edge, and it does not look at the
    other ranges again: when two ranges overlap, the edge of one can be inside the other, and
    the engine then proposes a value that the other range excludes. Ranges that the user gave
    apart can overlap after fit_exclusive_range has moved their edges outward ((3.5, 6.5) and
    (6.5, 9.5) on an INTEGER become (3, 7) and (6, 10)).

    With statements= (engine_built=False), overlapping ranges are merged into one range, which
    excludes the same values, and that range is checked as fit_exclusive_range checks a range:
    (-5, 3) and (2, 8) on an INTEGER from 0 are each right, but as one range from -5 to 8 the
    engine would move 0 onto -5, so the sampler refuses them, as it refuses that one range. With
    model= (engine_built=True), the caller built the ranges, so the sampler raises ValueError,
    names them and gives the one range to declare instead. `declared` is the one of
    fit_exclusive_range.

    Ranges that only touch ((3, 7) and (7, 10)) are kept: the shared edge is not inside either
    range, so it is a value that both keep. Two edges within a rounding error of each other
    touch ((0.1, 0.6000000000000001) and (0.6, 1.0) on a grid with a step of 0.1), and the
    second range then starts at the first range's edge, so that the shared value is inside
    neither range."""
    ordered = sorted((dict(s) for s in stmts), key=lambda s: (float(s["min"]), float(s["max"])))
    edge_tol = _edge_tol_of(str(ordered[0]["name"]), declared) if ordered else _edge_tol
    groups: list = []
    for stmt in ordered:
        if groups:
            top = max(float(s["max"]) for s in groups[-1])
            if float(stmt["min"]) < top - edge_tol(top):
                groups[-1].append(stmt)
                continue
        groups.append([stmt])
    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        name = group[0]["name"]
        low = min(float(s["min"]) for s in group)
        high = max(float(s["max"]) for s in group)
        given = " and ".join(f"from {s['min']!r} to {s['max']!r}" for s in group)
        span = {**group[0], "min": low, "max": high}
        if engine_built:
            try:
                fitted = fit_exclusive_range(span, declared, engine_built=False)
                instead = (
                    f"Declare one range from {fitted['min']!r} to {fitted['max']!r}, which "
                    f"excludes the same values."
                )
            except ValueError as exc:
                instead = (
                    f"As one range from {low!r} to {high!r}, they cannot be declared either: "
                    f"{str(exc).split(': ', 1)[-1]}"
                )
            raise ValueError(
                f"EXCLUSIVE {name!r}: the ranges {given} overlap, and the engine can move a "
                f"value of one range into the other. {instead}"
            )
        try:
            merged.append(fit_exclusive_range(span, declared, engine_built=False))
        except ValueError as exc:
            raise ValueError(
                f"EXCLUSIVE {name!r}: the ranges {given} overlap, so the sampler builds them as "
                f"one range from {low!r} to {high!r}; {str(exc).split(': ', 1)[-1]}"
            ) from exc
    for before, after in zip(merged, merged[1:]):
        if float(after["min"]) < float(before["max"]):
            after["min"] = before["max"]  # they touch: see above
    return merged


def _mixture_components(stmt) -> list[str]:
    """The component names of a MIXTURE statement. The statement grammar takes a count n in
    place of the names, and the engine then names the components 1 to n (OptQuest.java reads a
    MIXTURE line with a count that way), so the variables are "<name>@1" to "<name>@<n>"."""
    components = stmt["components"]
    if isinstance(components, (list, tuple)):
        return [str(component) for component in components]
    try:
        count = _as_int(components)
    except (TypeError, ValueError):
        count = None
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(
            f"MIXTURE {stmt['name']!r}: components must be a list of names or a count of at "
            f"least 1, got {components!r}."
        )
    return [str(i) for i in range(1, count + 1)]


def _mixture(stmt):
    name = stmt["name"]
    components = _mixture_components(stmt)
    n = len(components)
    sum_to = float(stmt.get("sum_to", 1.0))
    lows = _broadcast(stmt.get("min"), n, 0.0)
    highs = _broadcast(stmt.get("max"), n, sum_to)
    bounds = dict(zip(components, zip(lows, highs)))
    return _name_keyed(name, components, lambda item: _float(*bounds[item]))


def _sequence(stmt):
    # In a full ranking (top_k omitted, or top_k of n or more) the ranks are 1 to n. In a top-k
    # ranking the first top_k components get the ranks 1 to top_k, and every other component
    # gets top_k + 1. The engine uses a top_k less than 1 as 1.
    items = list(stmt["components"])
    n = len(items)
    top_k = stmt.get("top_k")
    highest = n if top_k is None or int(top_k) >= n else max(1, int(top_k)) + 1
    return _name_keyed(stmt["name"], items, lambda item: _int(1, highest))


def _multi_stage_sequence(stmt):
    name = stmt["name"]
    stages = [list(group) for group in stmt["stages"]]
    bound_by_item = {}
    for group in stages:
        for item in group:
            bound_by_item[item] = len(group)  # rank range is per-stage-sized
    all_items = list(bound_by_item.keys())
    return _name_keyed(name, all_items, lambda item: _int(1, bound_by_item[item]))


def _policy_sequence(stmt):
    # The value of a POLICYSEQUENCE variable is a linear scoring rule over the features of a set of
    # items. The item scores (name@<item>) are inputs. The coefficient, contribution and
    # confidence families are outputs that the engine computes, so this codec registers only
    # the item scores as Optuna parameters. The bound of each score is sized from the
    # magnitude of the feature matrix.
    items = list(stmt["items"])
    bound = _policy_sequence_score_bound(stmt["matrix"])
    return _name_keyed(stmt["name"], items, lambda item: _float(-bound, bound))


def _policy_sequence_score_bound(matrix) -> float:
    """The largest absolute item score the engine can report for this feature matrix.

    The engine scales each feature column, and an item score is the scaled feature row of that
    item applied to a unit-length weight vector. So no score is larger than the length of the
    longest scaled row. The scaling below follows the engine's rules: a constant column is 0, a
    two-level column maps to -1 and +1, an integer column with at most 10 levels is divided by
    its standard deviation, and any other column is divided by the standard deviation of its
    values after they are limited to 3 interquartile ranges outside the quartiles. A margin of
    1% covers rounding."""
    rows = [list(map(float, r)) for r in matrix]
    n = len(rows)
    k = len(rows[0]) if n else 0
    eps = 1.0e-12

    def std(values):
        # Scaled by the largest magnitude: the square of a feature value of 1e155 or more
        # overflows, and the engine takes such a value.
        top = max(abs(v) for v in values)
        if top == 0.0:
            return 0.0
        unit = [v / top for v in values]
        mean = sum(unit) / len(unit)
        return top * math.sqrt(sum((u - mean) ** 2 for u in unit) / len(unit))

    def percentile(ordered, q):
        pos = q * (len(ordered) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (1.0 - (pos - lo)) + ordered[hi] * (pos - lo)

    scaled = [[0.0] * k for _ in range(n)]
    for c in range(k):
        v = [rows[i][c] for i in range(n)]
        ordered = sorted(v)
        lo, hi = ordered[0], ordered[-1]
        if hi - lo < eps:
            continue
        distinct = 1 + sum(1 for a, b in zip(ordered, ordered[1:]) if b - a > eps)
        integer = all(abs(x - round(x)) <= 1.0e-9 for x in v)
        if distinct == 2:
            centre, scale = hi / 2.0 + lo / 2.0, hi / 2.0 - lo / 2.0
            for i in range(n):
                scaled[i][c] = (v[i] - centre) / scale
            continue
        if integer and distinct <= 10:
            base = v
        else:
            q1, q3 = percentile(ordered, 0.25), percentile(ordered, 0.75)
            iqr = q3 - q1
            if iqr > eps:
                fence_lo, fence_hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
                base = [min(max(x, fence_lo), fence_hi) for x in v]
            else:
                base = v
        sd = std(base)
        scale = hi / 2.0 - lo / 2.0 if sd < eps else sd
        for i in range(n):
            scaled[i][c] = v[i] / scale
    longest = max((math.sqrt(sum(x * x for x in row)) for row in scaled), default=0.0)
    return 1.01 * longest if longest > eps else 1.0


def _choice(stmt):
    return _fixed_subkeys(stmt["name"], {"choice": _int(1, int(stmt["k"]))})


def _multi_choice(stmt):
    # This codec registers only "<name>@<choice>" for each declared choice. Each holds 1 if the
    # choice is selected and 0 if it is not.
    choices = list(stmt["choices"])
    return _name_keyed(stmt["name"], choices, lambda item: _int(0, 1))


def _assignment(stmt):
    name = stmt["name"]
    m, n = int(stmt["m"]), int(stmt["n"])
    # Each value is the 1-based index of the bin that holds the item, or 0 when the item is in
    # no bin. The declared range includes 0.
    return _index_keyed(name, m, lambda i: _int(0, n))


def _candidate_flags(stmt):
    """ITEMSELECTION, BUDGETEDSELECTION: 1 for each of the n candidates that is selected."""
    return _index_keyed(stmt["name"], int(stmt["n"]), lambda i: _int(0, 1))


def _partition(stmt):
    m, k = int(stmt["m"]), int(stmt["k"])
    return _index_keyed(stmt["name"], m, lambda i: _int(1, k))


def _adjacency(stmt):
    """DAG, TREE: 1 for each ordered pair of nodes that is an edge."""
    return _all_pairs(stmt["name"], list(stmt["nodes"]), lambda: _int(0, 1))


def _topological_sequence(stmt):
    tasks = list(stmt["tasks"])
    n = len(tasks)
    return _name_keyed(stmt["name"], tasks, lambda task: _int(1, n))


def _path(stmt):
    # Each value is the 1-based position of the node on the path, from 1 to the length of the
    # path, or 0 if the node is not on the path.
    nodes = list(stmt["nodes"])
    n = len(nodes)
    return _name_keyed(stmt["name"], nodes, lambda node: _int(0, n))


def _region(stmt):
    vertices = [list(v) for v in stmt["vertices"]]
    x_lo, x_hi = _bbox(vertices, 0)
    y_lo, y_hi = _bbox(vertices, 1)
    # Each Distribution covers the range of the vertex coordinates on its axis. In every
    # solution that the engine generates, the point is inside the polygon or on its boundary, so
    # the Distribution only needs a valid range for trial.suggest_*() and not the exact region.
    return _fixed_subkeys(stmt["name"], {"x": _float(x_lo, x_hi), "y": _float(y_lo, y_hi)})


def _spherical_region(stmt):
    # The engine reports a point inside the region or on its boundary: @theta is the latitude
    # and @phi is the longitude, folded into [-180, 180]. The distributions cover the whole
    # sphere. A tighter range would have to follow the engine's own geometry exactly (great-
    # circle edges that rise above their end vertices, regions that hold a pole, the side of
    # each edge that is inside), and a range that is a little too tight makes Optuna reject the
    # engine's point and sample a random one. The engine keeps every point in the region.
    return _fixed_subkeys(
        stmt["name"], {"theta": _float(-90.0, 90.0), "phi": _float(-180.0, 180.0)}
    )


def _float_matrix(stmt):
    """LOWRANK and the spline and Fourier matrices: rows x cols numbers from min to max."""
    return _grid(
        stmt["name"],
        (int(stmt["rows"]), int(stmt["cols"])),
        lambda: _float(stmt["min"], stmt["max"]),
    )


def _float_matrix_of_initial(stmt):
    """The correction matrices: a number from min to max for each element of `initial`."""
    initial = stmt["initial"]
    dims = (len(initial), len(initial[0]))
    return _grid(stmt["name"], dims, lambda: _float(stmt["min"], stmt["max"]))


def _timeseries_vector(stmt):
    # The names of a TIMESERIESVECTOR variable are name@1 to name@n, where n is long_scale x
    # short_scale, row by row. They are not the name@row,col names of a matrix or low-rank
    # variable.
    length = int(stmt["long_scale"]) * int(stmt["short_scale"])
    return _grid(stmt["name"], (length,), lambda: _float(stmt["min"], stmt["max"]))


def _float_vector_of_initial(stmt):
    """The correction vectors: a number from min to max for each element of `initial`."""
    initial = stmt["initial"]
    return _grid(stmt["name"], (len(initial),), lambda: _float(stmt["min"], stmt["max"]))


def _float_vector(stmt):
    """The spline and Fourier vectors: `length` numbers from min to max."""
    return _grid(stmt["name"], (int(stmt["length"]),), lambda: _float(stmt["min"], stmt["max"]))


# ---------------------------------------------------------------------------------------------
# typed-API model builders (declarative construction mode: statement dict -> live OptQuestModel)
# ---------------------------------------------------------------------------------------------
# Each command is built by a call of a typed OptQuestModel method (add_*, set_*). The table at
# the end of this section, MODEL_BUILDERS, gives the method of each command and the fields of
# the statement in the order of the method's arguments, which is not always the order or the
# names of the statement (the "size" of a VECTOR is the argument `n` of add_vector_variable). A
# command whose call needs more (optional fields, keyword arguments, a value computed from two
# fields) has a function of its own. The typed API is used, and not the OptQuestCore client of
# the statement text, because the sampler works with the solution objects of the typed API.


def last_error(source) -> str:
    """The last error of an OptQuestModel or of an OptQuestSolution, as text. When the engine
    repeats a character outside ASCII in its message (of a constraint or an expression that it
    refuses), it writes one byte for it, which is not UTF-8, and the binding raises
    UnicodeDecodeError in place of the message. The message then comes with the replacement
    character for such a byte."""
    try:
        return source.get_last_error()
    except UnicodeDecodeError as exc:
        return bytes(exc.object).decode("utf-8", errors="replace")


def _check(ok: bool, model, stmt: Mapping[str, Any]) -> None:
    if not ok:
        raise RuntimeError(f"OptQuest rejected statement {dict(stmt)!r}: {last_error(model)}")


def _rows(value) -> list:
    """A table as a list of lists."""
    return [list(row) for row in value]


def _typed_call(method_name: str, *fields):
    """The builder of a command that is one call of the typed method `method_name` of the
    model. Each field is the key of the statement whose value the method takes in that place,
    or (convert, key) when the method takes the value in another type: (int, "size"),
    (list, "items"), (_rows, "tuples")."""

    def handler(model, stmt):
        method = getattr(model, method_name)
        args = [stmt[f] if isinstance(f, str) else f[0](stmt[f[1]]) for f in fields]
        _check(method(*args), model, stmt)

    return handler


def _build_nd(model, stmt):
    dims = [int(d) for d in stmt["dims"]]
    _check(model.add_nd_variable(stmt["name"], dims, stmt["min"], stmt["max"]), model, stmt)


def _build_integer_nd(model, stmt):
    dims = [int(d) for d in stmt["dims"]]
    _check(
        model.add_integer_nd_variable(stmt["name"], dims, stmt["min"], stmt["max"]), model, stmt
    )


def _build_discrete_nd(model, stmt):
    dims = [int(d) for d in stmt["dims"]]
    _check(
        model.add_discrete_nd_variable(stmt["name"], dims, stmt["min"], stmt["max"], stmt["step"]),
        model,
        stmt,
    )


def _build_binary_nd(model, stmt):
    dims = [int(d) for d in stmt["dims"]]
    _check(model.add_binary_nd_variable(stmt["name"], dims), model, stmt)


def _build_mixture(model, stmt):
    components = _mixture_components(stmt)
    n = len(components)
    sum_to = float(stmt.get("sum_to", 1.0))
    if stmt.get("min") is None and stmt.get("max") is None:
        lb, ub = None, None
    else:
        lb = _broadcast(stmt.get("min"), n, 0.0)
        ub = _broadcast(stmt.get("max"), n, sum_to)
    _check(model.add_mixture_variable(stmt["name"], components, lb, ub, sum_to), model, stmt)


def _build_sequence(model, stmt):
    _check(
        model.add_sequence_variable(stmt["name"], list(stmt["components"]), stmt.get("top_k")),
        model,
        stmt,
    )


def _build_multi_stage_sequence(model, stmt):
    stages = [list(g) for g in stmt["stages"]]
    _check(model.add_multi_stage_sequence_variable(stmt["name"], stages), model, stmt)


def _build_policy_sequence(model, stmt):
    # Statement/wire field order is items, features, matrix; the typed API takes the
    # feature matrix first.
    matrix = [list(row) for row in stmt["matrix"]]
    _check(
        model.add_policy_sequence_variable(
            stmt["name"], matrix, list(stmt["items"]), list(stmt["features"])
        ),
        model,
        stmt,
    )


def _build_assignment(model, stmt):
    # emit_model() takes one value or a list for each of must_assign (a list holds one value
    # for each item) and bin_cap_min and bin_cap_max (a list holds one value for each bin), and
    # the engine applies one value to every item or every bin. The typed method takes all three
    # as single values (OQAddAssignmentVariableHom) or all three as lists
    # (OQAddAssignmentVariableHet), so a statement with a list builds the list form, with each
    # single value repeated. ctypes fills a short list with zeros, which the engine reads as a
    # cap of 0, so the lengths are checked here, and the error names the statement.
    must_assign = stmt.get("must_assign")
    m, n = int(stmt["m"]), int(stmt["n"])
    bin_cap_min, bin_cap_max = stmt.get("bin_cap_min", 0), stmt.get("bin_cap_max")
    fields = (must_assign, bin_cap_min, bin_cap_max)
    if must_assign is not None and any(isinstance(v, (list, tuple)) for v in fields):
        if bin_cap_min is None or bin_cap_max is None:
            raise ValueError(
                f"ASSIGNMENT {stmt['name']!r}: give bin_cap_min and bin_cap_max with "
                f"must_assign. Got bin_cap_min={bin_cap_min!r}, bin_cap_max={bin_cap_max!r}."
            )

        def each(value, count):
            return list(value) if isinstance(value, (list, tuple)) else [value] * count

        must_assign = each(must_assign, m)
        bin_cap_min, bin_cap_max = each(bin_cap_min, n), each(bin_cap_max, n)
        if len(must_assign) != m or len(bin_cap_min) != n or len(bin_cap_max) != n:
            raise ValueError(
                f"ASSIGNMENT {stmt['name']!r}: must_assign takes one value, or {m} values (one "
                f"for each item), and bin_cap_min and bin_cap_max each take one value, or {n} "
                f"values (one for each bin). Got must_assign={stmt.get('must_assign')!r}, "
                f"bin_cap_min={stmt.get('bin_cap_min')!r}, "
                f"bin_cap_max={stmt.get('bin_cap_max')!r}."
            )
    _check(
        model.add_assignment_variable(
            stmt["name"],
            m,
            n,
            must_assign,
            bin_cap_min,
            bin_cap_max,
        ),
        model,
        stmt,
    )


def _build_item_selection(model, stmt):
    # k and k_min/k_max are two alternative forms, so a statement that mixes them or supplies
    # only one of k_min/k_max is refused with a message that names both forms (a bare
    # KeyError would read as "you forgot k_max").
    n = int(stmt["n"])
    has_k, has_k_min, has_k_max = "k" in stmt, "k_min" in stmt, "k_max" in stmt
    if has_k_min and has_k_max and not has_k:
        k, k_max = int(stmt["k_min"]), int(stmt["k_max"])
    elif has_k and not has_k_min and not has_k_max:
        k, k_max = int(stmt["k"]), None
    else:
        raise ValueError(
            f"ITEMSELECTION {stmt['name']!r}: statement must supply exactly one of the two legal "
            f"forms -- {{'k': <exact count>}} or {{'k_min': <min>, 'k_max': <max>}} -- got "
            f"{sorted(key for key in ('k', 'k_min', 'k_max') if key in stmt)!r}."
        )
    _check(model.add_item_selection_variable(stmt["name"], n, k, k_max), model, stmt)


def _build_budgeted_selection(model, stmt):
    _check(
        model.add_budgeted_selection_variable(
            stmt["name"],
            int(stmt["n"]),
            list(stmt["costs"]),
            stmt["budget_max"],
            stmt.get("budget_min", 0.0),
            stmt.get("k_min", 0),
            stmt.get("k_max"),
        ),
        model,
        stmt,
    )


def _build_dag(model, stmt):
    _check(
        model.add_dag_variable(
            stmt["name"],
            list(stmt["nodes"]),
            start_node=stmt.get("start"),
            end_node=stmt.get("end"),
            connected=bool(stmt.get("connected", False)),
            predecessors=stmt.get("predecessors"),
        ),
        model,
        stmt,
    )


def _build_tree(model, stmt):
    _check(
        model.add_tree_variable(
            stmt["name"],
            list(stmt["nodes"]),
            root_node=stmt.get("root"),
            predecessors=stmt.get("predecessors"),
        ),
        model,
        stmt,
    )


def _build_topological_sequence(model, stmt):
    _check(
        model.add_topological_sequence_variable(
            stmt["name"], list(stmt["tasks"]), stmt.get("predecessors")
        ),
        model,
        stmt,
    )


def _build_path(model, stmt):
    path = stmt.get("path") or {}
    _check(
        model.add_path_variable(
            stmt["name"],
            list(stmt["nodes"]),
            start_node=path.get("start"),
            end_node=path.get("end"),
            required_nodes=path.get("required"),
        ),
        model,
        stmt,
    )


# ---- non-variable scaffolding: outputs, expressions, constraints, objectives, groups, run knobs


def _build_output_nd(model, stmt):
    dims = [int(d) for d in stmt["dims"]]
    _check(model.add_output_nd_variable(stmt["name"], dims), model, stmt)


def _build_output_statistic(model, stmt):
    _check(
        model.set_output_statistic(stmt["name"], int(stmt["statistic"]), stmt.get("value")),
        model,
        stmt,
    )


def _build_objective(model, stmt):
    direction = str(stmt["direction"]).strip().lower()
    if direction == "min":
        ok = model.add_minimize_objective(stmt["name"], stmt["expression"])
    elif direction == "max":
        ok = model.add_maximize_objective(stmt["name"], stmt["expression"])
    else:
        raise ValueError(
            f"objective {stmt['name']!r}: direction must be 'min' or 'max', got {stmt['direction']!r}"
        )
    _check(ok, model, stmt)


def _build_objective_statistic(model, stmt):
    _check(
        model.set_objective_statistic(stmt["name"], int(stmt["statistic"]), stmt.get("value")),
        model,
        stmt,
    )


def _run_knob(method_name: str, *field_names: str):
    def handler(model, stmt):
        args = [stmt[f] for f in field_names]
        method = getattr(model, method_name, None)
        if method is None:
            raise NotImplementedError(
                f"OptQuestModel has no {method_name!r} method (needed for {stmt['cmd']!r}); "
                f"this run-control statement is not supported by the declarative construction "
                f"mode yet."
            )
        _check(method(*args), model, stmt)

    return handler


MODEL_BUILDERS: dict[str, Callable[[Any, Mapping[str, Any]], None]] = {
    "CONTINUOUS": _typed_call("add_continuous_variable", "name", "min", "max"),
    "INTEGER": _typed_call("add_integer_variable", "name", "min", "max"),
    "DISCRETE": _typed_call("add_discrete_variable", "name", "min", "max", "step"),
    "BINARY": _typed_call("add_binary_variable", "name"),
    "DESIGN": _typed_call("add_design_variable", "name", (int, "num")),
    "ENUMERATION": _typed_call("add_enumeration_variable", "name", (list, "values")),
    "VECTOR": _typed_call("add_vector_variable", "name", (int, "size"), "min", "max"),
    "MATRIX": _typed_call(
        "add_matrix_variable", "name", (int, "rows"), (int, "cols"), "min", "max"
    ),
    "BINARYVECTOR": _typed_call("add_binary_vector_variable", "name", (int, "k")),
    "BINARYMATRIX": _typed_call(
        "add_binary_matrix_variable", "name", (int, "rows"), (int, "cols")
    ),
    "INTEGERVECTOR": _typed_call(
        "add_integer_vector_variable", "name", (int, "size"), "min", "max"
    ),
    "INTEGERMATRIX": _typed_call(
        "add_integer_matrix_variable", "name", (int, "rows"), (int, "cols"), "min", "max"
    ),
    "DISCRETEVECTOR": _typed_call(
        "add_discrete_vector_variable", "name", (int, "size"), "min", "max", "step"
    ),
    "DISCRETEMATRIX": _typed_call(
        "add_discrete_matrix_variable", "name", (int, "rows"), (int, "cols"), "min", "max", "step"
    ),
    "NDVARIABLE": _build_nd,
    "INTEGERNVARIABLE": _build_integer_nd,
    "DISCRETENDVARIABLE": _build_discrete_nd,
    "BINARYNDVARIABLE": _build_binary_nd,
    "PERMUTATION": _typed_call("add_permutation_variable", "name", (list, "items")),
    "TUPLE": _typed_call("add_tuple_variable", "name", (_rows, "tuples")),
    "GEOLOCATION": _typed_call("add_geolocation_variable", "name", (_rows, "coords")),
    "SELECTION": _typed_call("add_selection_variable", "name", (list, "vars")),
    "ANGLE": _typed_call("add_angle_variable", "name", "min_deg", "max_deg"),
    "SPHERICALANGLE": _typed_call(
        "add_spherical_angle_variable",
        "name",
        "min_theta_deg",
        "max_theta_deg",
        "min_phi_deg",
        "max_phi_deg",
    ),
    "MIXTURE": _build_mixture,
    "SEQUENCE": _build_sequence,
    "MULTISTAGESEQUENCE": _build_multi_stage_sequence,
    "POLICYSEQUENCE": _build_policy_sequence,
    "CHOICE": _typed_call("add_choice_variable", "name", (int, "k")),
    "MULTICHOICE": _typed_call(
        "add_multi_choice_variable",
        "name",
        (list, "choices"),
        (int, "min_select"),
        (int, "max_select"),
    ),
    "ASSIGNMENT": _build_assignment,
    "ITEMSELECTION": _build_item_selection,
    "BUDGETEDSELECTION": _build_budgeted_selection,
    "PARTITION": _typed_call("add_partition_variable", "name", (int, "m"), (int, "k")),
    "DAG": _build_dag,
    "TREE": _build_tree,
    "TOPOLOGICALSEQUENCE": _build_topological_sequence,
    "PATH": _build_path,
    "REGION": _typed_call("add_region_variable", "name", (_rows, "vertices")),
    "SPHERICALREGION": _typed_call("add_spherical_region_variable", "name", (_rows, "vertices")),
    "LOWRANK": _typed_call(
        "add_lowrank_matrix_variable",
        "name",
        (int, "rows"),
        (int, "cols"),
        (int, "rank"),
        "min",
        "max",
    ),
    "LOWRANKCORRECTION": _typed_call(
        "add_lowrank_correction_matrix_variable",
        "name",
        (_rows, "initial"),
        (int, "rank"),
        "min",
        "max",
    ),
    "LOWRANKINCREMENTAL": _typed_call(
        "add_incremental_lowrank_correction_matrix_variable",
        "name",
        (_rows, "initial"),
        (int, "rank"),
        "min",
        "max",
    ),
    "TIMESERIESVECTOR": _typed_call(
        "add_timeseries_vector_variable",
        "name",
        (int, "long_scale"),
        (int, "short_scale"),
        (int, "rank"),
        "min",
        "max",
    ),
    "TIMESERIESCORRECTIONVECTOR": _typed_call(
        "add_timeseries_correction_vector_variable",
        "name",
        (list, "initial"),
        (int, "long_scale"),
        (int, "short_scale"),
        (int, "rank"),
        "min",
        "max",
    ),
    "TIMESERIESINCREMENTALVECTOR": _typed_call(
        "add_incremental_timeseries_correction_vector_variable",
        "name",
        (list, "initial"),
        (int, "long_scale"),
        (int, "short_scale"),
        (int, "rank"),
        "min",
        "max",
    ),
    "BSPLINEVECTOR": _typed_call(
        "add_bspline_vector_variable", "name", (int, "length"), (int, "num_knots"), "min", "max"
    ),
    "SPLINEVECTOR": _typed_call(
        "add_spline_vector_variable", "name", (int, "length"), (int, "num_knots"), "min", "max"
    ),
    "BSPLINECORRECTIONVECTOR": _typed_call(
        "add_bspline_correction_vector_variable",
        "name",
        (list, "initial"),
        (int, "num_knots"),
        "min",
        "max",
    ),
    "SPLINECORRECTIONVECTOR": _typed_call(
        "add_spline_correction_vector_variable",
        "name",
        (list, "initial"),
        (int, "num_knots"),
        "min",
        "max",
    ),
    "BSPLINEMATRIX": _typed_call(
        "add_bspline_matrix_variable",
        "name",
        (int, "rows"),
        (int, "cols"),
        (int, "knot_rows"),
        (int, "knot_cols"),
        "min",
        "max",
    ),
    "SPLINEMATRIX": _typed_call(
        "add_spline_matrix_variable",
        "name",
        (int, "rows"),
        (int, "cols"),
        (int, "knot_rows"),
        (int, "knot_cols"),
        "min",
        "max",
    ),
    "BSPLINECORRECTIONMATRIX": _typed_call(
        "add_bspline_correction_matrix_variable",
        "name",
        (_rows, "initial"),
        (int, "knot_rows"),
        (int, "knot_cols"),
        "min",
        "max",
    ),
    "SPLINECORRECTIONMATRIX": _typed_call(
        "add_spline_correction_matrix_variable",
        "name",
        (_rows, "initial"),
        (int, "knot_rows"),
        (int, "knot_cols"),
        "min",
        "max",
    ),
    "FOURIERVECTOR": _typed_call(
        "add_fourier_vector_variable", "name", (int, "length"), (int, "num_freq"), "min", "max"
    ),
    "FOURIERMATRIX": _typed_call(
        "add_fourier_matrix_variable",
        "name",
        (int, "rows"),
        (int, "cols"),
        (int, "freq_rows"),
        (int, "freq_cols"),
        "min",
        "max",
    ),
    "FOURIERCORRECTIONVECTOR": _typed_call(
        "add_fourier_correction_vector_variable",
        "name",
        (list, "initial"),
        (int, "num_freq"),
        "min",
        "max",
    ),
    "FOURIERCORRECTIONMATRIX": _typed_call(
        "add_fourier_correction_matrix_variable",
        "name",
        (_rows, "initial"),
        (int, "freq_rows"),
        (int, "freq_cols"),
        "min",
        "max",
    ),
    # non-variable scaffolding
    "OUTPUT": _typed_call("add_output_variable", "name"),
    "OUTPUTVECTOR": _typed_call("add_output_vector_variable", "name", (int, "n")),
    "OUTPUTMATRIX": _typed_call(
        "add_output_matrix_variable", "name", (int, "rows"), (int, "cols")
    ),
    "OUTPUTNDVARIABLE": _build_output_nd,
    "OUTPUTSTATISTIC": _build_output_statistic,
    "EXPRESSION": _typed_call("add_expression_variable", "name", "expression"),
    "CONSTRAINT": _typed_call("add_constraint", "name", "expression"),
    "OBJECTIVE": _build_objective,
    "OBJECTIVEGOAL": _typed_call("set_objective_goal", "name", "min", "max"),
    "OBJECTIVECONFIDENCE": _typed_call(
        "set_objective_confidence", "name", (int, "ctype"), (int, "clevel"), "epct"
    ),
    "OBJECTIVESTATISTIC": _build_objective_statistic,
    "FIXGROUP": _typed_call("add_fix_group", "name", (list, "vars")),
    "FLOATGROUP": _typed_call("add_float_group", "name", (list, "vars")),
    "EXCHANGE": _typed_call("add_exchangeable_group", "name", (_rows, "groups")),
    "EXCLUSIVE": _typed_call("add_variable_exclusive_range", "name", "min", "max"),
    # run settings. LICENSE and RANDOMSEED are absent on purpose: OptQuestSampler.__init__
    # applies a LICENSE or RANDOMSEED statement together with its license=/seed= arguments, and
    # _ingest_statement skips both.
    "REPLICATIONS": _run_knob("set_replications", "min", "max"),
    "SERIALREPLICATIONS": _run_knob("set_serial_replications"),
    "STOPCYCLES": _run_knob("set_stop_cycles", "cycles"),
    "STOPIMPROVEMENT": _run_knob("set_stop_improvement", "threshold", "cycles"),
    "STOPTIME": _run_knob("set_stop_time", "seconds"),
    "STOPMILP": _run_knob("stop_on_successful_milp"),
    "PARALLELHINT": _run_knob("set_parallel_hint", "threads"),
    "LOGSETUP": _run_knob("log_setup", "file"),
    "LOGSOLUTIONS": _run_knob("log_solutions", "file"),
}

# Explicitly out of scope for the declarative construction mode: Adaptive Sampling-only
# configuration (metrics are documented as "ignored in optimization workflows"; SAMPLE/
# VARIOGRAM/RDSBINS configure Adaptive Sampling response-surface exploration, not the
# optimization loop this sampler drives).
ADAPTIVE_SAMPLING_ONLY_CMDS = frozenset({"METRIC", "SAMPLE", "VARIOGRAM", "RDSBINS"})


def plain_value(value):
    """`value` with every numpy array and numpy number in it as a Python list or number:
    optquest.emit_model() refuses an array for a list field, and the typed model methods take
    Python numbers. An iterator (a generator, a map) in it is a list: the statements are read
    more than once, and an iterator is empty for the second reader."""
    if isinstance(value, Mapping):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, Iterator)):
        return [plain_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(plain_value(item) for item in value)
    if isinstance(value, (numpy.ndarray, numpy.generic)):
        plain = value.tolist()
        if isinstance(plain, numpy.generic):
            # A long double, which tolist() gives back as it is.
            plain = complex(plain) if isinstance(plain, numpy.complexfloating) else float(plain)
        return plain_value(plain)
    return value


def _as_int(value):
    if isinstance(value, (bool, int)):
        return value
    number = float(value)
    return int(number) if number.is_integer() else value


def _coerce(value, field_type):
    if field_type == "int":
        return _as_int(value)
    if field_type == "float":
        return float(value)
    if field_type == "csv_int":
        return [_as_int(v) for v in value]
    if field_type in ("csv_float", "tokens_float"):
        return [float(v) for v in value]
    if field_type == "int_or_csv_int":
        return [_as_int(v) for v in value] if isinstance(value, (list, tuple)) else _as_int(value)
    if field_type == "float_or_csv_float":
        return [float(v) for v in value] if isinstance(value, (list, tuple)) else float(value)
    if field_type == "ssv_csv_float":
        return [[float(v) for v in row] for row in value]
    if field_type in ("csv_str", "tokens_str"):
        return _names(value)
    if field_type == "ssv_csv_str":
        return [_names(row) for row in value]
    if field_type == "int_or_csv_str":
        return _names(value) if isinstance(value, (list, tuple)) else _as_int(value)
    return value


def _names(value):
    """A name, or each name of a collection of names, as text. emit_model() writes a name with
    str(), and the predecessors of a task or a node and the required nodes of a path in the
    order that it iterates them, so a set, a mapping (its keys) or a view is a list here in that
    order. A bytes value is one name, as emit_model() writes it."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Iterable):
        return str(value)
    return [_names(item) for item in value]


def _names_before(value):
    """The names of the nodes or tasks before one, as text. emit_model() and the typed methods
    take None or an empty name there for no predecessor (a table read from a file with empty
    cells), so neither is a name."""
    if value is None or isinstance(value, str):
        return value or None
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Iterable):
        return _names(value)
    names = [_names(item) for item in value if item is not None]
    return [name for name in names if name != ""]


def _one_name(stmt: Mapping[str, Any], field: str, value):
    """The name of the one node that `field` takes, as text. Raises ValueError for a collection
    of names: emit_model() writes it as one name ("['a']"), and the typed method does not take
    it."""
    name = _names(value)
    if not isinstance(name, str) and name:
        raise ValueError(f"{_label(stmt)}: {field} is {value!r}; it takes the name of one node.")
    return name  # an empty collection is no node, as it was


#: The fields of the `path` mapping of a PATH statement (path_spec in the emitter spec).
_PATH_FIELDS = ("start", "end", "required")


def _coerce_names(stmt: dict) -> None:
    """The node and task names inside the option fields of DAG, TREE, TOPOLOGICALSEQUENCE and
    PATH (dag_options, tree_options, topo_spec and path_spec in the emitter spec) as text.

    Raises ValueError for what emit_model() and the engine take without a word and then leave
    out, so that the model would not be the one declared: a name in `predecessors` that is not
    a node or a task of the statement (a misspelled one, or one in another case; the engine
    ignores it, and the order that it stands for), and a field of `path` that it does not have
    (`start_node`, the name of the typed method's argument, for `start`)."""
    cmd = str(stmt.get("cmd", "")).upper()
    if cmd in ("DAG", "TREE"):
        for key in ("start", "end", "root"):
            if stmt.get(key) is not None:
                stmt[key] = _one_name(stmt, key, stmt[key])
    if cmd in ("DAG", "TREE", "TOPOLOGICALSEQUENCE") and isinstance(
        stmt.get("predecessors"), Mapping
    ):
        stmt["predecessors"] = {
            _names(node): _names_before(before) for node, before in stmt["predecessors"].items()
        }
        field = "tasks" if cmd == "TOPOLOGICALSEQUENCE" else "nodes"
        declared = list(stmt.get(field) or [])
        known = set(declared)
        for node, before in stmt["predecessors"].items():
            named = [node] + ([before] if isinstance(before, str) else list(before or []))
            unknown = [name for name in named if name not in known]
            if unknown:
                raise ValueError(
                    f"{_label(stmt)}: predecessors names {unknown[0]!r}, which is not one of "
                    f"its {field} {declared!r}. The engine would ignore the name, and the "
                    f"order that it stands for."
                )
    if cmd == "PATH" and isinstance(stmt.get("path"), Mapping):
        unknown = [key for key in stmt["path"] if key not in _PATH_FIELDS]
        if unknown:
            raise ValueError(
                f"{_label(stmt)}: path has no field {unknown[0]!r}; its fields are "
                f"{', '.join(_PATH_FIELDS)}."
            )
        path = {}
        for key, names in stmt["path"].items():
            if names is None:
                path[key] = None
            elif key == "required":
                path[key] = _names(names)
            else:
                path[key] = _one_name(stmt, f"the {key} of its path", names)
        stmt["path"] = path


def _spec_fields(spec: Mapping[str, Any]):
    """Each (key, field type) of one command of the emitter spec, over every form of the
    command. A field is written "key:type" there, with "?" or "=default" after an optional one;
    a field with no type (the name) is left out."""
    for form in [spec.get("args", ""), *spec.get("forms", [])]:
        for token in form.split():
            token = token.split("=", 1)[0].rstrip("?")
            if ":" in token:
                key, field_type = token.split(":", 1)
                yield key, field_type


def coerce_fields(stmt: Mapping[str, Any], commands: Mapping[str, Any]) -> dict:
    """`stmt` with each field as the Python type of its field in the optquest emitter spec
    (`commands` is optquest.load_spec()["commands"]), in every form of the command.
    optquest.emit_model() accepts a whole float (30.0) or a numeric string ("30") for an int
    field and a numeric string for a float field, and it writes an item, node or task name given
    as a number as text; the typed model methods that build the model take neither. Call it on a
    statement that emit_model() accepted."""
    stmt = dict(stmt)
    spec = commands.get(str(stmt.get("cmd", "")).upper()) or {}
    field_types = {}
    for key, field_type in _spec_fields(spec):
        field_types.setdefault(key, field_type)  # the type of the first form that has the field
    for key, field_type in field_types.items():
        if stmt.get(key) is not None:
            stmt[key] = _coerce(stmt[key], field_type)
            if field_type == "ssv_csv_float":
                _check_rows(stmt, key)
            if field_type in _INT_FIELD_TYPES:
                _check_c_int(stmt, key)
    _coerce_names(stmt)
    return stmt


#: The int field types of the emitter spec. The typed methods give the engine each of them as a
#: C int, which keeps the low 32 bits of a larger int with no error (2**32 + 1 becomes 1). The
#: count of a MIXTURE (int_or_csv_str) is one too: the codec builds a name for each component.
_INT_FIELD_TYPES = frozenset({"int", "csv_int", "int_or_csv_int", "int_or_csv_str"})
_C_INT_MIN, _C_INT_MAX = -(2**31), 2**31 - 1
#: The int fields that the sampler does not give a typed method as they are: the seed, which
#: the sampler maps onto the engine's seeds (OptQuestSampler._engine_seed).
_INT_FIELDS_MAPPED = frozenset({("RANDOMSEED", "seed")})


def _label(stmt: Mapping[str, Any]) -> str:
    cmd = str(stmt.get("cmd", "")).upper()
    return f"{cmd} {stmt['name']!r}" if stmt.get("name") is not None else cmd


def _check_c_int(stmt: Mapping[str, Any], key: str) -> None:
    """Raise ValueError when the int field `key` holds an int that a C int cannot."""
    if (str(stmt.get("cmd", "")).upper(), key) in _INT_FIELDS_MAPPED:
        return
    value = stmt[key]
    for v in value if isinstance(value, list) else [value]:
        if isinstance(v, int) and not isinstance(v, bool) and not _C_INT_MIN <= v <= _C_INT_MAX:
            raise ValueError(
                f"{_label(stmt)}: {key} is {v!r}; the engine takes an int from -2**31 to "
                f"2**31 - 1 there."
            )


#: The field types of the emitter spec that hold text, which emit_model() writes with str().
_TEXT_FIELD_TYPES = frozenset({"string", "expr"})


def coerce_text(stmt: Any, commands: Mapping[str, Any]) -> Any:
    """`stmt` with its name and its text fields (a file, an expression) as str, as
    emit_model() writes them. The typed model methods take only a str, and emit_model() checks
    a name (a reserved name, a name that starts with a digit) only when it is a str. Call it
    before emit_model(); a value that is not a statement dict is returned as it is, for
    emit_model() to report."""
    if not isinstance(stmt, Mapping):
        return stmt
    stmt = dict(stmt)
    spec = commands.get(str(stmt.get("cmd", "")).upper()) or {}
    keys = {"name"}
    keys.update(key for key, field_type in _spec_fields(spec) if field_type in _TEXT_FIELD_TYPES)
    for key in keys:
        value = stmt.get(key)
        if value is not None and not isinstance(value, str):
            stmt[key] = str(value)
    return stmt


#: The number of values in each row of the table of these commands: a GEOLOCATION row is a
#: pair [lat, lon] (the typed method reads two values from each row), and the engine reads the
#: first two values of a REGION or SPHERICALREGION row as a vertex and refuses a row with
#: fewer, so such a row holds at least two.
_ROW_SIZES = {"GEOLOCATION": (2, 2), "REGION": (2, None), "SPHERICALREGION": (2, None)}


def _check_rows(stmt: Mapping[str, Any], key: str) -> None:
    """Raise ValueError when the rows of the table field `key` differ in length, or hold a
    number of values that the command does not take. emit_model() writes such rows, and the
    typed method that builds the model reads as many values from each row as the first row
    holds, so it fills a shorter row with zeros and builds another model."""
    cmd = str(stmt.get("cmd", "")).upper()
    lengths = sorted({len(row) for row in stmt[key]})
    if len(lengths) > 1:
        raise ValueError(
            f"{cmd} {stmt.get('name')!r}: every row of {key} must hold the same number of "
            f"values; the rows hold {lengths} values."
        )
    least, most = _ROW_SIZES.get(cmd, (0, None))
    if lengths and (lengths[0] < least or (most is not None and lengths[0] > most)):
        what = "a pair of numbers" if least == most == 2 else f"at least {least} numbers"
        raise ValueError(
            f"{cmd} {stmt.get('name')!r}: every row of {key} must be {what}; the rows hold "
            f"{lengths[0]} values."
        )


#: The field types of the emitter spec that hold a name or names.
_NAME_FIELD_TYPES = frozenset(
    {
        "csv_str",
        "tokens_str",
        "ssv_csv_str",
        "int_or_csv_str",
        "tokens_ref",
        "ssv_csv_ref",
        "topo_spec",
        "path_spec",
    }
)


def _texts(value):
    """Each text in `value`: a name, a collection of names, or a mapping of names."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _texts(key)
            yield from _texts(item)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _texts(item)


#: The categories of the emitter spec whose statements name engine variables (the variables,
#: outputs and expressions; optquest._VARIABLE_CATEGORIES), and the commands of those
#: categories that name a group of variables, which is not a variable.
_VARIABLE_CATEGORIES = frozenset(
    {"scalar", "container", "smooth", "manifold", "random", "output", "expression"}
)
_GROUP_CMDS = frozenset({"EXCHANGE", "FIXGROUP", "FLOATGROUP"})


def check_ascii_names(stmt: Mapping[str, Any], commands: Mapping[str, Any]) -> None:
    """Raise ValueError when a name of an engine variable in `stmt` (its own name, or an item,
    node, task or component name, from which the engine makes the names of its elements) has a
    character outside ASCII. The engine's reply holds the value of every variable, output and
    expression, and the engine writes each name in it as the low byte of each character, so the
    client reads NaN for that name; when that byte is a space or a line break (as for U+4E0A or
    U+4E20), it also reads NaN for the names after it. An objective, a constraint and a group
    are not in the reply, and their names are not checked. `commands` is
    optquest.load_spec()["commands"]."""
    cmd = str(stmt.get("cmd", "")).upper()
    spec = commands.get(cmd) or {}
    if spec.get("cat") not in _VARIABLE_CATEGORIES or cmd in _GROUP_CMDS:
        return
    keys = {"name"}
    keys.update(key for key, field_type in _spec_fields(spec) if field_type in _NAME_FIELD_TYPES)
    if cmd in ("DAG", "TREE"):
        keys.update(("start", "end", "root", "predecessors"))
    for key in sorted(keys):
        for text in _texts(stmt.get(key)):
            if not text.isascii():
                raise ValueError(
                    f"{cmd} {stmt.get('name')!r}: the name {text!r} has a character outside "
                    f"ASCII. The OptQuest engine gives back the value of such a name as NaN, so "
                    f"statements= and model= take ASCII names only. In plain mode "
                    f"(search_space=), the sampler gives each parameter an engine name of its "
                    f"own, and any parameter name works."
                )


def check_supported(cmd: str) -> None:
    """Raise NotImplementedError for a command that OptQuestSampler does not build: the
    Adaptive Sampling commands and every command with no builder (such as the random variable
    commands). statements= calls this for every statement; model= calls it, through
    check_ingestible, for the Adaptive Sampling commands and NOGENERATE only, because the caller
    already built the rest of the model."""
    cmd = cmd.upper()
    if cmd in ADAPTIVE_SAMPLING_ONLY_CMDS:
        raise NotImplementedError(
            f"{cmd} configures Adaptive Sampling (response-surface exploration), which "
            f"OptQuestSampler does not drive -- it always runs the optimization loop. Not "
            f"supported here."
        )
    if cmd not in MODEL_BUILDERS and cmd not in ("LICENSE", "RANDOMSEED"):
        raise NotImplementedError(
            f"OptQuest command {cmd!r} is not supported by OptQuestSampler. Random variables "
            f"are values that the engine draws, not decision inputs, and NOGENERATE stops the "
            f"engine from generating solutions; see the README's 'What the sampler does not "
            f"support'."
        )


def check_ingestible(cmd: str) -> None:
    """model=: raise NotImplementedError, before the sampler initializes the caller's model, for
    a command that the sampler cannot take from a built model: an Adaptive Sampling command, or
    a command (NOGENERATE) that is neither a variable that the codec registers nor a statement
    that it knows to skip. Random variables and run settings stay in the model."""
    cmd = cmd.upper()
    if cmd in ADAPTIVE_SAMPLING_ONLY_CMDS or (
        classify(cmd) == "variable" and cmd not in _HANDLERS
    ):
        check_supported(cmd)


def build_model_statement(model, stmt: Mapping[str, Any]) -> None:
    """Apply one statement to a live typed OptQuestModel (the declarative construction mode's
    model-building counterpart to build() above). Raises NotImplementedError for the Adaptive
    Sampling commands and for anything with no registered builder. OptQuestSampler applies
    LICENSE and RANDOMSEED statements itself and does not pass them here."""
    cmd = stmt["cmd"].upper()
    if cmd in ("LICENSE", "RANDOMSEED"):
        raise ValueError(f"{cmd} statements are applied by OptQuestSampler.__init__, not here.")
    check_supported(cmd)
    MODEL_BUILDERS[cmd](model, stmt)


# ---------------------------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------------------------

_HANDLERS: dict[str, Callable[[Mapping[str, Any]], VariableCodec]] = {
    "CONTINUOUS": _continuous,
    "INTEGER": _integer,
    "DISCRETE": _discrete,
    "BINARY": _binary,
    "DESIGN": _design,
    "ENUMERATION": _enumeration,
    "VECTOR": _vector_like("size"),
    "MATRIX": _matrix_like("rows", "cols"),
    "BINARYVECTOR": _binary_vector,
    "BINARYMATRIX": _binary_matrix,
    "INTEGERVECTOR": _integer_vector,
    "INTEGERMATRIX": _integer_matrix,
    "DISCRETEVECTOR": _vector_like("size"),
    "DISCRETEMATRIX": _matrix_like("rows", "cols"),
    "NDVARIABLE": _nd("float"),
    "INTEGERNVARIABLE": _nd("int"),
    "DISCRETENDVARIABLE": _nd("discrete"),
    "BINARYNDVARIABLE": _nd("binary"),
    "PERMUTATION": _permutation,
    "TUPLE": _tuple_var,
    "GEOLOCATION": _geolocation,
    "SELECTION": _selection,
    "ANGLE": _angle,
    "SPHERICALANGLE": _spherical_angle,
    "MIXTURE": _mixture,
    "SEQUENCE": _sequence,
    "MULTISTAGESEQUENCE": _multi_stage_sequence,
    "POLICYSEQUENCE": _policy_sequence,
    "CHOICE": _choice,
    "MULTICHOICE": _multi_choice,
    "ASSIGNMENT": _assignment,
    "ITEMSELECTION": _candidate_flags,
    "BUDGETEDSELECTION": _candidate_flags,
    "PARTITION": _partition,
    "DAG": _adjacency,
    "TREE": _adjacency,
    "TOPOLOGICALSEQUENCE": _topological_sequence,
    "PATH": _path,
    "REGION": _region,
    "SPHERICALREGION": _spherical_region,
    "LOWRANK": _float_matrix,
    "LOWRANKCORRECTION": _float_matrix_of_initial,
    "LOWRANKINCREMENTAL": _float_matrix_of_initial,
    "TIMESERIESVECTOR": _timeseries_vector,
    "TIMESERIESCORRECTIONVECTOR": _float_vector_of_initial,
    "TIMESERIESINCREMENTALVECTOR": _float_vector_of_initial,
    "BSPLINEVECTOR": _float_vector,
    "SPLINEVECTOR": _float_vector,
    "BSPLINECORRECTIONVECTOR": _float_vector_of_initial,
    "SPLINECORRECTIONVECTOR": _float_vector_of_initial,
    "BSPLINEMATRIX": _float_matrix,
    "SPLINEMATRIX": _float_matrix,
    "BSPLINECORRECTIONMATRIX": _float_matrix_of_initial,
    "SPLINECORRECTIONMATRIX": _float_matrix_of_initial,
    "FOURIERVECTOR": _float_vector,
    "FOURIERMATRIX": _float_matrix,
    "FOURIERCORRECTIONVECTOR": _float_vector_of_initial,
    "FOURIERCORRECTIONMATRIX": _float_matrix_of_initial,
}

# Commands that modify an already-declared variable/output (rather than declaring a new one) or
# are otherwise not part of the Optuna-facing decision search space. Callers recognize and skip
# them: they add no Optuna params, and they are not errors.
SKIP_CMDS = frozenset(
    {
        "EXCLUSIVE",  # narrows an existing variable's domain by excluding a sub-range; Optuna
        # has no Distribution that can represent a "hole". The engine keeps its values out of
        # the range (see fit_exclusive_range), but the Optuna-facing distribution does not show
        # it, so a value that Optuna samples itself (sample_independent) can be in the range.
        "FIXGROUP",
        "FLOATGROUP",
        "EXCHANGE",
        "OUTPUTSTATISTIC",
    }
)

# Random variables are noise that the engine redraws, and they are not decision inputs, so
# the search space does not include them. Their realizations do not appear in trial.params.
RANDOM_CMDS = frozenset(
    {
        "BETARANDOM",
        "CAUCHYRANDOM",
        "CHISQRANDOM",
        "EXPONENTIALRANDOM",
        "GAMMARANDOM",
        "LOGNORMALRANDOM",
        "NORMALRANDOM",
        "PARETORANDOM",
        "POISSONRANDOM",
        "POWERRANDOM",
        "SPHERICALRANDOM",
        "STUDENTTRANDOM",
        "UNIFORMRANDOM",
        "WEIBULLRANDOM",
    }
)

# Output-declaring commands: not part of the decision search space (tracked separately by the
# sampler for the outputs/objectives contract, see OptQuestSampler.wrap()).
OUTPUT_CMDS = frozenset({"OUTPUT", "OUTPUTVECTOR", "OUTPUTMATRIX", "OUTPUTNDVARIABLE"})

# Non-variable model statements the sampler tracks itself (objectives/expressions/constraints)
# or that are pure engine run-configuration with no Optuna-facing effect.
OTHER_CMDS = frozenset(
    {
        "LICENSE",
        "LOGSETUP",
        "LOGSOLUTIONS",
        "EXPRESSION",
        "CONSTRAINT",
        "OBJECTIVE",
        "OBJECTIVEGOAL",
        "OBJECTIVECONFIDENCE",
        "OBJECTIVESTATISTIC",
        "METRIC",
        "REPLICATIONS",
        "SERIALREPLICATIONS",
        "RANDOMSEED",
        "STOPMILP",
        "STOPCYCLES",
        "STOPIMPROVEMENT",
        "STOPTIME",
        "PARALLELHINT",
        "SAMPLE",
        "VARIOGRAM",
        "RDSBINS",
    }
)


def build(stmt: Mapping[str, Any]) -> VariableCodec:
    """Build the VariableCodec for one variable-declaring statement. Raises
    NotImplementedError for a command with no handler (including SKIP_CMDS/RANDOM_CMDS/
    OUTPUT_CMDS/OTHER_CMDS; callers should check those sets first and not call build() for
    them; see OptQuestSampler for the classify-then-dispatch loop)."""
    cmd = stmt["cmd"].upper()
    handler = _HANDLERS.get(cmd)
    if handler is None:
        raise NotImplementedError(
            f"No Optuna shape strategy registered for OptQuest command {cmd!r} "
            f"(statement: {dict(stmt)!r}). This type is not supported by "
            f"OptQuestSampler's declarative/typed construction modes."
        )
    try:
        return handler(stmt)
    except ValueError as exc:
        # An error from Optuna's distributions (a MIXTURE sum_to below 0, a CHOICE with k = 0,
        # bounds in the wrong order on a built model) does not say which statement it is.
        prefix = f"{cmd} {stmt.get('name')!r}"
        if str(exc).upper().startswith(prefix.upper()):
            raise
        raise ValueError(f"{prefix}: {exc}") from exc


def classify(cmd: str) -> str:
    """Return one of "variable", "output", "skip", "other" for a command token."""
    cmd = cmd.upper()
    if cmd in OUTPUT_CMDS:
        return "output"
    if cmd in SKIP_CMDS or cmd in RANDOM_CMDS:
        return "skip"
    if cmd in OTHER_CMDS:
        return "other"
    return "variable"
