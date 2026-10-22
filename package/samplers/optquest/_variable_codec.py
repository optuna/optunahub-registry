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
produced by `optquest_loop.model_doc_to_statements()` and consumed by `optquest.emit_statement()`
/`OptQuestCore.create_model_from_statements()`. Field names below match that grammar exactly
(see `_EMITTER_SPEC["commands"]` in optquest.py) so statements built either declaratively or
recovered from a `describe()` reverse-parse work identically here.
"""

from __future__ import annotations

import itertools
import math
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence

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


def _float(
    low: float, high: float, *, step: float | None = None
) -> optuna.distributions.FloatDistribution:
    return optuna.distributions.FloatDistribution(float(low), float(high), step=step)


def _int(low: int, high: int) -> optuna.distributions.IntDistribution:
    return optuna.distributions.IntDistribution(int(low), int(high))


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
        return choices[labels.index(raw[name])]

    return VariableCodec(
        {name: dist}, decode, to_engine={name: to_engine}, from_engine={name: from_engine}
    )


# ---------------------------------------------------------------------------------------------
# per-command handlers
# ---------------------------------------------------------------------------------------------


def _continuous(stmt):
    return _scalar_bare(stmt["name"], _float(stmt["min"], stmt["max"]))


def _integer(stmt):
    return _scalar_bare(stmt["name"], _int(stmt["min"], stmt["max"]))


def _discrete(stmt):
    return _scalar_bare(stmt["name"], _float(stmt["min"], stmt["max"], step=float(stmt["step"])))


def _binary(stmt):
    return _scalar_bare(stmt["name"], _int(0, 1))


def _design(stmt):
    return _scalar_index(stmt["name"], list(range(1, int(stmt["num"]) + 1)))


def _enumeration(stmt):
    # The raw engine value is one of the declared values and not an index (see
    # _scalar_index's docstring), so it must not go through the index-based shape.
    values = [float(v) for v in stmt["values"]]
    dist = optuna.distributions.CategoricalDistribution(values)
    return VariableCodec({stmt["name"]: dist}, lambda raw: raw[stmt["name"]])


def _vector_like(count_field: str):
    def handler(stmt):
        name = stmt["name"]
        n = int(stmt[count_field])
        step = float(stmt["step"]) if "step" in stmt else None
        return _grid(name, (n,), lambda: _float(stmt["min"], stmt["max"], step=step))

    return handler


def _matrix_like(rows_field: str, cols_field: str):
    def handler(stmt):
        name = stmt["name"]
        rows, cols = int(stmt[rows_field]), int(stmt[cols_field])
        step = float(stmt["step"]) if "step" in stmt else None
        return _grid(name, (rows, cols), lambda: _float(stmt["min"], stmt["max"], step=step))

    return handler


def _binary_vector(stmt):
    return _grid(stmt["name"], (int(stmt["k"]),), lambda: _int(0, 1))


def _binary_matrix(stmt):
    return _grid(stmt["name"], (int(stmt["rows"]), int(stmt["cols"])), lambda: _int(0, 1))


def _integer_vector(stmt):
    return _grid(stmt["name"], (int(stmt["size"]),), lambda: _int(stmt["min"], stmt["max"]))


def _integer_matrix(stmt):
    return _grid(
        stmt["name"],
        (int(stmt["rows"]), int(stmt["cols"])),
        lambda: _int(stmt["min"], stmt["max"]),
    )


def _nd(dist_factory_kind: str):
    def handler(stmt):
        dims = [int(d) for d in stmt["dims"]]

        def f():
            if dist_factory_kind == "float":
                return _float(stmt["min"], stmt["max"])
            if dist_factory_kind == "int":
                return _int(stmt["min"], stmt["max"])
            if dist_factory_kind == "discrete":
                return _float(stmt["min"], stmt["max"], step=float(stmt["step"]))
            return _int(0, 1)  # binary

        return _grid(stmt["name"], dims, f)

    return handler


def _permutation(stmt):
    items = list(stmt["items"])
    n = len(items)
    return _name_keyed(stmt["name"], items, lambda item: _int(1, n))


def _tuple_var(stmt):
    # Real rows (list-of-lists), and not pre-joined strings: decode() returns _scalar_index's
    # `choices` verbatim (choices[labels.index(...)]), so a join here would make decode()
    # return a comma-joined string instead of the tuple. `labels` (the Optuna-facing
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


def _azimuth_decoder(low: float, high: float) -> Callable[[float], float]:
    """Move an angle that the engine reports in [0, 360) back into its declared arc.

    The engine reports an angle as its equivalent in [0, 360), not in the range that you
    declare: an ANGLE declared from -45 to 45 reports 315 for -45. If this codec registers
    FloatDistribution(-45, 45) and gives Optuna the value 315, Optuna rejects the value of the
    engine, puts a random value in its place, and sends that value back to the engine. This
    function subtracts 360 as necessary, so the value is in the declared arc.

    When the arc is 360 degrees or more, the arc is the full circle and the value stays as it is.
    """
    if high - low >= 360.0:
        return _identity

    def decode(value):
        wrapped = low + math.fmod(float(value) - low, 360.0)
        if wrapped < low:
            wrapped += 360.0
        # Clamp for the same floating-point reason the log decoders clamp: at a bound, fmod
        # can give a value one ulp outside, and the distribution's bounds are inclusive.
        return min(high, max(low, wrapped))

    return decode


def _angle(stmt):
    name = stmt["name"]
    low, high = float(stmt["min_deg"]), float(stmt["max_deg"])
    codec = _fixed_subkeys(name, {"theta": _float(low, high)})
    # The engine reports the ANGLE value in [0, 360). (The SPHERICALANGLE theta is an
    # elevation, and the engine reports it as it is.)
    codec.from_engine[f"{name}@theta"] = _azimuth_decoder(low, high)
    return codec


def _spherical_angle(stmt):
    name = stmt["name"]
    phi_low, phi_high = float(stmt["min_phi_deg"]), float(stmt["max_phi_deg"])
    codec = _fixed_subkeys(
        name,
        {
            "theta": _float(stmt["min_theta_deg"], stmt["max_theta_deg"]),
            "phi": _float(phi_low, phi_high),
        },
    )
    # phi is the azimuth, and the engine reports it in [0, 360). theta is the elevation, and
    # the engine reports it as it is.
    codec.from_engine[f"{name}@phi"] = _azimuth_decoder(phi_low, phi_high)
    return codec


def _mixture(stmt):
    name = stmt["name"]
    components = list(stmt["components"])
    if not components or not isinstance(components[0], str):
        raise NotImplementedError(
            f"MIXTURE {name!r}: this codec requires named components (a list of strings); a "
            f"bare component count is not supported (OptQuestModel.add_mixture_variable always "
            f"requires names, so this should not occur from that path)."
        )
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
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

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
            centre, scale = (hi + lo) / 2.0, (hi - lo) / 2.0
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
        scale = (hi - lo) / 2.0 if sd < eps else sd
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


def _item_selection(stmt):
    return _index_keyed(stmt["name"], int(stmt["n"]), lambda i: _int(0, 1))


def _budgeted_selection(stmt):
    return _index_keyed(stmt["name"], int(stmt["n"]), lambda i: _int(0, 1))


def _partition(stmt):
    m, k = int(stmt["m"]), int(stmt["k"])
    return _index_keyed(stmt["name"], m, lambda i: _int(1, k))


def _dag(stmt):
    return _all_pairs(stmt["name"], list(stmt["nodes"]), lambda: _int(0, 1))


def _tree(stmt):
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
    # and @phi is the longitude, folded into [-180, 180]. The region is on the right of each edge
    # as you walk from one vertex to the next, so it can hold a pole. A region that holds a pole
    # reaches the latitude of that pole and has points at every longitude. A region that holds
    # no pole has its latitude and longitude range on its boundary. The edges are great-circle
    # arcs, which can reach a higher latitude than their end vertices, so the latitude range
    # comes from points along each edge. The longitudes are folded the same way the engine folds
    # them; a polygon whose folded longitudes span more than 180 degrees can cross the
    # antimeridian, and its longitude range is then the full [-180, 180].
    vertices = _distinct_ring([list(v) for v in stmt["vertices"]])
    theta_lo, theta_hi = _great_circle_latitude_range(vertices)
    north = theta_hi >= 90.0 or _spherical_region_holds(vertices, (0.0, 0.0, 1.0))
    south = theta_lo <= -90.0 or _spherical_region_holds(vertices, (0.0, 0.0, -1.0))
    if north:
        theta_hi = 90.0
    if south:
        theta_lo = -90.0
    folded = [((float(v[1]) + 180.0) % 360.0) - 180.0 for v in vertices]
    phi_lo, phi_hi = min(folded), max(folded)
    if north or south or phi_hi - phi_lo > 180.0:
        phi_lo, phi_hi = -180.0, 180.0
    return _fixed_subkeys(
        stmt["name"], {"theta": _float(theta_lo, theta_hi), "phi": _float(phi_lo, phi_hi)}
    )


def _distinct_ring(vertices):
    """The vertices without a vertex that repeats the one before it, and without a last vertex
    that repeats the first. The engine ignores these repeated vertices."""
    ring = []
    for v in vertices:
        if not ring or (float(v[0]), float(v[1])) != (float(ring[-1][0]), float(ring[-1][1])):
            ring.append(v)
    while len(ring) > 1 and (float(ring[-1][0]), float(ring[-1][1])) == (
        float(ring[0][0]),
        float(ring[0][1]),
    ):
        ring.pop()
    return ring


def _unit(lat, lon):
    la, lo = math.radians(float(lat)), math.radians(float(lon))
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalized(a):
    n = math.sqrt(_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


def _arcs_cross(a, b, c, d) -> bool:
    """True when the great-circle arc a-b crosses the arc c-d at a point inside both arcs."""
    ab = _cross(a, b)
    acb = -_dot(ab, c)
    bda = _dot(ab, d)
    if acb * bda <= 0.0:
        return False
    cd = _cross(c, d)
    cbd = -_dot(cd, b)
    dac = _dot(cd, a)
    return acb * cbd > 0.0 and acb * dac > 0.0


def _spherical_region_holds(vertices, point) -> bool:
    """True when the unit vector `point` is inside the region on the right of each edge of a
    ring of [latitude, longitude] vertices.

    A reference point just to the right of the middle of the first edge is inside. The point is
    inside when a path from the reference point crosses the boundary an even number of times.
    The path is two arcs through a point near the middle of the path. That middle point is moved
    a small distance off the great circle through the two ends, so that the path does not go
    through a vertex that is on that great circle, such as a vertex on the meridian of a pole."""
    ring = [_unit(v[0], v[1]) for v in vertices]
    if len(ring) < 3:
        return False
    edges = list(zip(ring, ring[1:] + ring[:1]))
    a, b = edges[0]
    normal = _cross(a, b)
    if _dot(normal, normal) < 1.0e-24:
        return False
    normal = _normalized(normal)
    middle = _normalized((a[0] + b[0], a[1] + b[1], a[2] + b[2]))
    step = 1.0e-7
    ref = _normalized(tuple(m - step * n for m, n in zip(middle, normal)))
    cosine = _dot(ref, point)
    if cosine > -0.5:
        middle = tuple(q + r for q, r in zip(point, ref))
    else:
        middle = tuple(q - cosine * r for q, r in zip(point, ref))
        if _dot(middle, middle) < 1.0e-24:
            middle = _cross(ref, (0.0, 0.0, 1.0) if abs(ref[2]) < 0.9 else (1.0, 0.0, 0.0))
    middle = _normalized(middle)
    offset = (0.0157079, -0.0135914, 0.0070710)
    middle = _normalized(tuple(m + o for m, o in zip(middle, offset)))
    path = [ref, middle, point]
    crossings = 0
    for start, end in zip(path, path[1:]):
        for c, d in edges:
            crossings += _arcs_cross(start, end, c, d)
    return crossings % 2 == 0


def _great_circle_latitude_range(vertices, samples_per_edge: int = 256) -> tuple[float, float]:
    """The smallest and largest latitude, in degrees, along the great-circle edges of a closed
    ring of [latitude, longitude] vertices, with a small margin, limited to [-90, 90]."""

    def unit(lat, lon):
        la, lo = math.radians(lat), math.radians(lon)
        return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))

    points = [unit(float(v[0]), float(v[1])) for v in vertices]
    lats = [float(v[0]) for v in vertices]
    for a, b in zip(points, points[1:] + points[:1]):
        dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
        omega = math.acos(dot)
        if omega < 1.0e-12 or math.pi - omega < 1.0e-12:
            continue
        for s in range(1, samples_per_edge):
            t = s / samples_per_edge
            wa = math.sin((1.0 - t) * omega) / math.sin(omega)
            wb = math.sin(t * omega) / math.sin(omega)
            z = wa * a[2] + wb * b[2]
            lats.append(math.degrees(math.asin(max(-1.0, min(1.0, z)))))
    margin = 1.0e-6
    return max(-90.0, min(lats) - margin), min(90.0, max(lats) + margin)


def _lowrank(stmt):
    return _grid(
        stmt["name"],
        (int(stmt["rows"]), int(stmt["cols"])),
        lambda: _float(stmt["min"], stmt["max"]),
    )


def _lowrank_correction(stmt):
    initial = stmt["initial"]
    dims = (len(initial), len(initial[0]))
    return _grid(stmt["name"], dims, lambda: _float(stmt["min"], stmt["max"]))


def _timeseries_vector(stmt):
    # The names of a TIMESERIESVECTOR variable are name@1 to name@n, where n is long_scale x
    # short_scale, row by row. They are not the name@row,col names of a matrix or low-rank
    # variable.
    length = int(stmt["long_scale"]) * int(stmt["short_scale"])
    return _grid(stmt["name"], (length,), lambda: _float(stmt["min"], stmt["max"]))


def _timeseries_correction_vector(stmt):
    initial = stmt["initial"]
    return _grid(stmt["name"], (len(initial),), lambda: _float(stmt["min"], stmt["max"]))


def _spline_vector(stmt):
    return _grid(stmt["name"], (int(stmt["length"]),), lambda: _float(stmt["min"], stmt["max"]))


def _spline_correction_vector(stmt):
    initial = stmt["initial"]
    return _grid(stmt["name"], (len(initial),), lambda: _float(stmt["min"], stmt["max"]))


def _spline_matrix(stmt):
    return _grid(
        stmt["name"],
        (int(stmt["rows"]), int(stmt["cols"])),
        lambda: _float(stmt["min"], stmt["max"]),
    )


def _spline_correction_matrix(stmt):
    initial = stmt["initial"]
    dims = (len(initial), len(initial[0]))
    return _grid(stmt["name"], dims, lambda: _float(stmt["min"], stmt["max"]))


def _fourier_vector(stmt):
    return _grid(stmt["name"], (int(stmt["length"]),), lambda: _float(stmt["min"], stmt["max"]))


def _fourier_matrix(stmt):
    return _grid(
        stmt["name"],
        (int(stmt["rows"]), int(stmt["cols"])),
        lambda: _float(stmt["min"], stmt["max"]),
    )


def _fourier_correction_vector(stmt):
    initial = stmt["initial"]
    return _grid(stmt["name"], (len(initial),), lambda: _float(stmt["min"], stmt["max"]))


def _fourier_correction_matrix(stmt):
    initial = stmt["initial"]
    dims = (len(initial), len(initial[0]))
    return _grid(stmt["name"], dims, lambda: _float(stmt["min"], stmt["max"]))


# ---------------------------------------------------------------------------------------------
# typed-API model builders (declarative construction mode: statement dict -> live OptQuestModel)
# ---------------------------------------------------------------------------------------------
# One function per command that calls the corresponding typed OptQuestModel.add_*/set_* method
# with positional args in the method's own declared order (not by matching keyword names to the
# emitter's field names, which sometimes differ: e.g. VECTOR's "size" field is add_vector_
# variable's positional "n" parameter; POLICYSEQUENCE's emitter field order "items, features, matrix"
# differs from add_policy_sequence_variable's "feature_matrix, item_names, feature_names"). Targets the
# typed OptQuestModel/OptQuestSolution API rather than the separate OptQuestCore string-protocol
# client, so sample_relative only needs one solution-object contract (OptQuestCore's Solution is
# a plain dataclass with no is_valid()/get_iteration()/submit() methods of its own).


def _check(ok: bool, model, stmt: Mapping[str, Any]) -> None:
    if not ok:
        raise RuntimeError(f"OptQuest rejected statement {dict(stmt)!r}: {model.get_last_error()}")


def _build_continuous(model, stmt):
    _check(model.add_continuous_variable(stmt["name"], stmt["min"], stmt["max"]), model, stmt)


def _build_integer(model, stmt):
    _check(model.add_integer_variable(stmt["name"], stmt["min"], stmt["max"]), model, stmt)


def _build_discrete(model, stmt):
    _check(
        model.add_discrete_variable(stmt["name"], stmt["min"], stmt["max"], stmt["step"]),
        model,
        stmt,
    )


def _build_binary(model, stmt):
    _check(model.add_binary_variable(stmt["name"]), model, stmt)


def _build_design(model, stmt):
    _check(model.add_design_variable(stmt["name"], int(stmt["num"])), model, stmt)


def _build_enumeration(model, stmt):
    _check(model.add_enumeration_variable(stmt["name"], list(stmt["values"])), model, stmt)


def _build_vector(model, stmt):
    _check(
        model.add_vector_variable(stmt["name"], int(stmt["size"]), stmt["min"], stmt["max"]),
        model,
        stmt,
    )


def _build_matrix(model, stmt):
    _check(
        model.add_matrix_variable(
            stmt["name"], int(stmt["rows"]), int(stmt["cols"]), stmt["min"], stmt["max"]
        ),
        model,
        stmt,
    )


def _build_binary_vector(model, stmt):
    _check(model.add_binary_vector_variable(stmt["name"], int(stmt["k"])), model, stmt)


def _build_binary_matrix(model, stmt):
    _check(
        model.add_binary_matrix_variable(stmt["name"], int(stmt["rows"]), int(stmt["cols"])),
        model,
        stmt,
    )


def _build_integer_vector(model, stmt):
    _check(
        model.add_integer_vector_variable(
            stmt["name"], int(stmt["size"]), stmt["min"], stmt["max"]
        ),
        model,
        stmt,
    )


def _build_integer_matrix(model, stmt):
    _check(
        model.add_integer_matrix_variable(
            stmt["name"], int(stmt["rows"]), int(stmt["cols"]), stmt["min"], stmt["max"]
        ),
        model,
        stmt,
    )


def _build_discrete_vector(model, stmt):
    _check(
        model.add_discrete_vector_variable(
            stmt["name"], int(stmt["size"]), stmt["min"], stmt["max"], stmt["step"]
        ),
        model,
        stmt,
    )


def _build_discrete_matrix(model, stmt):
    _check(
        model.add_discrete_matrix_variable(
            stmt["name"],
            int(stmt["rows"]),
            int(stmt["cols"]),
            stmt["min"],
            stmt["max"],
            stmt["step"],
        ),
        model,
        stmt,
    )


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


def _build_permutation(model, stmt):
    _check(model.add_permutation_variable(stmt["name"], list(stmt["items"])), model, stmt)


def _build_tuple(model, stmt):
    _check(
        model.add_tuple_variable(stmt["name"], [list(row) for row in stmt["tuples"]]), model, stmt
    )


def _build_geolocation(model, stmt):
    _check(
        model.add_geolocation_variable(stmt["name"], [list(c) for c in stmt["coords"]]),
        model,
        stmt,
    )


def _build_selection(model, stmt):
    _check(model.add_selection_variable(stmt["name"], list(stmt["vars"])), model, stmt)


def _build_angle(model, stmt):
    _check(model.add_angle_variable(stmt["name"], stmt["min_deg"], stmt["max_deg"]), model, stmt)


def _build_spherical_angle(model, stmt):
    _check(
        model.add_spherical_angle_variable(
            stmt["name"],
            stmt["min_theta_deg"],
            stmt["max_theta_deg"],
            stmt["min_phi_deg"],
            stmt["max_phi_deg"],
        ),
        model,
        stmt,
    )


def _build_mixture(model, stmt):
    components = list(stmt["components"])
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


def _build_choice(model, stmt):
    _check(model.add_choice_variable(stmt["name"], int(stmt["k"])), model, stmt)


def _build_multi_choice(model, stmt):
    _check(
        model.add_multi_choice_variable(
            stmt["name"], list(stmt["choices"]), int(stmt["min_select"]), int(stmt["max_select"])
        ),
        model,
        stmt,
    )


def _build_assignment(model, stmt):
    # Same validation as _manifold_codec._build_assignment. The heterogeneous form (a per-item
    # must_assign list) uses OQAddAssignmentVariableHet, which has no scalar/omitted-value
    # fallback for bin_cap_min/max, unlike the homogeneous and unconstrained forms. Without
    # this check, a scalar or a missing bound causes a crash deep inside ctypes marshalling
    # ("c_long_Array_2() argument after * must be an iterable, not int"), and the error does
    # not say which statement was wrong.
    must_assign = stmt.get("must_assign")
    n = int(stmt["n"])
    if isinstance(must_assign, (list, tuple)):
        bin_cap_min, bin_cap_max = stmt.get("bin_cap_min"), stmt.get("bin_cap_max")
        valid = (
            isinstance(bin_cap_min, (list, tuple))
            and len(bin_cap_min) == n
            and isinstance(bin_cap_max, (list, tuple))
            and len(bin_cap_max) == n
        )
        if not valid:
            raise ValueError(
                f"ASSIGNMENT {stmt['name']!r}: when must_assign is a per-item list, bin_cap_min "
                f"and bin_cap_max must both be supplied as length-{n} lists too -- the engine's "
                f"heterogeneous form has no scalar/omitted-value fallback. Got "
                f"bin_cap_min={bin_cap_min!r}, bin_cap_max={bin_cap_max!r}."
            )
    else:
        bin_cap_min, bin_cap_max = stmt.get("bin_cap_min", 0), stmt.get("bin_cap_max")
    _check(
        model.add_assignment_variable(
            stmt["name"],
            int(stmt["m"]),
            n,
            must_assign,
            bin_cap_min,
            bin_cap_max,
        ),
        model,
        stmt,
    )


def _build_item_selection(model, stmt):
    # Same validation as _manifold_codec._build_item_selection, which RayTune and Hydra use:
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


def _build_partition(model, stmt):
    _check(model.add_partition_variable(stmt["name"], int(stmt["m"]), int(stmt["k"])), model, stmt)


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


def _build_region(model, stmt):
    _check(
        model.add_region_variable(stmt["name"], [list(v) for v in stmt["vertices"]]), model, stmt
    )


def _build_spherical_region(model, stmt):
    _check(
        model.add_spherical_region_variable(stmt["name"], [list(v) for v in stmt["vertices"]]),
        model,
        stmt,
    )


def _build_lowrank(model, stmt):
    _check(
        model.add_lowrank_matrix_variable(
            stmt["name"],
            int(stmt["rows"]),
            int(stmt["cols"]),
            int(stmt["rank"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_lowrank_correction(model, stmt):
    _check(
        model.add_lowrank_correction_matrix_variable(
            stmt["name"],
            [list(r) for r in stmt["initial"]],
            int(stmt["rank"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_lowrank_incremental(model, stmt):
    _check(
        model.add_incremental_lowrank_correction_matrix_variable(
            stmt["name"],
            [list(r) for r in stmt["initial"]],
            int(stmt["rank"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_timeseries_vector(model, stmt):
    _check(
        model.add_timeseries_vector_variable(
            stmt["name"],
            int(stmt["long_scale"]),
            int(stmt["short_scale"]),
            int(stmt["rank"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_timeseries_correction_vector(model, stmt):
    _check(
        model.add_timeseries_correction_vector_variable(
            stmt["name"],
            list(stmt["initial"]),
            int(stmt["long_scale"]),
            int(stmt["short_scale"]),
            int(stmt["rank"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_timeseries_incremental_vector(model, stmt):
    _check(
        model.add_incremental_timeseries_correction_vector_variable(
            stmt["name"],
            list(stmt["initial"]),
            int(stmt["long_scale"]),
            int(stmt["short_scale"]),
            int(stmt["rank"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_bspline_vector(model, stmt):
    _check(
        model.add_bspline_vector_variable(
            stmt["name"], int(stmt["length"]), int(stmt["num_knots"]), stmt["min"], stmt["max"]
        ),
        model,
        stmt,
    )


def _build_spline_vector(model, stmt):
    _check(
        model.add_spline_vector_variable(
            stmt["name"], int(stmt["length"]), int(stmt["num_knots"]), stmt["min"], stmt["max"]
        ),
        model,
        stmt,
    )


def _build_bspline_correction_vector(model, stmt):
    _check(
        model.add_bspline_correction_vector_variable(
            stmt["name"], list(stmt["initial"]), int(stmt["num_knots"]), stmt["min"], stmt["max"]
        ),
        model,
        stmt,
    )


def _build_spline_correction_vector(model, stmt):
    _check(
        model.add_spline_correction_vector_variable(
            stmt["name"], list(stmt["initial"]), int(stmt["num_knots"]), stmt["min"], stmt["max"]
        ),
        model,
        stmt,
    )


def _build_bspline_matrix(model, stmt):
    _check(
        model.add_bspline_matrix_variable(
            stmt["name"],
            int(stmt["rows"]),
            int(stmt["cols"]),
            int(stmt["knot_rows"]),
            int(stmt["knot_cols"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_spline_matrix(model, stmt):
    _check(
        model.add_spline_matrix_variable(
            stmt["name"],
            int(stmt["rows"]),
            int(stmt["cols"]),
            int(stmt["knot_rows"]),
            int(stmt["knot_cols"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_bspline_correction_matrix(model, stmt):
    _check(
        model.add_bspline_correction_matrix_variable(
            stmt["name"],
            [list(r) for r in stmt["initial"]],
            int(stmt["knot_rows"]),
            int(stmt["knot_cols"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_spline_correction_matrix(model, stmt):
    _check(
        model.add_spline_correction_matrix_variable(
            stmt["name"],
            [list(r) for r in stmt["initial"]],
            int(stmt["knot_rows"]),
            int(stmt["knot_cols"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_fourier_vector(model, stmt):
    _check(
        model.add_fourier_vector_variable(
            stmt["name"], int(stmt["length"]), int(stmt["num_freq"]), stmt["min"], stmt["max"]
        ),
        model,
        stmt,
    )


def _build_fourier_matrix(model, stmt):
    _check(
        model.add_fourier_matrix_variable(
            stmt["name"],
            int(stmt["rows"]),
            int(stmt["cols"]),
            int(stmt["freq_rows"]),
            int(stmt["freq_cols"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


def _build_fourier_correction_vector(model, stmt):
    _check(
        model.add_fourier_correction_vector_variable(
            stmt["name"], list(stmt["initial"]), int(stmt["num_freq"]), stmt["min"], stmt["max"]
        ),
        model,
        stmt,
    )


def _build_fourier_correction_matrix(model, stmt):
    _check(
        model.add_fourier_correction_matrix_variable(
            stmt["name"],
            [list(r) for r in stmt["initial"]],
            int(stmt["freq_rows"]),
            int(stmt["freq_cols"]),
            stmt["min"],
            stmt["max"],
        ),
        model,
        stmt,
    )


# ---- non-variable scaffolding: outputs, expressions, constraints, objectives, groups, run knobs


def _build_output(model, stmt):
    _check(model.add_output_variable(stmt["name"]), model, stmt)


def _build_output_vector(model, stmt):
    _check(model.add_output_vector_variable(stmt["name"], int(stmt["n"])), model, stmt)


def _build_output_matrix(model, stmt):
    _check(
        model.add_output_matrix_variable(stmt["name"], int(stmt["rows"]), int(stmt["cols"])),
        model,
        stmt,
    )


def _build_output_nd(model, stmt):
    dims = [int(d) for d in stmt["dims"]]
    _check(model.add_output_nd_variable(stmt["name"], dims), model, stmt)


def _build_output_statistic(model, stmt):
    _check(
        model.set_output_statistic(stmt["name"], int(stmt["statistic"]), stmt.get("value")),
        model,
        stmt,
    )


def _build_expression(model, stmt):
    _check(model.add_expression_variable(stmt["name"], stmt["expression"]), model, stmt)


def _build_constraint(model, stmt):
    _check(model.add_constraint(stmt["name"], stmt["expression"]), model, stmt)


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


def _build_objective_goal(model, stmt):
    _check(model.set_objective_goal(stmt["name"], stmt["min"], stmt["max"]), model, stmt)


def _build_objective_confidence(model, stmt):
    _check(
        model.set_objective_confidence(
            stmt["name"], int(stmt["ctype"]), int(stmt["clevel"]), stmt["epct"]
        ),
        model,
        stmt,
    )


def _build_objective_statistic(model, stmt):
    _check(
        model.set_objective_statistic(stmt["name"], int(stmt["statistic"]), stmt.get("value")),
        model,
        stmt,
    )


def _build_fix_group(model, stmt):
    _check(model.add_fix_group(stmt["name"], list(stmt["vars"])), model, stmt)


def _build_float_group(model, stmt):
    _check(model.add_float_group(stmt["name"], list(stmt["vars"])), model, stmt)


def _build_exchange(model, stmt):
    _check(
        model.add_exchangeable_group(stmt["name"], [list(g) for g in stmt["groups"]]), model, stmt
    )


def _build_exclusive(model, stmt):
    _check(model.add_variable_exclusive_range(stmt["name"], stmt["min"], stmt["max"]), model, stmt)


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
    "CONTINUOUS": _build_continuous,
    "INTEGER": _build_integer,
    "DISCRETE": _build_discrete,
    "BINARY": _build_binary,
    "DESIGN": _build_design,
    "ENUMERATION": _build_enumeration,
    "VECTOR": _build_vector,
    "MATRIX": _build_matrix,
    "BINARYVECTOR": _build_binary_vector,
    "BINARYMATRIX": _build_binary_matrix,
    "INTEGERVECTOR": _build_integer_vector,
    "INTEGERMATRIX": _build_integer_matrix,
    "DISCRETEVECTOR": _build_discrete_vector,
    "DISCRETEMATRIX": _build_discrete_matrix,
    "NDVARIABLE": _build_nd,
    "INTEGERNVARIABLE": _build_integer_nd,
    "DISCRETENDVARIABLE": _build_discrete_nd,
    "BINARYNDVARIABLE": _build_binary_nd,
    "PERMUTATION": _build_permutation,
    "TUPLE": _build_tuple,
    "GEOLOCATION": _build_geolocation,
    "SELECTION": _build_selection,
    "ANGLE": _build_angle,
    "SPHERICALANGLE": _build_spherical_angle,
    "MIXTURE": _build_mixture,
    "SEQUENCE": _build_sequence,
    "MULTISTAGESEQUENCE": _build_multi_stage_sequence,
    "POLICYSEQUENCE": _build_policy_sequence,
    "CHOICE": _build_choice,
    "MULTICHOICE": _build_multi_choice,
    "ASSIGNMENT": _build_assignment,
    "ITEMSELECTION": _build_item_selection,
    "BUDGETEDSELECTION": _build_budgeted_selection,
    "PARTITION": _build_partition,
    "DAG": _build_dag,
    "TREE": _build_tree,
    "TOPOLOGICALSEQUENCE": _build_topological_sequence,
    "PATH": _build_path,
    "REGION": _build_region,
    "SPHERICALREGION": _build_spherical_region,
    "LOWRANK": _build_lowrank,
    "LOWRANKCORRECTION": _build_lowrank_correction,
    "LOWRANKINCREMENTAL": _build_lowrank_incremental,
    "TIMESERIESVECTOR": _build_timeseries_vector,
    "TIMESERIESCORRECTIONVECTOR": _build_timeseries_correction_vector,
    "TIMESERIESINCREMENTALVECTOR": _build_timeseries_incremental_vector,
    "BSPLINEVECTOR": _build_bspline_vector,
    "SPLINEVECTOR": _build_spline_vector,
    "BSPLINECORRECTIONVECTOR": _build_bspline_correction_vector,
    "SPLINECORRECTIONVECTOR": _build_spline_correction_vector,
    "BSPLINEMATRIX": _build_bspline_matrix,
    "SPLINEMATRIX": _build_spline_matrix,
    "BSPLINECORRECTIONMATRIX": _build_bspline_correction_matrix,
    "SPLINECORRECTIONMATRIX": _build_spline_correction_matrix,
    "FOURIERVECTOR": _build_fourier_vector,
    "FOURIERMATRIX": _build_fourier_matrix,
    "FOURIERCORRECTIONVECTOR": _build_fourier_correction_vector,
    "FOURIERCORRECTIONMATRIX": _build_fourier_correction_matrix,
    # non-variable scaffolding
    "OUTPUT": _build_output,
    "OUTPUTVECTOR": _build_output_vector,
    "OUTPUTMATRIX": _build_output_matrix,
    "OUTPUTNDVARIABLE": _build_output_nd,
    "OUTPUTSTATISTIC": _build_output_statistic,
    "EXPRESSION": _build_expression,
    "CONSTRAINT": _build_constraint,
    "OBJECTIVE": _build_objective,
    "OBJECTIVEGOAL": _build_objective_goal,
    "OBJECTIVECONFIDENCE": _build_objective_confidence,
    "OBJECTIVESTATISTIC": _build_objective_statistic,
    "FIXGROUP": _build_fix_group,
    "FLOATGROUP": _build_float_group,
    "EXCHANGE": _build_exchange,
    "EXCLUSIVE": _build_exclusive,
    # run settings. LICENSE and RANDOMSEED are absent on purpose: OptQuestSampler applies them
    # in every construction mode through its own license=/seed= kwargs, so that secrets stay
    # out of statements/YAML. A LICENSE or RANDOMSEED statement here would be a second,
    # redundant path to the same setting, so the declarative-construction loop skips both by
    # name instead of dispatching them here.
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


def build_model_statement(model, stmt: Mapping[str, Any]) -> None:
    """Apply one statement to a live typed OptQuestModel (the declarative construction mode's
    model-building counterpart to build() above). Raises ValueError for LICENSE/RANDOMSEED
    (handled via OptQuestSampler's own kwargs; see callers) and NotImplementedError for the
    Adaptive Sampling commands and for anything with no registered builder."""
    cmd = stmt["cmd"].upper()
    if cmd in ("LICENSE", "RANDOMSEED"):
        raise ValueError(
            f"{cmd} statements are not applied here -- pass license=/seed= to OptQuestSampler() "
            f"instead (consistent across all construction modes; also keeps secrets out of your "
            f"statements/YAML)."
        )
    if cmd in ADAPTIVE_SAMPLING_ONLY_CMDS:
        raise NotImplementedError(
            f"{cmd} configures Adaptive Sampling (response-surface exploration), which "
            f"OptQuestSampler does not drive -- it always runs the optimization loop. Not "
            f"supported here."
        )
    builder = MODEL_BUILDERS.get(cmd)
    if builder is None:
        raise NotImplementedError(
            f"No model-builder registered for OptQuest command {cmd!r} "
            f"(statement: {dict(stmt)!r}). This type is not supported by "
            f"OptQuestSampler's declarative construction mode."
        )
    builder(model, stmt)


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
    "ITEMSELECTION": _item_selection,
    "BUDGETEDSELECTION": _budgeted_selection,
    "PARTITION": _partition,
    "DAG": _dag,
    "TREE": _tree,
    "TOPOLOGICALSEQUENCE": _topological_sequence,
    "PATH": _path,
    "REGION": _region,
    "SPHERICALREGION": _spherical_region,
    "LOWRANK": _lowrank,
    "LOWRANKCORRECTION": _lowrank_correction,
    "LOWRANKINCREMENTAL": _lowrank_correction,
    "TIMESERIESVECTOR": _timeseries_vector,
    "TIMESERIESCORRECTIONVECTOR": _timeseries_correction_vector,
    "TIMESERIESINCREMENTALVECTOR": _timeseries_correction_vector,
    "BSPLINEVECTOR": _spline_vector,
    "SPLINEVECTOR": _spline_vector,
    "BSPLINECORRECTIONVECTOR": _spline_correction_vector,
    "SPLINECORRECTIONVECTOR": _spline_correction_vector,
    "BSPLINEMATRIX": _spline_matrix,
    "SPLINEMATRIX": _spline_matrix,
    "BSPLINECORRECTIONMATRIX": _spline_correction_matrix,
    "SPLINECORRECTIONMATRIX": _spline_correction_matrix,
    "FOURIERVECTOR": _fourier_vector,
    "FOURIERMATRIX": _fourier_matrix,
    "FOURIERCORRECTIONVECTOR": _fourier_correction_vector,
    "FOURIERCORRECTIONMATRIX": _fourier_correction_matrix,
}

# Commands that modify an already-declared variable/output (rather than declaring a new one) or
# are otherwise not part of the Optuna-facing decision search space. Callers recognize and skip
# them: they add no Optuna params, and they are not errors.
SKIP_CMDS = frozenset(
    {
        "EXCLUSIVE",  # narrows an existing variable's domain by excluding a sub-range; Optuna
        # has no Distribution that can represent a "hole". The engine enforces the exclusion,
        # but the Optuna-facing distribution does not show it. Known limitation: Optuna may
        # sometimes suggest a value in the excluded range for OptQuest to reject/repair.
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
    return handler(stmt)


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
