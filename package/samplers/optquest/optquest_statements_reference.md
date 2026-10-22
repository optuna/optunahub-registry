# The `statements=` reference

This document is for people who use OptQuest from Ray Tune, Hydra or Optuna, and for
clients of the OptQuest REST API and MCP server, which accept a model as the same list of
statements. If you write a model in Python with `OptQuestModel`, read
`API/next/Python/README.md` instead.

Three of OptTek's optimization-framework integrations (the Ray Tune searcher (`optquest-ray`),
the Hydra sweeper (`hydra-optquest-sweeper`), and the Optuna sampler (via OptunaHub)) accept a
**statements** construction mode alongside their native, framework-specific way of declaring a
search space:

| Integration | Constructor / config argument            |
| ----------- | ---------------------------------------- |
| Ray Tune    | `OptQuestSearcher(statements=[...])`     |
| Hydra       | `hydra.sweeper.statements` (a YAML list) |
| Optuna      | `OptQuestSampler(statements=[...])`      |

`statements=` is a plain list of dicts. Each dict declares one element of an OptQuest model (a
decision variable, an output, a constraint, an objective, or a handful of other model elements)
using the same command vocabulary as the OptQuest engine. It is the
**recommended way to reach OptQuest's manifold variable types** (DAGs, mixtures, assignments,
sequences, and the rest) and its permutation variable: none of the three integrations' own native
config idioms (Ray Tune's `tune.uniform`/`tune.choice`, Hydra's `--multirun` override grammar,
Optuna's `trial.suggest_*`) have any way to express them.

```python
statements = [
    {"cmd": "CONTINUOUS", "name": "x", "min": -5.0, "max": 5.0},
    {"cmd": "DAG", "name": "flow", "nodes": ["A", "B", "C"]},
    {"cmd": "OUTPUT", "name": "cost"},
    {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "cost"},
]
```

Every dict has one required key, `"cmd"`, whose value is the command (case-insensitive on the engine
side, but written upper-case by convention). The remaining keys are that command's fields: the
same fields, in the same units, as the corresponding `OptQuestModel` method
(`model.add_continuous_variable(name, min, max)` ↔ `{"cmd": "CONTINUOUS", "name": ..., "min": ..., "max": ...}`) and the STDIO command-line protocol (`CONTINUOUS <name> <min> <max>`)
documented in the OptQuest Optimization API help. The key names are the ones in the tables
below; a few differ from the typed API's parameter names (for example `VECTOR` takes `size`, and
`PARALLELHINT` takes `threads`).

This document is the field-by-field reference for that dict shape. It also gives the *value*
that each variable has when it comes back to your code. That part is specific to these three
integrations, because the engine and its typed API have no notion of a "Python value".

## The same model in the YAML format

`optquest_cli.py` reads a model from YAML. Some keys have different names there. This table
gives the YAML form, the statement, and the command line that the statement becomes.

| YAML (`optquest_cli.py`)                                     | Statement (`cmd` and its keys)                                                            | Command line                                        |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `variables: [{name: x, type: continuous, min: 0, max: 10}]`  | `{"cmd": "CONTINUOUS", "name": "x", "min": 0, "max": 10}`                                 | `CONTINUOUS x 0 10`                                 |
| a variable of any other `type`, with its parameters          | `cmd` is the type in upper case, with the same parameter keys                             | the command, then the values in the order of `args` |
| `outputs: [{name: cost}]`                                    | `{"cmd": "OUTPUT", "name": "cost"}`                                                       | `OUTPUT cost`                                       |
| an output with `statistic: mean`                             | `{"cmd": "OUTPUTSTATISTIC", "name": "cost", "statistic": 1, "value": 0}`                  | `OUTPUTSTATISTIC cost 1 0`                          |
| `expressions: [{name: margin, expr: "price - cost"}]`        | `{"cmd": "EXPRESSION", "name": "margin", "expression": "price - cost"}`                   | `EXPRESSION margin price - cost`                    |
| `objectives: [{name: obj, sense: min, expr: cost}]`          | `{"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "cost"}`           | `OBJECTIVE obj min cost`                            |
| an objective with `statistic`                                | `OBJECTIVESTATISTIC`: `name`, `statistic`, `value`                                        | `OBJECTIVESTATISTIC obj 1 0`                        |
| an objective with `goal: {min, max}`                         | `OBJECTIVEGOAL`: `name`, `min`, `max`                                                     | `OBJECTIVEGOAL obj 0 10`                            |
| an objective with `confidence: {type, level, error_pct}`     | `OBJECTIVECONFIDENCE`: `name`, `ctype`, `clevel`, `epct`                                  | `OBJECTIVECONFIDENCE obj 1 3 0.05`                  |
| `constraints: [{name: cap, expr: "x + y <= 8"}]`             | `{"cmd": "CONSTRAINT", "name": "cap", "expression": "x + y <= 8"}`                        | `CONSTRAINT cap x + y <= 8`                         |
| `selections: [{name, vars}]`                                 | `SELECTION`: `name`, `vars`                                                               | `SELECTION s a b`                                   |
| `exchangeable_groups: [{name, groups}]`                      | `EXCHANGE`: `name`, `groups`                                                              | `EXCHANGE g a,b;c,d`                                |
| `fix_groups` or `float_groups: [{name, vars}]`               | `FIXGROUP` or `FLOATGROUP`: `name`, `vars`                                                | `FIXGROUP g a b`                                    |
| `metrics: [{name, response, error}]`                         | `METRIC`: `name`, `response`, `error`                                                     | `METRIC m y;0`                                      |
| `optimization.replications: 5` or `{min: 2, max: 5}`         | `REPLICATIONS`: `min`, `max` (5 gives `min` = `max` = 5)                                  | `REPLICATIONS 2 5`                                  |
| `optimization.serial_replications: true`                     | `SERIALREPLICATIONS`                                                                      | `SERIALREPLICATIONS`                                |
| `optimization.random_seed`                                   | `RANDOMSEED`: `seed`                                                                      | `RANDOMSEED 7`                                      |
| `optimization.stop_cycles`                                   | `STOPCYCLES`: `cycles`                                                                    | `STOPCYCLES 3`                                      |
| `optimization.stop_improvement: {threshold, cycles}`         | `STOPIMPROVEMENT`: `threshold`, `cycles`                                                  | `STOPIMPROVEMENT 0.01 50`                           |
| `optimization.stop_time`                                     | `STOPTIME`: `seconds`                                                                     | `STOPTIME 600`                                      |
| `optimization.stop_on_milp: true`                            | `STOPMILP`                                                                                | `STOPMILP`                                          |
| `optimization.suggested_solutions_only: true`                | `NOGENERATE`                                                                              | `NOGENERATE`                                        |
| `optimization.parallel_hint`                                 | `PARALLELHINT`: `threads`                                                                 | `PARALLELHINT 6`                                    |
| `optimization.max_trials`, `optimization.workers`            | none: the CLI applies these itself                                                        | none                                                |
| `sampling.method`, `sampling.variogram`, `sampling.rds_bins` | `SAMPLE`: `method`; `VARIOGRAM`: `variogram`; `RDSBINS`: `name`, `distribution`, `params` | `SAMPLE 1`                                          |
| `license`                                                    | `LICENSE`: `license_id`                                                                   | `LICENSE <key>`                                     |
| `logging.setup`, `logging.solutions`                         | `LOGSETUP`, `LOGSOLUTIONS`: `file`                                                        | `LOGSETUP setup.log`                                |

The YAML format takes statistic names (`none`, `mean`, `median`, `percentile`, `stddev` or
`std`, `variance` or `var`, `coeffofvar` or `cv`, `min`, `max`, `sum`). A statement takes the
code: 0, 1, 2, 3, 4, 6, 8, 14, 15, 16 in the same order. The YAML format also takes sample
method names, and a statement takes the method code.

## Command reference

Fields are listed by their dict key. A field marked `?` is optional. `name` (the variable's own
name) is required on every command that has one.

### Scalar variables

| `cmd`         | Fields                                                  | Notes                                                                                                                                                                                                                            |
| ------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONTINUOUS`  | `name`, `min`, `max`                                    |                                                                                                                                                                                                                                  |
| `INTEGER`     | `name`, `min`, `max`                                    | The values `min`, `min` + 1, `min` + 2, and so on, up to `max`. Give whole-number bounds to get whole-number values.                                                                                                             |
| `DISCRETE`    | `name`, `min`, `max`, `step`                            | The values `min`, `min` + `step`, `min` + 2 × `step`, and so on. The largest value is the one nearest to `max`, so it can differ from `max` when `max` - `min` is not a whole multiple of `step`. `step` must be greater than 0. |
| `BINARY`      | `name`                                                  |                                                                                                                                                                                                                                  |
| `DESIGN`      | `name`, `num`                                           | A 1-based choice among `num` unnamed alternatives.                                                                                                                                                                               |
| `ENUMERATION` | `name`, `values` (list of numbers)                      | The declared values themselves, not indices into them.                                                                                                                                                                           |
| `TUPLE`       | `name`, `tuples` (list of rows, each a list of numbers) | The engine reports the 1-based index of the selected row.                                                                                                                                                                        |
| `GEOLOCATION` | `name`, `coords` (list of `[lat, lon]` pairs)           | The engine reports the 1-based index of the selected pair.                                                                                                                                                                       |

### Container variables (vector / matrix / N-D)

| `cmd`                | Fields                                       | Notes                                                                                                         |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `VECTOR`             | `name`, `size`, `min`, `max`                 |                                                                                                               |
| `MATRIX`             | `name`, `rows`, `cols`, `min`, `max`         |                                                                                                               |
| `NDVARIABLE`         | `name`, `dims` (list of ints), `min`, `max`  |                                                                                                               |
| `INTEGERVECTOR`      | `name`, `size`, `min`, `max`                 |                                                                                                               |
| `INTEGERMATRIX`      | `name`, `rows`, `cols`, `min`, `max`         |                                                                                                               |
| `INTEGERNVARIABLE`   | `name`, `dims`, `min`, `max`                 | The command name has no D, unlike `DISCRETENDVARIABLE` and `BINARYNDVARIABLE`. The engine spells it this way. |
| `DISCRETEVECTOR`     | `name`, `size`, `min`, `max`, `step`         |                                                                                                               |
| `DISCRETEMATRIX`     | `name`, `rows`, `cols`, `min`, `max`, `step` |                                                                                                               |
| `DISCRETENDVARIABLE` | `name`, `dims`, `min`, `max`, `step`         |                                                                                                               |
| `BINARYVECTOR`       | `name`, `k`                                  |                                                                                                               |
| `BINARYMATRIX`       | `name`, `rows`, `cols`                       |                                                                                                               |
| `BINARYNDVARIABLE`   | `name`, `dims`                               |                                                                                                               |

The elements of these variables are ordinary input variables.

- **Variable names**: a vector has `<name>@1` to `<name>@n`, one for each element. A matrix has
  `<name>@<row>,<col>` for each element, with 1-based row and column numbers, row by row. An
  N-dimensional variable has `<name>@<i1>,<i2>,...` for each element, with one 1-based index for
  each dimension; the last index changes fastest.
- A name that contains a comma must be in brackets in an expression: `[<name>@1,2]`.
- Integer elements follow the `INTEGER` rule and discrete elements follow the `DISCRETE` rule
  above. Binary elements hold 0 or 1.

### Permutation

| `cmd`         | Fields                            | Notes                                                                              |
| ------------- | --------------------------------- | ---------------------------------------------------------------------------------- |
| `PERMUTATION` | `name`, `items` (list of strings) | An ordinary decision variable, not a manifold type: one 1-based position per item. |

### Manifold variables

In every solution that the engine generates, the values of a manifold variable meet the guarantee in
the Notes column. The engine can change the values in a suggested solution so that they meet the
guarantee. It uses the values of an evaluated or a required solution as given. The values of a
manifold variable are inputs, except the `POLICYSEQUENCE` outputs that the Notes column names.
`ASSIGNMENT`, `ITEMSELECTION` and `BUDGETEDSELECTION` work differently: the engine adds their rules
to the model as constraints, so a solution that breaks a rule is infeasible.

| `cmd`                 | Fields                                                                                                                                                                         | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CHOICE`              | `name`, `k`                                                                                                                                                                    | **Variable names**: `<name>@choice`, the 1-based index of the selected option, a whole number from 1 to `k`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MULTICHOICE`         | `name`, `choices` (list of names), `min_select`, `max_select`                                                                                                                  | **Variable names**: `<name>@<choice>` for each choice: 1 if the choice is selected and 0 if it is not. The number of selected choices is from `min_select` to `max_select`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `MIXTURE`             | `name`, `components` (an int count or a list of names), `min`?, `max`? (each a single number broadcast to every component, or a per-component list), `sum_to`? (default `1.0`) | **Variable names**: `<name>@<component>` for each component, in the order of `components`: the weight of that component. The weights sum to `sum_to` within 1e-12, and each weight is within the bounds of its component. A negative lower bound lets the weights go below zero, down to that bound. `sum_to` must be greater than 0. The statements= codecs below need named `components` (a list of strings), not a bare count.                                                                                                                                                                                                                          |
| `SEQUENCE`            | `name`, `components` (list of names), `top_k`?                                                                                                                                 | **Variable names**: `<name>@<component>` for each component: the integer rank of that component, where 1 is the highest priority. A full ranking is a permutation of 1 to n. In a top-k ranking, the first `top_k` components get the ranks 1 to `top_k`, and every other component gets the rank `top_k` + 1. Omit `top_k`, or give n or more, for a full ranking. A `top_k` less than 1 is used as 1.                                                                                                                                                                                                                                                    |
| `MULTISTAGESEQUENCE`  | `name`, `stages` (list of stage groups, each a list of names)                                                                                                                  | **Variable names**: `<name>@<component>` for every component, stage by stage: the integer rank of that component within its stage. The ranks in each stage are a permutation of 1 to the number of components in that stage. The engine ranks each stage separately. A component name can appear in only one stage; to rank the same item in two stages, give it a different name in each stage.                                                                                                                                                                                                                                                           |
| `POLICYSEQUENCE`      | `name`, `items` (list of names), `features` (list of names), `matrix` (rows × features)                                                                                        | **Variable names**: `<name>@<item>` for each item, an input: the priority score of that item. Sort the items by descending score to get the ranked list. `<name>@<feature>` (the coefficient: score per unit of the feature), `<name>@<feature>@contribution`, `<name>@<feature>@confidence` and `<name>@<item>@confidence` are outputs that the engine computes, also for evaluated and required solutions. Every item name and feature name must be different from all the others. Give the feature values in their own units: the engine scales the columns. Field order here matches the wire/statement grammar; the typed API takes the matrix first. |
| `TOPOLOGICALSEQUENCE` | `name`, `tasks` (list of names), `predecessors`? (a dict of task → list of predecessor tasks)                                                                                  | **Variable names**: `<name>@<task>` for each task: the integer rank of that task in the order, a permutation of 1 to n. A predecessor always has a smaller rank than the task that depends on it.                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `ASSIGNMENT`          | `name`, `m`, `n`, `must_assign`?, `bin_cap_min`?, `bin_cap_max`?                                                                                                               | **Variable names**: `<name>@1` to `<name>@m`, one for each item: the 1-based index of the bin that holds the item, or 0 when the item is in no bin. In the simple form, every item is in exactly one bin. The rules are model constraints, and the engine refuses capacities that no assignment can meet. Simple form omits the optional fields; homogeneous form gives each as a single value; heterogeneous form gives each as a per-item/per-bin list. `must_assign`/`bin_cap_min`/`bin_cap_max` must appear together or not at all.                                                                                                                    |
| `ITEMSELECTION`       | Exact: `name`, `n`, `k`. Range: `name`, `n`, `k_min`, `k_max`.                                                                                                                 | **Variable names**: `<name>@1` to `<name>@n`, one for each candidate: 1 if the candidate is selected and 0 if it is not. The count rule is a model constraint, and the engine refuses counts that no selection can meet. This is not the same as `SELECTION`, which turns a group of variables on or off.                                                                                                                                                                                                                                                                                                                                                  |
| `BUDGETEDSELECTION`   | Simple: `name`, `n`, `budget_max`, `costs`. Full: `name`, `n`, `budget_min`, `budget_max`, `k_min`, `k_max`, `costs`.                                                          | **Variable names**: the same as `ITEMSELECTION`. The total cost of the selected candidates is at most `budget_max` (full form: from `budget_min` to `budget_max`, with from `k_min` to `k_max` candidates). A cost can be negative. The rules are model constraints, and the engine refuses counts and budgets that no selection can meet.                                                                                                                                                                                                                                                                                                                 |
| `PARTITION`           | `name`, `m`, `k`                                                                                                                                                               | **Variable names**: `<name>@1` to `<name>@m`, one for each item: the 1-based number of the group that holds the item. There are exactly `k` groups, and no group is empty. Item 1 is always in group 1. For each later item, the group number is at most one more than the largest group number among the items before it.                                                                                                                                                                                                                                                                                                                                 |
| `DAG`                 | `name`, `nodes` (list of names), `start`?, `end`?, `connected`? (true or false), `predecessors`? (a dict of node → list of predecessor nodes)                                  | **Variable names**: `<name>@<from>,<to>` for every ordered pair of nodes, including a node paired with itself: 1 if the edge from `from` to `to` is present and 0 if it is not. The diagonal is always 0. The edges form no cycle. There must be at least one node. The `start` node has no incoming edge, and the `end` node has no outgoing edge. With `connected: true`, every node can be reached from `start` and every node can reach `end`; `connected` needs `start`, `end` or both. No directed path runs from a node to one of its predecessors.                                                                                                 |
| `TREE`                | `name`, `nodes` (list of names), `root`?, `predecessors`? (a dict of node → list of predecessor nodes)                                                                         | **Variable names**: the same as `DAG`. The edges form a directed spanning tree: one node is the root, every other node has exactly one incoming edge, and every node can be reached from the root. The tree has exactly n - 1 edges. The root is `root` when it is given, and any node otherwise. No path from the root to a predecessor passes through the node that depends on it.                                                                                                                                                                                                                                                                       |
| `PATH`                | `name`, `nodes`, `path`? (a dict with optional `start`, `end`, `required` keys)                                                                                                | **Variable names**: `<name>@<node>` for each node: the 1-based position of that node on the path, or 0 if the node is not on the path. The start node has position 1. The end node, when different from the start node, has the last position. Every required node is on the path, but not necessarily in the middle. The path can have only one node. `nodes` must hold at least two nodes.                                                                                                                                                                                                                                                               |
| `ANGLE`               | `name`, `min_deg`, `max_deg`                                                                                                                                                   | **Variable names**: `<name>@theta`, the angle in degrees. The angle is on the arc from `min_deg` to `max_deg` in the direction of increasing angle, so an arc from 320 to 40 crosses 0. A difference of 360 or more is the full circle. The engine reports the angle as its equivalent in \[0, 360), not in the declared range.                                                                                                                                                                                                                                                                                                                            |
| `SPHERICALANGLE`      | `name`, `min_theta_deg`, `max_theta_deg`, `min_phi_deg`, `max_phi_deg`                                                                                                         | **Variable names**: `<name>@theta`, the elevation in degrees, from -90 (straight down) through 0 (the horizon) to 90 (straight up); and `<name>@phi`, the azimuth in degrees, where 0 is north and 90 is east. The elevation is from `min_theta_deg` to `max_theta_deg`, and the engine reports it as it is. The azimuth is on the arc from `min_phi_deg` to `max_phi_deg`, with the same rules as `ANGLE`, and the engine reports it in \[0, 360).                                                                                                                                                                                                        |
| `REGION`              | `name`, `vertices` (list of `[x, y]` pairs)                                                                                                                                    | **Variable names**: `<name>@x` and `<name>@y`, the coordinates of the point, in the units of the polygon. The point is inside the polygon or on its boundary. The polygon can be non-convex. There must be at least three vertices.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `SPHERICALREGION`     | `name`, `vertices` (list of `[lat, lon]` pairs)                                                                                                                                | **Variable names**: `<name>@theta`, the latitude in degrees; and `<name>@phi`, the longitude in degrees, from -180 to 180. The point is inside the polygon or on its boundary. The edges are great-circle arcs. List the vertices clockwise, as on a map with north up: the region is on your right as you walk from each vertex to the next. The same vertices in counterclockwise order give the rest of the Earth. A region can enclose a pole, span more than 180 degrees of longitude, or be larger than a hemisphere.                                                                                                                                |

### Smooth variables (low-rank / time-series / spline / Fourier)

These are manifold variables too. In every solution that the engine generates, every entry is
from `min` to `max`. The bounds apply to the values that the simulator receives.

| `cmd`                                                | Fields                                                                             | Notes                                                                                                                                                                                                                                                                  |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LOWRANK`                                            | `name`, `rows`, `cols`, `rank`, `min`, `max`                                       | The matrix is the constant matrix (`min` + `max`) / 2 plus a correction of rank at most `rank`, so the matrix itself can have rank `rank` + 1.                                                                                                                         |
| `LOWRANKCORRECTION`                                  | `name`, `initial` (a matrix), `rank`, `min`, `max`                                 | The matrix is `initial` plus a correction of rank at most `rank`. The engine first moves each entry of `initial` into \[`min`, `max`\].                                                                                                                                |
| `LOWRANKINCREMENTAL`                                 | `name`, `initial`, `rank`, `min`, `max`                                            | The matrix is `initial` plus a correction of rank at most 1. `rank` is the number of this step in the sequence and does not change the values. Each variable does not keep the corrections of earlier variables: pass the best value of the previous run as `initial`. |
| `TIMESERIESVECTOR`                                   | `name`, `long_scale`, `short_scale`, `rank`, `min`, `max`                          | The engine arranges the values row by row as a `long_scale` by `short_scale` matrix. The matrix is the constant matrix (`min` + `max`) / 2 plus a correction of rank at most `rank`.                                                                                   |
| `TIMESERIESCORRECTIONVECTOR`                         | `name`, `initial` (a flat list), `long_scale`, `short_scale`, `rank`, `min`, `max` | The matrix is `initial` plus a correction of rank at most `rank`.                                                                                                                                                                                                      |
| `TIMESERIESINCREMENTALVECTOR`                        | `name`, `initial`, `long_scale`, `short_scale`, `rank`, `min`, `max`               | The matrix is `initial` plus a correction of rank at most 1. `rank` is the number of this step and does not change the values.                                                                                                                                         |
| `BSPLINEVECTOR` / `SPLINEVECTOR`                     | `name`, `length`, `num_knots`, `min`, `max`                                        | A smooth curve with `num_knots` knots at evenly spaced elements. The B-spline curve does not pass through its knots. The spline (centripetal Catmull-Rom) curve passes through its knots.                                                                              |
| `BSPLINECORRECTIONVECTOR` / `SPLINECORRECTIONVECTOR` | `name`, `initial` (a flat list), `num_knots`, `min`, `max`                         | `initial` plus a correction on the same kind of curve.                                                                                                                                                                                                                 |
| `BSPLINEMATRIX` / `SPLINEMATRIX`                     | `name`, `rows`, `cols`, `knot_rows`, `knot_cols`, `min`, `max`                     | A smooth surface with a `knot_rows` by `knot_cols` grid of knots. The B-spline surface does not pass through its knots. The spline surface passes through its knots.                                                                                                   |
| `BSPLINECORRECTIONMATRIX` / `SPLINECORRECTIONMATRIX` | `name`, `initial` (a matrix), `knot_rows`, `knot_cols`, `min`, `max`               | `initial` plus a correction on the same kind of surface.                                                                                                                                                                                                               |
| `FOURIERVECTOR`                                      | `name`, `length`, `num_freq`, `min`, `max`                                         | A sum of the `num_freq` lowest-frequency cosine terms, including the constant term.                                                                                                                                                                                    |
| `FOURIERMATRIX`                                      | `name`, `rows`, `cols`, `freq_rows`, `freq_cols`, `min`, `max`                     | A sum of the lowest-frequency cosine terms: `freq_rows` terms down the columns and `freq_cols` terms along the rows.                                                                                                                                                   |
| `FOURIERCORRECTIONVECTOR`                            | `name`, `initial` (a flat list), `num_freq`, `min`, `max`                          | `initial` plus a correction that is a sum of the `num_freq` lowest-frequency cosine terms.                                                                                                                                                                             |
| `FOURIERCORRECTIONMATRIX`                            | `name`, `initial` (a matrix), `freq_rows`, `freq_cols`, `min`, `max`               | `initial` plus a correction that is a sum of the lowest-frequency cosine terms.                                                                                                                                                                                        |

**Variable names**: the vector commands and the time-series vector commands have `<name>@1` to
`<name>@n`. For a time-series vector command, n is `long_scale` × `short_scale`, and element
i × `short_scale` + j + 1 is the entry in row i and column j of the matrix, with 0-based i and j.
The matrix commands have `<name>@<row>,<col>`, with 1-based row and column numbers, row by row.
Write a name that contains a comma in brackets in an expression: `[<name>@1,2]`. Every correction
command first moves each entry of `initial` into \[`min`, `max`\].

**`TIMESERIESCORRECTIONVECTOR`/`TIMESERIESINCREMENTALVECTOR`'s `initial` must have exactly
`long_scale * short_scale` elements.** Unlike every other correction command above (where the engine
reads the element count from the length of `initial`), the time-series vector pair has `long_scale`
and `short_scale` as separate fields that must already agree with `initial`'s length. A mismatch is
rejected up front with an error.

### Variable modifiers

These act on variables that are already declared. Declare the variables first, then the modifier.
A member of `SELECTION`, `EXCHANGE`, `FIXGROUP` or `FLOATGROUP`, and the name in `EXCLUSIVE`, is
the name of a scalar decision variable, or an element of a vector, matrix, N-dimensional,
permutation, item-selection or budgeted-selection variable, such as `v@1`, `m@1,2`, `n@1,2,3` or
`perm@a`. The model check refuses any other name that holds `@`, for example the value
`dir@theta` of an angle variable or `v@4` for a vector of 3 elements. A member name can hold a comma
but no whitespace or `;`. `EXCHANGE` joins the members of a group with commas, so in an `EXCHANGE`
member each comma must be followed by a digit, as it is in every generated element name: the
engine reads `EXCHANGE g m@1,1,m@1,2;m@2,1,m@2,2` as the groups (`m@1,1`, `m@1,2`) and
(`m@2,1`, `m@2,2`).

| `cmd`        | Fields                                                   | Notes                                                                                                                                                                                                                                                                                                                                                    |
| ------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EXCLUSIVE`  | `name`, `min`, `max`                                     | Narrows an *already-declared* variable's domain by excluding the open sub-range `(min, max)`; not a new variable.                                                                                                                                                                                                                                        |
| `SELECTION`  | `name`, `vars` (list of already-declared variable names) | One binary flag named `name` that gates the listed variables: `1` and they participate, `0` and they are forced to their lower bound for that solution.                                                                                                                                                                                                  |
| `EXCHANGE`   | `name`, `groups` (list of lists of variable names)       | Declares identical items, one group of variables for each item. In each generated trial solution, the optimizer puts the items in ascending order, so it does not evaluate two solutions that differ only by an exchange of items. Use it only for items with the same type, bounds and role, and put a variable in no more than one exchangeable group. |
| `FIXGROUP`   | `name`, `vars`                                           | After an initial part of the search, the optimizer makes one more trial solution from each new trial solution, with these variables at their values in the best solution found so far and every other variable at its new value.                                                                                                                         |
| `FLOATGROUP` | `name`, `vars`                                           | After an initial part of the search, the optimizer makes one more trial solution from each new trial solution, with these variables at their new values and every other variable at its value in the best solution found so far.                                                                                                                         |

### Outputs, expressions, constraints, objectives

| `cmd`                 | Fields                                              | Notes                                                                                                                                                                                                                                                                                                                              |
| --------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OUTPUT`              | `name`                                              | A scalar output the simulation/task function reports back.                                                                                                                                                                                                                                                                         |
| `OUTPUTVECTOR`        | `name`, `n`                                         |                                                                                                                                                                                                                                                                                                                                    |
| `OUTPUTMATRIX`        | `name`, `rows`, `cols`                              |                                                                                                                                                                                                                                                                                                                                    |
| `OUTPUTNDVARIABLE`    | `name`, `dims`                                      |                                                                                                                                                                                                                                                                                                                                    |
| `EXPRESSION`          | `name`, `expression` (a string)                     | A named intermediate expression over already-declared variables/outputs.                                                                                                                                                                                                                                                           |
| `CONSTRAINT`          | `name`, `expression`                                | A real OptQuest constraint: the engine will not propose a trial that violates an input-only constraint, and drives toward feasibility for one that uses an output.                                                                                                                                                                 |
| `OBJECTIVE`           | `name`, `direction` (`"min"`/`"max"`), `expression` | See "Objectives must be identity pass-throughs" below.                                                                                                                                                                                                                                                                             |
| `OBJECTIVEGOAL`       | `name`, `min`, `max`                                | Sets a target range for an already-declared objective in a multi-objective model: the engine prefers the part of the Pareto frontier where that objective lies in `[min, max]`. Ignored in a single-objective model.                                                                                                               |
| `OBJECTIVECONFIDENCE` | `name`, `ctype`, `clevel`, `epct`                   | Statistical confidence requirement for a replicated (stochastic) objective; it lets the engine stop replicating a trial early once the interval is met. `epct` is the error as a fraction greater than 0 and less than 1 (0.05 is 5%). Without this statement, the engine runs the maximum number of replications for every trial. |
| `OBJECTIVESTATISTIC`  | `name`, `statistic`, `value`                        | The per-replication statistic (mean, percentile, std-dev, ...) that the objective uses; `value` is the percentile as a fraction in `[0, 1]` when `statistic` is the percentile statistic, otherwise `0`.                                                                                                                           |
| `OUTPUTSTATISTIC`     | `name`, `statistic`, `value`                        | Same statistic selection as `OBJECTIVESTATISTIC`, applied to a declared `OUTPUT` instead of an objective.                                                                                                                                                                                                                          |

`OBJECTIVECONFIDENCE`/`OBJECTIVESTATISTIC`/`OUTPUTSTATISTIC`'s integer fields (`ctype`, `clevel`,
`statistic`) take the same `OQConfidence*`/`OQStatistic*` constant values as the typed API:

- `statistic`: 0 = None, 1 = Mean, 2 = Median, 3 = Percentile, 4 = StdDev, 6 = Variance,
  8 = CoeffOfVar, 14 = Min, 15 = Max, 16 = Sum.
- `ctype`: 1 = Type1, 2 = Type2.
- `clevel`: 1 = 80%, 2 = 90%, 3 = 95%, 4 = 98%, 5 = 99%, 6 = 99.9%.

### Groups and run settings

`FIXGROUP`/`FLOATGROUP`/`EXCHANGE`/`EXCLUSIVE`/`SELECTION` are listed under Variable modifiers above
(they act on already-declared variables, not new ones). The run-configuration commands are
supported through `statements=` too, with these fields:

| `cmd`                | Fields                              |
| -------------------- | ----------------------------------- |
| `REPLICATIONS`       | `min`, `max`                        |
| `SERIALREPLICATIONS` | (none)                              |
| `STOPCYCLES`         | `cycles`                            |
| `STOPIMPROVEMENT`    | `threshold`, `cycles` (at least 50) |
| `STOPTIME`           | `seconds`                           |
| `STOPMILP`           | (none)                              |
| `PARALLELHINT`       | `threads`                           |
| `LOGSETUP`           | `file`                              |
| `LOGSOLUTIONS`       | `file`                              |

There is one exception: **Hydra rejects `PARALLELHINT`** with an error. The sweeper always
sets its concurrency from `hydra.sweeper.n_jobs`, and with a second setting, one of the two
could override the other without a message. `LICENSE` and `RANDOMSEED` are excluded everywhere: pass `license=`/`seed=`
to the searcher/sweeper/sampler's own constructor instead of a statement. This is the same in
every construction mode, so your statements list contains no secrets.

All three integrations refuse the random variable commands (`NORMALRANDOM`,
`UNIFORMRANDOM`, and the rest of the probability-distribution family) and the Adaptive Sampling
configuration (`METRIC`, `SAMPLE`, `VARIOGRAM`, `RDSBINS`) in a `statements=` list: they belong to a
different workflow (response-surface exploration, or engine-redrawn noise) from the optimization
loop that these integrations run. Use the OptQuest API directly for those.

The integrations refuse `NOGENERATE` with an error, for the same reason. It restricts
the engine to solutions the caller supplies and stops it generating any of its own, which is the
batch-mapping workflow rather than a search: under a sweeper there is nothing to supply those
solutions, so a sweep with it would finish with nothing evaluated. Use the OptQuest API
directly for that workflow.

## What comes back: decoded values

This section gives the value that you receive for a variable declared through `statements=`
once the search is running:

- Ray Tune: `config["<name>"]` inside your trainable function.
- Hydra: `cfg.<name>` inside your task function.
- Optuna: the value under `params["<name>"]` your `sim(params)` function (passed to
  `sampler.wrap(sim)`) receives.

The scalar and container commands decode to plain values: a `CONTINUOUS`/
`INTEGER` variable is a plain number, a `VECTOR` is a flat list, a `MATRIX` or `NDVARIABLE` is a
correspondingly nested list. The manifold/structured types decode as follows:

| Shape                                | Commands                                                                                                                   | Decoded value                                                                                                                                          |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Bare scalar                          | `CONTINUOUS`, `INTEGER`, `DISCRETE`, `BINARY`, `ENUMERATION`                                                               | The number itself.                                                                                                                                     |
| Index resolved to its original value | `TUPLE`, `GEOLOCATION`                                                                                                     | The declared row/coordinate the engine picked, not the 1-based index that the engine reports.                                                          |
| 1-based index                        | `DESIGN`                                                                                                                   | A whole number from 1 to `num`. A design variable has no declared list: the value is a 1-based index into your own catalog.                            |
| Named sub-keys                       | `ANGLE`, `SPHERICALANGLE`, `CHOICE`, `REGION`, `SPHERICALREGION`                                                           | A single value for a one-field shape (`ANGLE`, `CHOICE`); a small dict with fixed keys (`"theta"`/`"phi"`, `"x"`/`"y"`) for a two-field shape.         |
| One value per declared item          | `MIXTURE`, `SEQUENCE`, `PERMUTATION`, `MULTICHOICE`, `TOPOLOGICALSEQUENCE`, `PATH`, `POLICYSEQUENCE`, `MULTISTAGESEQUENCE` | A dict keyed by the item/component name you declared, e.g. `{"gas": 0.4, "liquid": 0.35, "solid": 0.25}` for a `MIXTURE`.                              |
| One value per index                  | `ASSIGNMENT`, `PARTITION`, `ITEMSELECTION`, `BUDGETEDSELECTION`                                                            | A flat list, one entry per item (1-based order).                                                                                                       |
| One value per edge                   | `DAG`, `TREE`                                                                                                              | A dict with one entry per ordered pair of nodes, including a node paired with itself (always 0). **The key format differs by integration; see below.** |
| Binary gate over a group             | `SELECTION`                                                                                                                | The list of gated variable names when the gate is on, an empty list when it is off.                                                                    |
| Grid                                 | `VECTOR`, `MATRIX`, `NDVARIABLE` and every smooth-variable command                                                         | A (possibly multi-level nested) list.                                                                                                                  |

### `DAG`/`TREE` edge keys differ by integration

A `DAG`/`TREE` variable's natural value is "one entry per ordered pair of nodes": the full n by n
adjacency matrix, including each node paired with itself, whose value is always 0. The dict key
for a pair is not the same in all three integrations:

- **Optuna**: a `(from, to)` **tuple**-keyed dict, e.g. `{("A", "B"): 1, ("A", "C"): 0, ...}`.
- **Ray Tune and Hydra**: a `"from,to"` **string**-keyed dict, e.g. `{"A,B": 1, "A,C": 0, ...}`.
  The difference is deliberate: Ray Tune's
  trial tracking serializes the resolved config to JSON before a trial runs, and a
  Python tuple is not a legal JSON object key. Hydra uses the same string keys so that the
  two integrations match.

### Hydra can pass a dict-shaped value as a list of pairs

Hydra materializes a manifold variable's decoded value by writing it into your app's config as a
single override, and Hydra's override grammar has no way to *quote* a dict key, only bare
identifiers (letters, digits, `.`, `-`, `_`). Hydra passes a `MIXTURE`/`SEQUENCE`/... variable whose
item names are all safe bare identifiers (e.g. `"gas"`, `"liquid"`) as a `DictConfig`, and you can
use `cfg.<name>` directly. Hydra passes a variable with any item name that isn't a safe bare
identifier (including every `DAG`/`TREE` variable, since its `"A,B"`-style keys always contain a
comma) as a `[[key, value], ...]` list of pairs.

**Always write `dict(cfg.<name>)` rather than assuming `cfg.<name>` is already a mapping.**
`dict(...)` recovers the natural dict either way, and which of the two forms a given variable uses
is stable for the whole sweep (item names are fixed when you declare the variable, not chosen
per-trial).

Ray Tune and Optuna do not have this restriction: both give you a Python dict directly, with no
serialization step.

## Outputs, objectives, and constraints

Declaring any `OUTPUT`/`OUTPUTVECTOR`/`OUTPUTMATRIX`/`OUTPUTNDVARIABLE`/`OBJECTIVE`/`CONSTRAINT`/
`EXPRESSION` statement switches how your trainable/task/simulation function reports results back,
and the switch works a little differently per integration:

- **Ray Tune**: report a dict keyed by output name, `tune.report({"cost": ...})`, matching
  `metric=` on the searcher. A shaped output (`OUTPUTVECTOR`/`OUTPUTMATRIX`/`OUTPUTNDVARIABLE`)
  takes a nested list under its key; it's flattened to the engine's expected shape automatically.
- **Hydra**: return a dict keyed by output name from your task function, `{"cost": ...}`, instead
  of the bare float/list `hydra.sweeper.direction` expects when no `OUTPUT`-family statement is
  declared. One sweep uses only one of the two return shapes. Shaped outputs are supported the
  same way as Ray Tune.
- **Optuna**: `sampler.wrap(sim)` expects `sim(params)` to return a dict keyed by output name (or
  `None` to reject the point), the same idiom.

### Objectives must be identity pass-throughs

Across all three integrations, an `OBJECTIVE` statement's `expression` must be exactly the name
of one already-declared `OUTPUT`. A compound expression (`"cost * 2"`, `"cost + penalty"`) is
refused there; declare the computation as an `EXPRESSION` or include it in the output your task
function reports instead. `CONSTRAINT` and `EXPRESSION` take an arbitrary formula over declared
variables, outputs and expressions.

## Known restrictions

These apply the same way in all three integrations, because they come from the engine or from
the decode logic that the three integrations share:

- **`POLICYSEQUENCE`**: only the item priority scores are inputs. The coefficient, contribution and
  confidence values are outputs that the engine computes, also for evaluated and required
  solutions. You cannot set them.
- **`EXCLUSIVE`** (narrowing an existing variable's domain) is supported in all three, and
  is enforced by the engine regardless of how the search space is described.

Three restrictions are specific to a single integration:

- **Ray Tune's `add_evaluated_point`** (injecting an already-known point without asking OptQuest
  to propose it) takes scalar-shaped variables. The searcher cannot turn the natural Python
  value of a manifold variable or another structured variable back into the values that the
  engine reads, so seed those points through `CopySolution` on the OptQuest API instead.
- **Hydra seeds points through the OptQuest API rather than the sweeper**: Hydra's sweeper contract
  has no "inject an already-known point" hook.
- **Hydra rejects a `PARALLELHINT` statement** with an error; see "Groups and run settings"
  above.
