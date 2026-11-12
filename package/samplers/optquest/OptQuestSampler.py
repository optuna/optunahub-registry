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

"""OptQuestSampler: an Optuna sampler that takes each point of a study from the OptQuest engine.

How this module is organized:

- A sampler is built in one of three ways. Each has a builder method, which ends with an
  initialized engine model and the Optuna search space: plain Optuna distributions
  (_build_from_search_space), a list of OptQuest statements (_build_from_statements), or a
  model built with the typed OptQuest API (_attach_model). The last two read every variable
  through _variable_codec, which gives the Optuna distributions of each OptQuest variable type
  and the converters between the engine's values and Optuna's.
- A trial: sample_relative() asks the engine for a solution and gives Optuna its values.
  after_trial() gives the engine the result of the trial in that solution, or rejects the
  solution (_finalize_trial). sample_independent() runs only when Optuna discards one of those
  values, and warns.
- wrap() and suggest_all() are for the objective: they make the suggest calls, and turn the
  parameters of a structured variable ("<name>@<key>") into one Python value.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Set
import functools
import itertools
import math
import numbers
import re
import threading
import warnings
import weakref

import optuna


try:
    import optquest
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install optquest` to use `OptQuestSampler`.")

if not hasattr(optquest, "parse_model"):
    # parse_model, emit_model and the structured variable types are new in optquest 10.2.0.0.
    raise ImportError(
        "OptQuestSampler needs optquest 10.2.0.0 or newer. Run `pip install -U optquest`."
    )

try:
    from . import _variable_codec
except ImportError:
    # optunahub loads this module as a package (relative import above); a plain script/test
    # importing OptQuestSampler directly off sys.path needs the sibling-module fallback.
    import _variable_codec


class OptQuestOutputMismatchError(ValueError):
    """Raised by wrap() when sim(params) returns a dict whose keys don't exactly match the
    model's declared outputs."""


def _identity(value):
    return value


def _log_int_decoder(low, high):
    """from_engine converter for a log-scale IntDistribution. The engine variable for it is
    CONTINUOUS in log-space, as for a log-scale FloatDistribution, to give full resolution
    across the range. An INTEGER engine variable bounded by log(low)..log(high) would explore
    only as many distinct points as there are integers in that small log-space range, e.g. 7
    for low=1/high=1000. So the decoder exponentiates the raw engine value and rounds it to
    the nearest integer, because optuna.distributions.IntDistribution requires an int. It
    clamps the result to [low, high] to remove floating-point overshoot at the bounds (e.g.
    exp(log(high)) a small amount above high)."""

    def decode(raw_log_value):
        return min(high, max(low, round(math.exp(raw_log_value))))

    return decode


def _log_float_decoder(low, high):
    """from_engine converter for a log-scale FloatDistribution.

    The same clamp _log_int_decoder applies, and for the same reason: the engine variable is
    CONTINUOUS over log(low)..log(high), and exp() of either bound is not exactly that bound.
    Without the clamp, the decoded value of a proposal at a bound can be a fraction of an ulp
    outside [low, high], which optuna.distributions.FloatDistribution refuses. Optuna then
    discards the engine's point, draws a random one instead, and writes that one back onto the
    engine's solution, so the engine is told that it evaluated a point it never proposed. Both
    bounds are inclusive, so the clamped value, which is equal to the bound, is valid.
    """

    def decode(raw_log_value):
        return min(high, max(low, math.exp(raw_log_value)))

    return decode


def _engine_number(value):
    """`value` as a finite float the engine can take, or None when it cannot take it.

    set_value marshals through ctypes c_double, so a value that is not a number (None, a
    string) would raise TypeError from inside after_trial and leave the popped solution
    neither submitted nor rejected. A NaN or inf is a number the engine cannot rank, and after
    one non-finite value is submitted, every later comparison in the search is wrong. Both are
    refused here and the trial is rejected.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _converted(to_engine, value):
    """`value` as the engine's number for it, or None when the model cannot take it: a value of
    another run, such as the log of a value <= 0, or a label that is not one of the choices."""
    try:
        return _engine_number(to_engine(value))
    except (TypeError, ValueError, ArithmeticError):
        return None


def _single_value(dist):
    """The one value of a distribution whose single() is true."""
    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        return dist.choices[0]
    return dist.low


def _index_converters(dist, name):
    """Converters for a plain CategoricalDistribution, which the engine holds as a DESIGN
    variable: the engine stores the 1-based index of the choice. The index of a choice is the
    one that Optuna gives it (to_internal_repr): Optuna gives back a NaN choice as another NaN
    object (from RDB storage, or from the list of a suggest_categorical() call), and
    list.index() finds only the same object."""
    choices = dist.choices

    def to_engine(label):
        return float(dist.to_internal_repr(label)) + 1.0

    def from_engine(raw_index):
        index = int(round(raw_index)) - 1
        if not 0 <= index < len(choices):
            raise ValueError(
                f"{name!r}: engine returned index {raw_index!r}, out of range for "
                f"{len(choices)} choice(s)"
            )
        return choices[index]

    return to_engine, from_engine


def _as_direction(direction):
    if isinstance(direction, optuna.study.StudyDirection):
        return direction
    if isinstance(direction, str):
        key = direction.strip().lower()
        if key in ("minimize", "min"):
            return optuna.study.StudyDirection.MINIMIZE
        if key in ("maximize", "max"):
            return optuna.study.StudyDirection.MAXIMIZE
    raise ValueError(f"Unsupported study direction: {direction!r}")


# The engine takes a random seed from 1 to 2**31 - 1 and ignores any other value.
_ENGINE_SEED_MAX = 2**31 - 1


def _engine_seed(seed):
    """The seed to give the engine for `seed`: (seed - 1) mod (2**31 - 1) + 1. An int from 1 to
    2**31 - 1 is given as it is, 0 becomes 2**31 - 1, and every other int is mapped into that
    range too, so that every int seed has an effect. Two seeds give the same engine seed only
    when they differ by a multiple of 2**31 - 1, so the seeds of a run over range(n) all
    differ."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be an int or None, got {seed!r}")
    return (int(seed) - 1) % _ENGINE_SEED_MAX + 1


def _model_seed(value):
    """The seed that the engine applies for a RANDOMSEED line of a caller's model, or None when
    it applies none. model.set_random_seed() gives the engine the seed as it is, and the engine
    ignores a seed below 1 (describe() also shows a seed above 2**31 - 1 as one below 1)."""
    value = int(value)
    return value if 1 <= value <= _ENGINE_SEED_MAX else None


# The seed and the license key that an OptQuestSampler set on a caller's model (model=), by
# model. The engine can refuse the model after the sampler set them, when the sampler initializes
# it; a later OptQuestSampler for the same model must not take them for the caller's own.
_SET_BY_SAMPLER = weakref.WeakKeyDictionary()


@functools.lru_cache(maxsize=1)
def _spec_commands():
    return optquest.load_spec()["commands"]


# Variables whose parameters follow no rule between them, although the emitter spec lists them
# with the manifold variables: the azimuth and the elevation of a SPHERICALANGLE are two ranges.
_FREE_PARAMETER_CMDS = frozenset({"SPHERICALANGLE"})


# Function names of the expression language that the 10.2.0.0 engine refuses as the name of a
# variable and that the optquest.RESERVED_NAMES of its binding does not have. In the plain mode
# a parameter with one of them gets an internal engine name, like one with a name of that list
# (see _engine_names_for).
_UNLISTED_RESERVED_NAMES = frozenset(
    """
    betacdf betapdf cauchycdf cauchypdf chisqcdf chisqpdf correl covar expcdf exppdf gammacdf
    gammapdf interp1 interp2 klarge ksmall lognormcdf lognormpdf lookup1 lookup2 normcdf normpdf
    paretocdf paretopdf poissoncdf poissonpdf step1 tcdf tpdf uniformcdf uniformpdf weibullcdf
    weibullpdf
    """.split()
)


# The models that an OptQuestSampler initialized. The engine keeps the state of the search in
# the model, so a second OptQuestSampler for one of them would go on with the first one's search.
_SEARCHED_BY_SAMPLER = weakref.WeakSet()


# A frame of a Java stack trace, such as "\tat com.opttek.optquest.Optimization.check(X.java:9)".
_JAVA_FRAME = re.compile(r"^\s*at [\w$.<>]+\(.*\)\s*$")


def _last_error(source):
    """The last error of a model or of a solution, for a message."""
    return _variable_codec.last_error(source) or "no engine error message"


def _engine_error(model):
    """The last error of the engine without its Java stack trace, which tells a Python caller
    nothing."""
    text = _last_error(model)
    lines = [line.strip() for line in text.splitlines() if not _JAVA_FRAME.match(line)]
    text = " ".join(line for line in lines if line)
    # Nor the Java class of the exception and the engine's " :::N" trailer.
    text = text.replace("com.opttek.optquest.COptQuestException: ", "")
    return re.sub(r"\s*:::\s*-?\d+\s*$", "", text)


def _apply_license(model, license):
    """Set the license key on `model`, and raise if the engine refuses it. set_license answers
    False, and does not raise, for a key with whitespace in it and for a model that is already
    initialized; the engine would then run in demo mode with no error."""
    if model.set_license(license):
        return
    hint = ""
    if any(c.isspace() for c in license):
        hint = " A key read from a file keeps the line break at its end; strip() it."
    raise RuntimeError(f"OptQuest refused the license key: {_engine_error(model)}.{hint}")


class _FitToDistribution:
    """A from_engine decoder that moves a value a few ulps outside the bounds of a
    FloatDistribution onto the bound. The engine computes some values in binary floating point
    (a grid top such as 0 + 3 * 0.4 = 1.2000000000000002, or a point on the edge of a region
    such as 0.09999999999999998 for 0.1), and Optuna would reject such a value, sample a random
    one in its place and send that one back to the engine. A value further outside is left as
    it is, so that the warning in sample_independent reports it. A class rather than a closure,
    like the other module-level converters.

    For a parameter of a DISCRETE-type grid (`grid` is its step), the tolerance outside a bound
    is the engine's epsilon of 1e-5, and a 0.0 that is not a value of the grid is moved onto the
    value of the grid nearest to it, when that one is within 1e-5 of 0. Once it has proposed
    every value of such a grid, the engine proposes 0.0 (the entry of its table of values that it
    did not fill), which its epsilon takes for the value next to 0: a bound of the grid, or a
    value between the bounds. No other value between the bounds is moved. Next to a `max` that it
    keeps, the engine also proposes the grid value below it (0.999999 and 1 from 0 to 1 with
    step 0.333333), and each of the two is a value."""

    def __init__(self, dist, decode, grid=None):
        self._decode = decode
        self._low, self._high = dist.low, dist.high
        self._tol = 1e-9 * max(1.0, abs(dist.low), abs(dist.high))
        self._grid = None
        if grid:
            self._epsilon = _variable_codec._ENGINE_EPSILON * (1.0 + 1e-6)
            self._tol = max(self._tol, self._epsilon)
            self._grid = _variable_codec._Values(low=dist.low, top=dist.high, step=grid)

    def __call__(self, raw):
        value = self._decode(raw)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        if value == 0.0 and self._grid is not None and not self._grid.contains(0.0):
            nearest = self._grid.nearest(0.0)
            if abs(nearest) <= self._epsilon:
                value = nearest
        if self._low - self._tol <= value < self._low:
            return self._low
        if self._high < value <= self._high + self._tol:
            return self._high
        return value


class _FitToInt:
    """A from_engine decoder for an IntDistribution. The engine reports every value as a float
    (3.0), and Optuna keeps the value that sample_relative gives it in trial.params, so the
    objective would read 3.0 where Optuna's own samplers give 3. The value is given as an int.
    A value that is not within a rounding error of an int is left as it is, so that the warning
    in sample_independent reports it."""

    def __init__(self, decode):
        self._decode = decode

    def __call__(self, raw):
        value = self._decode(raw)
        if isinstance(value, float) and math.isfinite(value):
            nearest = round(value)
            if abs(value - nearest) <= 1e-9 * max(1.0, abs(value)):
                return int(nearest)
        return value


def _fit_to_distribution(dist, decode, grid=None):
    if isinstance(dist, optuna.distributions.FloatDistribution):
        return _FitToDistribution(dist, decode, grid)
    if isinstance(dist, optuna.distributions.IntDistribution):
        return _FitToInt(decode)
    return decode


def _suggest_one(trial, name, dist):
    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    if isinstance(dist, optuna.distributions.IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high, step=dist.step or 1, log=dist.log)
    if isinstance(dist, optuna.distributions.FloatDistribution):
        return trial.suggest_float(name, dist.low, dist.high, step=dist.step, log=dist.log)
    raise TypeError(f"Unsupported distribution for {name!r}: {type(dist)}")


def _length(value):
    """len(value) for a list, a tuple or an array, or None for a number or a string. A dict and
    a set have no length here: a loop over a dict gives its keys, and a set has no order, so
    neither holds the elements of an output in their places."""
    if isinstance(value, (str, bytes, Mapping, Set)):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _output_number(name, value):
    """The value of the output (or output element) `name` as a float, or ValueError that names
    it. A NaN or an infinite value is a float: the trial is rejected in the engine later."""
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"output {name!r}: expected a number, got {value!r}") from None


def _flatten_output(name, kind, shape, value):
    """Inverse of _variable_codec's grid decode: a natural (nested-list) output value -> the
    engine's flat "<name>@<key>" component dict. Shaped outputs use the same "<name>@<key>"
    convention as shaped inputs (e.g. an OUTPUTVECTOR "v" of length 3 is "v@1"/"v@2"/"v@3").
    Every level of the value must have the length of its dimension, and a value must be a
    number: anything else raises ValueError, which names the output and its shape, rather than
    sending part of the value or failing with an error that names neither."""
    if kind == "vector":
        dims, expected = (shape,), f"{shape} value(s)"
    elif kind == "matrix":
        dims, expected = tuple(shape), f"a {shape[0]}x{shape[1]} matrix"
    elif kind == "nd":
        dims = tuple(shape)
        expected = f"a {'x'.join(str(d) for d in dims)} nested list (dims {list(dims)})"
    else:
        raise AssertionError(f"unknown output kind {kind!r}")
    flat = {}

    def walk(index, remaining, item):
        if remaining:
            ok = _length(item) == remaining[0]
        else:
            ok = _length(item) is None
        if not ok:
            raise ValueError(f"output {name!r}: expected {expected}, got {value!r}")
        if not remaining:
            key = f"{name}@" + ",".join(str(i) for i in index)
            flat[key] = _output_number(key, item)
            return
        for i, element in enumerate(item, start=1):
            walk((*index, i), remaining[1:], element)

    walk((), dims, value)
    return flat


def _grid_decoder(low, high, step, as_int):
    """from_engine for a plain IntDistribution or stepped FloatDistribution whose grid has more
    steps than the engine's grid can hold (see _variable_codec.fits_engine_grid). Its engine
    variable is CONTINUOUS from `low` to `high`, and this moves the value to the nearest value of
    the grid `low` + k * `step`."""
    last = round((high - low) / step)

    def decode(raw):
        k = min(max(round((raw - low) / step), 0), last)
        value = low + k * step
        return int(value) if as_int else value

    return decode


def _registered_distribution(var_name, dist):
    """The distribution that the plain mode registers for the parameter `var_name`: `dist`,
    except for a stepped FloatDistribution, which gets the distribution that the statements=
    mode gives the same grid (see _variable_codec._stepped). For a step such as 1/3, a
    trial.suggest_float() call with the caller's `high`, which Optuna already moved down to the
    grid, would move it down one more step; suggest_all() and wrap() make that call, so they
    would lose the top of the grid. Raises ValueError for a bound that the engine cannot take."""
    numeric = isinstance(
        dist, (optuna.distributions.IntDistribution, optuna.distributions.FloatDistribution)
    )
    if numeric and not dist.log:
        # The engine refuses a bound beyond 2**50 (see _variable_codec.check_engine_bounds). A
        # log-scale parameter gets the log of its bounds, which is always within it.
        try:
            _variable_codec.check_engine_bounds(dist.low, dist.high)
        except ValueError as exc:
            raise ValueError(f"the parameter {var_name!r}: {exc}") from exc
    elif numeric:
        # A log-scale parameter is the exp of the engine's value, a float.
        try:
            float(dist.low), float(dist.high)
        except OverflowError:
            raise ValueError(
                f"the parameter {var_name!r}: a bound of a log-scale distribution "
                f"must be a number that a float can hold, got {dist.high!r}"
            ) from None
    if isinstance(dist, optuna.distributions.FloatDistribution) and dist.step is not None:
        return _variable_codec._stepped(dist.low, dist.high, dist.step)
    return dist


# The termination reasons that are a failure, and not an end of the search, when
# create_solution() gives an invalid solution: the sampler warns for them (see _stop_search).
# The engine sets Infeasible, with a COptQuestException, when the constraints of the model
# cannot be met. NotStarted and Running are no ends at all: an invalid solution with one of
# them means that the engine failed while it made the solution. A Java error inside it (an
# OutOfMemoryError, an index out of bounds) leaves the reason as it was, with the error as the
# engine's last error, and the binding raises nothing.
_ABNORMAL_TERMINATION_REASONS = {
    optquest.OQTerminationReasonException: "an internal engine exception",
    optquest.OQTerminationReasonInfeasible: "the model's constraints being infeasible",
    optquest.OQTerminationReasonNotStarted: "an error inside the engine",
    optquest.OQTerminationReasonRunning: "an error inside the engine",
}

# CannotGenerate is not a failure: a finite search space with a CHOICE, MULTICHOICE, SEQUENCE,
# MULTISTAGESEQUENCE, PARTITION, PATH, TOPOLOGICALSEQUENCE or POLICYSEQUENCE variable ends with
# it when the engine has proposed every value. It gets a warning of its own, which says that
# the space is exhausted, because the run ends before n_trials. (A finite space of other
# variables, such as INTEGER, BINARY, DESIGN, PERMUTATION, ITEMSELECTION or ASSIGNMENT, ends
# with OptimalFound, with no warning, and a space with a DAG or a TREE does not end.)
_EXHAUSTED_TERMINATION_REASON = optquest.OQTerminationReasonCannotGenerate


# The states of a trial that has ended.
_FINISHED_STATES = (
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.FAIL,
    optuna.trial.TrialState.PRUNED,
)


def _another_study_message(study, number):
    return (
        f"Trial {number} of the study {study.study_name!r} has the number of a trial that this "
        f"OptQuestSampler already finished, so the study is another one with the same name, "
        f"and an OptQuest search keeps the state of one study. Create a new OptQuestSampler "
        f"for the new study."
    )


class OptQuestSampler(optuna.samplers.BaseSampler):
    def __init__(
        self,
        model=None,
        search_space=None,
        directions=None,
        seed=None,
        license="",
        statements=None,
    ):
        engine_seed = None if seed is None else _engine_seed(seed)
        if license is None:
            license = ""
        if not isinstance(license, str):
            raise TypeError(f"license must be a str, got {type(license).__name__}")
        # What the model declares. The builder of the mode fills these, further down.
        # Optuna parameter name -> the converter of its value to the engine's value, and back.
        # The plain mode sets them where it declares each variable, the other two modes take
        # them from _variable_codec.VariableCodec (see _ingest_statement).
        self._to_engine = {}
        self._from_engine = {}
        # Parameter name -> the step of its grid, when its engine variable is of a DISCRETE type
        # (see _FitToDistribution).
        self._grid_params = {}
        # Optuna parameter name -> engine variable name, for a parameter of the plain mode whose
        # name the engine cannot take (see _engine_names_for). Every other name is the engine's.
        self._engine_names = {}
        # Statement name -> the VariableCodec of each variable of statements= and model=;
        # suggest_all() makes the natural value of a variable from its parameters with it.
        self._var_codecs = {}
        # Lowercase variable name -> its statement, and -> its fitted EXCLUSIVE statements,
        # which are built after every other statement (see _build_exclusive).
        self._declared = {}
        self._exclusive = {}
        # Variable name -> (command, its parameters that have more than one value), for each
        # variable whose parameters follow a rule between them (a DAG has no cycle, the weights
        # of a MIXTURE sum to sum_to): see _warn_partly_enqueued.
        self._ruled = {}
        # The names of the scalar outputs, and name -> ("scalar" | "vector" | "matrix" | "nd",
        # shape) for every output of statements= and model= (see _record_output). The plain
        # mode has scalar outputs only, and leaves _out_shapes empty.
        self._out_names = []
        self._out_shapes = {}
        # For each objective, in the order of the OBJECTIVE statements: the output that it
        # passes through (statements= and model= only; see _require_identity_objectives), and
        # its direction (the `directions` property, and the check of a study's directions).
        self._objective_exprs = []
        self._objective_directions = []

        # The state of the search.
        # The name of the study that this sampler searches for. The engine keeps the state of
        # one search, and the trial numbers below are unique only within one study, so the
        # sampler refuses a second study (see _check_study).
        self._study_name = None
        self._directions_validated = False
        # Trial number -> the engine's solution for the trial, until the trial ends and its
        # result goes back to the engine in it (see _finalize_trial). An entry that is left
        # after study.optimize() returns is a trial that is still running, or one of a process
        # that was killed.
        self._solutions = {}
        # Trial number -> the values that sample_relative gave Optuna for the trial. Optuna
        # discards a value that is outside the distribution of the suggest call, and the
        # sampler then warns (see sample_independent).
        self._proposed = {}
        # The numbers of the trials whose result the engine has (or that it was told failed).
        self._processed_trial_ids = set()
        # False until the first scan of sample_relative() is done: the trials that it finds are
        # the ones that were in the study before the sampler (see _finalize_trial).
        self._first_scan_done = False
        # The warnings that the sampler gives once (see _first_time).
        self._warned = set()
        # sample_relative() and after_trial() change the state above and talk to the engine,
        # and Optuna calls them from several threads under study.optimize(..., n_jobs>1). One
        # lock makes each call atomic: without it, two threads could submit one solution twice.
        self._sync_lock = threading.Lock()

        provided = {
            "model": model is not None,
            "search_space": search_space is not None,
            "statements": statements is not None,
        }
        given = [name for name, present in provided.items() if present]
        if len(given) == 0:
            raise ValueError(
                "Provide exactly one of: model, search_space (with directions), or statements."
            )
        if len(given) > 1:
            raise ValueError(
                f"Provide exactly one of: model, search_space (with directions), or "
                f"statements -- got more than one: {given}."
            )

        if search_space is not None and directions is None:
            raise ValueError("If search space is provided, directions must also be provided.")

        if search_space is not None:
            model, search_space = self._build_from_search_space(
                search_space, directions, engine_seed, license
            )
        elif statements is not None:
            model, search_space, seed = self._build_from_statements(statements, seed, license)
        elif isinstance(model, optquest.OptQuestModel):
            search_space, seed = self._attach_model(model, seed, engine_seed, license)
        else:
            raise TypeError(f"model must be an optquest.OptQuestModel, got {type(model).__name__}")

        # Each decoder also moves a value that is a rounding error outside its distribution
        # onto it (see _FitToDistribution, _FitToInt).
        for var_name, dist in search_space.items():
            self._from_engine[var_name] = _fit_to_distribution(
                dist, self._from_engine[var_name], grid=self._grid_params.get(var_name)
            )

        # The name of every declared output; wrap() checks the dict that sim() returns against
        # it. In the plain mode _out_shapes is empty, and _out_names has them all.
        self._declared_outputs = (
            list(self._out_shapes) if self._out_shapes else list(self._out_names)
        )
        # The engine's name of every output value, "<name>@<key>" for each element of a shaped
        # output: the keys of the dict that wrap() stores with a trial (see _pack_outputs).
        self._output_keys = []
        for name in self._declared_outputs:
            kind, shape = self._out_shapes.get(name, ("scalar", None))
            if kind == "scalar":
                self._output_keys.append(name)
            else:
                dims = (shape,) if kind == "vector" else tuple(shape)
                self._output_keys.extend(
                    f"{name}@" + ",".join(str(i) for i in index)
                    for index in itertools.product(*(range(1, d + 1) for d in dims))
                )
        self._output_key_set = frozenset(self._output_keys)

        self._space = search_space
        self._model = model
        _SEARCHED_BY_SAMPLER.add(model)
        # The fallback sampler gets the engine's seed, so that one seed has one meaning.
        self._rng = optuna.samplers.RandomSampler(
            seed=None if seed is None else _engine_seed(seed)
        )

    def _build_from_search_space(self, search_space, directions, engine_seed, license):
        """The plain Optuna mode: build and initialize a model with one engine variable for
        each distribution of `search_space` and one output and objective for each direction.
        Returns (the model, the distributions that the sampler registers)."""
        if not isinstance(search_space, Mapping):
            raise TypeError(
                f"search_space must be a dict of Optuna distributions, got "
                f"{type(search_space).__name__}"
            )
        if isinstance(directions, (str, optuna.study.StudyDirection)):
            directions = [directions]
        try:
            directions = [_as_direction(d) for d in directions]
        except TypeError:
            raise TypeError(
                f"directions must be a list of optuna.study.StudyDirection, got "
                f"{type(directions).__name__}"
            ) from None
        self._engine_names = self._engine_names_for(search_space, len(directions))
        # create model from search space
        model = optquest.OptQuestModel()
        _apply_license(model, license)

        def declare(added, what):
            # add_* answers False for a name or a value the engine refuses; without this
            # check the variable would be missing and the first trial would fail on a NaN.
            if not added:
                raise RuntimeError(f"OptQuest refused {what}: {_engine_error(model)}")

        # Every bound is checked before the first variable is declared.
        registered = {
            var_name: _registered_distribution(var_name, dist)
            for var_name, dist in search_space.items()
        }
        wide_grids = {}
        for var_name, dist in search_space.items():
            engine_name = self._engine_names.get(var_name, var_name)
            param = f"the parameter {var_name!r}"
            # The engine holds the value of the parameter as it is, unless a branch says more.
            self._to_engine[var_name] = self._from_engine[var_name] = _identity
            if isinstance(dist, optuna.distributions.CategoricalDistribution):
                # A DESIGN variable, which holds the 1-based index of the choice.
                declare(model.add_design_variable(engine_name, len(dist.choices)), param)
                self._to_engine[var_name], self._from_engine[var_name] = _index_converters(
                    dist, var_name
                )
            elif isinstance(dist, optuna.distributions.IntDistribution):
                # The step of an IntDistribution is 1 by default and never None. Only a larger
                # step is a grid of its own (DISCRETE); with a step of 1 the parameter is a
                # plain or a log-scale integer.
                int_step = dist.step or 1
                if not dist.log and not _variable_codec.fits_engine_grid(
                    dist.low, dist.high, int_step
                ):
                    # The engine would propose only the start of a grid with more steps than
                    # it can count (see _variable_codec.fits_engine_grid). A CONTINUOUS engine
                    # variable over the range, moved to the nearest value of the grid when it
                    # leaves the engine, reaches every value.
                    declare(model.add_continuous_variable(engine_name, dist.low, dist.high), param)
                    wide_grids[var_name] = (dist.low, dist.high, int_step, True)
                elif int_step > 1:
                    declare(
                        model.add_discrete_variable(engine_name, dist.low, dist.high, int_step),
                        param,
                    )
                elif dist.log:
                    # CONTINUOUS over log(low)..log(high), as for a log-scale FloatDistribution.
                    # An INTEGER engine variable over that range would have only as many
                    # values as there are integers in it (7 from 1 to 1000). _log_int_decoder
                    # rounds the value to an int when it leaves the engine.
                    declare(
                        model.add_continuous_variable(
                            engine_name, math.log(dist.low), math.log(dist.high)
                        ),
                        param,
                    )
                    self._to_engine[var_name] = math.log
                    self._from_engine[var_name] = _log_int_decoder(dist.low, dist.high)
                else:
                    declare(model.add_integer_variable(engine_name, dist.low, dist.high), param)
            elif isinstance(dist, optuna.distributions.FloatDistribution):
                if dist.step is not None:
                    low, high, step = dist.low, dist.high, dist.step
                    problem = _variable_codec.narrow_grid_problem(low, high)
                    if problem is not None:
                        raise ValueError(f"{param}: {problem}")
                    if not _variable_codec.fits_engine_grid(low, high, step):
                        # As for a wide IntDistribution above.
                        declare(model.add_continuous_variable(engine_name, low, high), param)
                        wide_grids[var_name] = (low, high, step, False)
                    else:
                        # Optuna already moved `high` onto its grid. The engine gets it as
                        # `max`, raised by a few ulps when the grid has two values and `high`
                        # - `low` is a few ulps short of one step in binary floating point
                        # (FloatDistribution(0.1, 0.3, step=0.2)); the engine refuses a range
                        # shorter than one step.
                        engine_max = high
                        if _variable_codec.spans_a_step(low, high, step):
                            engine_max = _variable_codec.engine_grid_max(low, high, step)
                        declare(
                            model.add_discrete_variable(engine_name, low, engine_max, step),
                            param,
                        )
                        self._grid_params[var_name] = step
                elif dist.log:
                    declare(
                        model.add_continuous_variable(
                            engine_name, math.log(dist.low), math.log(dist.high)
                        ),
                        param,
                    )
                    self._to_engine[var_name] = math.log
                    self._from_engine[var_name] = _log_float_decoder(dist.low, dist.high)
                else:
                    declare(model.add_continuous_variable(engine_name, dist.low, dist.high), param)
            else:
                raise ValueError(f"Unsupported variable type: {type(dist)}")
        for i, direction in enumerate(directions):
            # create output and objective variables
            out_name = f"OQout{i:02d}"
            self._out_names.append(out_name)
            declare(model.add_output_variable(out_name), f"the output {out_name!r}")
            self._objective_directions.append(direction)
            obj_name = f"OQobj{i:02d}"
            if direction == optuna.study.StudyDirection.MINIMIZE:
                added = model.add_minimize_objective(obj_name, out_name)
            elif direction == optuna.study.StudyDirection.MAXIMIZE:
                added = model.add_maximize_objective(obj_name, out_name)
            else:
                raise ValueError(f"Unsupported study direction: {direction}")
            declare(added, f"the objective {obj_name!r}")
        if engine_seed is not None:
            model.set_random_seed(engine_seed)
        if not model.initialize():
            raise RuntimeError(_engine_error(model))
        # The decoder of a wide grid is built last: with a number of steps beyond a float it
        # raises OverflowError, and an error of the engine above comes before that one.
        for var_name, grid in wide_grids.items():
            self._from_engine[var_name] = _grid_decoder(*grid)
        return model, registered

    def _build_from_statements(self, statements, seed, license):
        """The declarative mode: check the statements, then build and initialize a model from
        them. Every command goes through _variable_codec, so this class has no code for one
        variable type. Returns (the model, the search space, the seed), where the seed is the
        one of a RANDOMSEED statement when `seed` is None."""
        # The statements are checked as optquest checks a model, before anything is built: a
        # misspelled or missing field, a value of the wrong type, a name that the engine
        # refuses, bounds in the wrong order, or a group member or an EXCLUSIVE variable that
        # no earlier statement declares raises optquest.EmitError (a ValueError) that names the
        # statement. The engine checks the names in an expression when it initializes the
        # model. numpy values become Python lists and numbers first, and each field then gets
        # the type that the engine's methods take (emit_model() accepts 30.0 or "30" for an int
        # field).
        if isinstance(statements, (Mapping, str, bytes)) or not isinstance(statements, Iterable):
            raise TypeError(
                f"statements must be a list of statement dicts, got {type(statements).__name__}"
            )
        commands = _spec_commands()
        statements = [_variable_codec.plain_value(stmt) for stmt in statements]
        statements = [_variable_codec.coerce_text(stmt, commands) for stmt in statements]
        try:
            optquest.emit_model(statements)
        except OverflowError as exc:
            # emit_model() converts each number to a float, and an int beyond the range of
            # a float raises an OverflowError that names no statement.
            for stmt in statements:
                try:
                    optquest.emit_statement(stmt)
                except OverflowError:
                    raise ValueError(
                        f"{str(stmt.get('cmd')).upper()} {stmt.get('name')!r}: a number is "
                        f"too large for the engine ({exc}); the engine takes a bound from "
                        f"-2**50 to 2**50."
                    ) from exc
            raise
        optquest.check_model(statements)
        statements = [_variable_codec.coerce_fields(stmt, commands) for stmt in statements]
        # emit_model() checks the order and the range of numbers only when they are numbers, so
        # the fields that were text ("5" and "2" for a `min` and a `max`) are checked now.
        optquest.emit_model(statements)
        optquest.check_model(statements)
        # The engine gives back the value of a name outside ASCII as NaN (see
        # _variable_codec.check_ascii_names).
        for stmt in statements:
            _variable_codec.check_ascii_names(stmt, commands)
        # A LICENSE or RANDOMSEED statement (a describe() round trip has them) applies when
        # the license=/seed= argument is not given, and raises when it differs from the
        # argument or from another statement, rather than being ignored. An empty key is no
        # key, as on a model (see model= below), and two seeds agree when they give the
        # engine the same seed (see _engine_seed).
        license_from = "the license= argument" if license else None
        seed_from = "the seed= argument" if seed is not None else None
        for stmt in statements:
            cmd = str(stmt["cmd"]).upper()
            if cmd == "LICENSE" and str(stmt["license_id"]):
                key = str(stmt["license_id"])
                if license_from is not None and key != license:
                    raise ValueError(
                        f"a LICENSE statement and {license_from} give different keys; "
                        f"give the key once."
                    )
                if license_from is None:
                    license, license_from = key, "another LICENSE statement"
            elif cmd == "RANDOMSEED":
                value = int(stmt["seed"])
                if seed_from is not None and _engine_seed(value) != _engine_seed(seed):
                    raise ValueError(
                        f"a RANDOMSEED statement ({value!r}) and {seed_from} ({seed!r}) "
                        f"differ; give the seed once."
                    )
                if seed_from is None:
                    seed, seed_from = value, "another RANDOMSEED statement"
        engine_seed = None if seed is None else _engine_seed(seed)
        model = optquest.OptQuestModel()
        _apply_license(model, license)
        if engine_seed is not None:
            model.set_random_seed(engine_seed)
        search_space = {}
        for stmt in statements:
            self._ingest_statement(stmt, search_space, build_on=model)
        self._build_exclusive(model)
        self._require_identity_objectives()
        if not model.initialize():
            raise RuntimeError(_engine_error(model))
        return model, search_space, seed

    def _attach_model(self, model, seed, engine_seed, license):
        """The typed mode: read the search space of the caller's `model`, refuse what the
        sampler cannot take, apply `seed` and `license`, and initialize it. Returns (the
        search space, the seed), where the seed is the one of the model when `seed` is None."""
        # A built model keeps no statement list, so the sampler reads one from its describe()
        # text with optquest.parse_model(), and takes the statements through _ingest_statement
        # as the declarative mode does (build_on=None: the model already holds them).
        # What the sampler cannot take is refused before the model is changed or initialized,
        # because initialize() locks the caller's model: the Adaptive Sampling configuration
        # (the engine would explore a response surface, and the run would end after one
        # trial), NOGENERATE, what the codec cannot register, and an objective that is not an
        # output. Random variables and run settings of the model are the engine's to apply;
        # random variables are not in the search space (see _variable_codec.classify).
        if model in _SEARCHED_BY_SAMPLER:
            raise RuntimeError(
                "This model is already initialized by an earlier OptQuestSampler, and the "
                "engine keeps the state of that search in the model, so a new "
                "OptQuestSampler would go on with it. Build a new model for the new "
                "OptQuestSampler."
            )
        statements = optquest.parse_model(model.describe())
        commands = _spec_commands()
        for stmt in statements:
            _variable_codec.check_ingestible(stmt["cmd"])
            _variable_codec.check_ascii_names(stmt, commands)
        search_space = {}
        for stmt in statements:
            self._ingest_statement(stmt, search_space, build_on=None)
        self._build_exclusive(None)
        self._require_identity_objectives()
        # A license key or a seed that the caller set on the model applies when the
        # license=/seed= argument is not given, and the sampler raises when the argument
        # would give the engine another one, as for a LICENSE or RANDOMSEED statement in
        # statements=, rather than replacing it with no error. A seed that the engine
        # ignores is no seed (see _model_seed). A key or a seed that an earlier
        # OptQuestSampler set on this model, in a construction that failed after it set
        # them, is not the caller's (see _SET_BY_SAMPLER).
        set_before = _SET_BY_SAMPLER.get(model, {})
        model_license = model_seed = earlier_license = None
        current_license = current_seed = None
        for stmt in statements:
            cmd = str(stmt["cmd"]).upper()
            if cmd == "LICENSE" and str(stmt["license_id"]):
                current_license = str(stmt["license_id"])
                if current_license == set_before.get("license"):
                    earlier_license = current_license
                else:
                    model_license = current_license
            elif cmd == "RANDOMSEED":
                current_seed = _model_seed(stmt["seed"])
                if current_seed is not None and current_seed != set_before.get("seed"):
                    model_seed = current_seed
        if license and model_license not in (None, license):
            raise ValueError(
                "the license key set on the model and the license= argument differ; "
                "give the key once."
            )
        if engine_seed is not None and model_seed not in (None, engine_seed):
            raise ValueError(
                f"the seed set on the model ({model_seed!r}) and the seed= argument "
                f"({seed!r}) differ; give the seed once."
            )
        # A key or a seed that the model already has is not set again (the engine refuses to
        # set either on a model that the caller already initialized, even an equal one), and
        # only a value that this construction sets is recorded as the sampler's.
        if (license or earlier_license is not None) and (license or None) != current_license:
            # An empty license= leaves the key that the caller set on the model, and takes
            # away one that an earlier OptQuestSampler set.
            _apply_license(model, license)
            _SET_BY_SAMPLER.setdefault(model, {})["license"] = license or None
        if engine_seed is not None and engine_seed != current_seed:
            if not model.set_random_seed(engine_seed):
                # set_random_seed returns False on a model that the caller already
                # initialized (ERROR already initialized). The seed would not apply, and
                # there would be no error.
                raise RuntimeError(f"seed={seed!r} was not applied: {_engine_error(model)}")
            _SET_BY_SAMPLER.setdefault(model, {})["seed"] = engine_seed
        if not model.initialize():
            raise RuntimeError(_engine_error(model))
        if seed is None:
            # The seed that the engine applies, from the model, seeds the fallback sampler
            # too, so that one seed has one meaning.
            for stmt in statements:
                cmd = str(stmt["cmd"]).upper()
                if cmd == "RANDOMSEED" and _model_seed(stmt["seed"]) is not None:
                    seed = int(stmt["seed"])
        return search_space, seed

    @property
    def directions(self):
        """This sampler's declared objective directions, in order. Pass to
        optuna.create_study(directions=sampler.directions, ...) instead of hand-duplicating the
        direction list. On first use, infer_relative_search_space() compares the study's
        directions against these and raises if they don't match."""
        return list(self._objective_directions)

    @staticmethod
    def _engine_names_for(search_space, n_objectives):
        """Optuna parameter name -> engine variable name for the plain search_space= mode, for
        the names that the engine cannot take: an empty name, a name with whitespace or a
        semicolon, one that starts with a digit or holds a non-ASCII character, a function or
        constant name of the expression language (such as `e`, `max` or `rate`), a name that
        differs only by case from an earlier one (the engine compares names without regard to
        case), and the name of one of the sampler's own outputs and objectives (OQout00,
        OQobj00, ...). Each of these gets an internal name; trial.params keeps the Optuna
        name."""
        internal = {f"oqout{i:02d}" for i in range(n_objectives)}
        internal |= {f"oqobj{i:02d}" for i in range(n_objectives)}
        taken = internal | {name.lower() for name in search_space if isinstance(name, str)}
        mapping = {}
        used = set()
        counter = 0
        for name in search_space:
            usable = (
                isinstance(name, str)
                and name != ""
                and name.isascii()
                and not name[0].isdigit()
                and not any(c.isspace() or c == ";" for c in name)
                and name.lower() not in optquest.RESERVED_NAMES
                and name.lower() not in _UNLISTED_RESERVED_NAMES
                and name.lower() not in internal
                and name.lower() not in used
            )
            if usable:
                used.add(name.lower())
                continue
            while f"oqparam{counter:03d}" in taken:
                counter += 1
            engine_name = f"OQparam{counter:03d}"
            taken.add(engine_name.lower())
            used.add(engine_name.lower())
            mapping[name] = engine_name
        return mapping

    def _ingest_statement(self, stmt, search_space, *, build_on):
        """Classify one emitter statement and update self.*/search_space accordingly. Shared by
        the declarative (statements=) and typed-model (model=) construction modes, which differ
        only in whether the model still needs to be built:
          - build_on=<live OptQuestModel>: also apply the statement to it (declarative mode,
            where the caller's statements are the only description of the model that exists).
          - build_on=None: the model already contains this statement (typed mode, where the
            statement was recovered from the model's own describe() text), so there is
            nothing to apply."""
        cmd = stmt["cmd"].upper()
        if cmd in ("LICENSE", "RANDOMSEED"):
            # statements=: applied with the license=/seed= arguments in __init__. model=: the
            # caller already applied it to the model.
            return
        if cmd == "EXCLUSIVE":
            # The edges of an excluded range must be values of its variable, because the engine
            # proposes them; see _variable_codec.fit_exclusive_range. The ranges of a variable
            # are built when the sampler has them all (see _build_exclusive).
            fitted = _variable_codec.fit_exclusive_range(
                stmt, self._declared, engine_built=build_on is None
            )
            self._exclusive.setdefault(str(stmt["name"]).lower(), []).append(fitted)
            return
        else:
            # The engine and Optuna must agree on the top of a DISCRETE-type grid; see
            # _variable_codec.align_stepped_max.
            engine_stmt, stmt = _variable_codec.align_stepped_max(
                stmt, engine_built=build_on is None
            )
        if build_on is not None:
            _variable_codec.build_model_statement(build_on, engine_stmt)
        kind = _variable_codec.classify(cmd)
        if cmd == "OBJECTIVE":
            self._objective_exprs.append(stmt["expression"])
            self._objective_directions.append(_as_direction(stmt["direction"]))
        elif kind == "variable":
            vc = _variable_codec.build(stmt)
            search_space.update(vc.params)
            category = (_spec_commands().get(cmd) or {}).get("cat")
            if category in ("manifold", "smooth") and cmd not in _FREE_PARAMETER_CMDS:
                ruled = frozenset(k for k, d in vc.params.items() if not d.single())
                if len(ruled) > 1:
                    self._ruled[stmt["name"]] = (cmd, ruled)
            if cmd in _variable_codec.STEPPED_CMDS:
                self._grid_params.update(dict.fromkeys(vc.params, float(stmt["step"])))
            self._to_engine.update(vc.to_engine)
            self._from_engine.update(vc.from_engine)
            self._var_codecs[stmt["name"]] = vc
            self._declared.setdefault(str(stmt["name"]).lower(), stmt)
        elif kind == "output":
            self._record_output(stmt)
        # "skip"/other non-objective "other" commands (session/run knobs, random
        # variables, EXCLUSIVE/FIXGROUP/FLOATGROUP/EXCHANGE, EXPRESSION/CONSTRAINT, ...):
        # already applied to the model (build_on branch above, when applicable); nothing
        # further needed for the Optuna-facing side.

    def _build_exclusive(self, build_on):
        """Build the excluded ranges that _ingest_statement recorded, on `build_on` (None for
        a model that already holds them). The engine can propose a value of one range of a
        variable when two of its ranges overlap, so overlapping ranges of statements= are built
        as one range, and the sampler refuses them on a model (see
        _variable_codec.merge_exclusive_ranges)."""
        for fitted in self._exclusive.values():
            merged = _variable_codec.merge_exclusive_ranges(
                fitted, self._declared, engine_built=build_on is None
            )
            if build_on is not None:
                for stmt in merged:
                    _variable_codec.build_model_statement(build_on, stmt)

    def _first_time(self, key):
        """Whether the warning `key`, which the sampler gives once, has not been given yet;
        it then counts as given. The caller holds self._sync_lock."""
        if key in self._warned:
            return False
        self._warned.add(key)
        return True

    def _warn_partly_enqueued(self, number, fixed):
        """Warn once for each sampler when an enqueued trial does not fix every parameter of a
        variable whose parameters follow a rule. Optuna's RandomSampler draws each of the others
        on its own, so the variable's value need not follow the rule. Caller must hold
        self._sync_lock."""
        if "partly enqueued" in self._warned:
            return
        for name, (cmd, params) in self._ruled.items():
            missing = sorted(params - set(fixed))
            if missing:
                self._warned.add("partly enqueued")
                warnings.warn(
                    f"OptQuestSampler: trial {number} is enqueued without every parameter of "
                    f"{name!r} ({cmd}; {missing[0]!r} is one of the others). Optuna's "
                    f"RandomSampler draws each missing parameter on its own, so the value of "
                    f"{name!r} need not follow the rules of {cmd} (a DAG can have a cycle, the "
                    f"weights of a MIXTURE need not sum to sum_to). Enqueue every parameter of "
                    f"such a variable. The sampler warns once.",
                    stacklevel=1,
                )
                return

    def _warn_partly_fixed(self, number, search_space):
        """Warn once for each sampler when the search space that Optuna passes for a trial holds
        some, and not all, of the parameters of a variable whose parameters follow a rule. A
        wrapper such as optuna.samplers.PartialFixedSampler takes the parameters that it fixes
        out of the search space, and its value replaces the engine's for those parameters only,
        so the variable's value need not follow the rule. Caller must hold self._sync_lock."""
        if "partly fixed" in self._warned or len(search_space) >= len(self._space):
            return
        for name, (cmd, params) in self._ruled.items():
            missing = sorted(param for param in params if param not in search_space)
            if missing and len(missing) < len(params):
                self._warned.add("partly fixed")
                warnings.warn(
                    f"OptQuestSampler: the search space of trial {number} holds some of the "
                    f"parameters of {name!r} ({cmd}) and not {missing[0]!r}: a wrapper such as "
                    f"optuna.samplers.PartialFixedSampler fixes it. The fixed value replaces "
                    f"the engine's value for that parameter only, so the value of {name!r} "
                    f"need not follow the rules of {cmd} (a PERMUTATION can repeat a rank, the "
                    f"weights of a MIXTURE need not sum to sum_to). Fix every parameter of "
                    f"such a variable, or none. The sampler warns once.",
                    stacklevel=1,
                )
                return

    def _is_proposed_choice(self, param_name, value, proposed):
        """Whether `value` is the choice that the engine proposed for the categorical parameter
        `param_name` (`proposed` holds the values that sample_relative gave Optuna for the
        trial). The engine holds the 1-based index of a choice, and two choices can be equal: a
        choice given twice in a CategoricalDistribution, which Optuna takes, or two equal rows
        of a TUPLE. The index of the label is then the index of the first of them, which need
        not be the one that the engine proposed, so the label is not written back over it."""
        if param_name not in proposed or not isinstance(
            self._space.get(param_name), optuna.distributions.CategoricalDistribution
        ):
            return False
        was = proposed[param_name]
        return bool(was == value) or (was != was and value != value)  # a NaN choice too

    def _warn_unregistered(self, param_name):
        """Warn once per parameter name that the objective suggests outside this sampler's
        search space. Caller must hold self._sync_lock."""
        if not self._first_time(("unregistered", param_name)):
            return
        warnings.warn(
            f"OptQuestSampler: parameter {param_name!r} is not in this sampler's search "
            f"space, so OptQuest does not model it; its values are sampled independently "
            f"and are not sent to the engine. Declare it in search_space/statements/model "
            f"to let OptQuest search it.",
            stacklevel=4,  # _finalize_trial, three calls up
        )

    def _record_output(self, stmt):
        cmd = stmt["cmd"].upper()
        name = stmt["name"]
        if cmd == "OUTPUT":
            self._out_names.append(name)
            self._out_shapes[name] = ("scalar", None)
        elif cmd == "OUTPUTVECTOR":
            self._out_shapes[name] = ("vector", int(stmt["n"]))
        elif cmd == "OUTPUTMATRIX":
            self._out_shapes[name] = ("matrix", (int(stmt["rows"]), int(stmt["cols"])))
        elif cmd == "OUTPUTNDVARIABLE":
            self._out_shapes[name] = ("nd", tuple(int(d) for d in stmt["dims"]))

    def _require_identity_objectives(self):
        """Require that each objective passes one declared output through. When a trial ends,
        each value that objective(trial) returned goes to the engine as the output of its
        objective (see _set_outputs), and a compound expression ("out00 + out01") has no one
        output to take the value. So this is checked when the sampler is created, before it
        initializes the model. The engine compares names without regard to case, so this does
        too, and each objective is then mapped to the output name as it was declared."""
        if not self._objective_exprs:
            return
        outputs = {name.lower(): name for name in self._out_names}
        unknown = [
            str(expr).strip()
            for expr in self._objective_exprs
            if str(expr).strip().lower() not in outputs
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(expr).strip())
        ]
        if unknown:
            # One name that is not an output: a misspelled output, or an input or expression
            # name, which has no output to take the value that objective(trial) returns.
            raise ValueError(
                f"the objective expression(s) {unknown!r} name no declared output; the "
                f"declared outputs are {self._out_names!r}. Each objective of OptQuestSampler "
                f"passes one declared output through: check the spelling, or send the value "
                f"through an output of its own."
            )
        non_identity = [
            expr for expr in self._objective_exprs if str(expr).strip().lower() not in outputs
        ]
        if non_identity:
            raise NotImplementedError(
                "OptQuestSampler's objective(trial) contract only supports objectives that "
                "are an identity pass-through of exactly one declared output (e.g. "
                "add_minimize_objective(name, 'out00')). Found compound/non-identity "
                f"objective expression(s): {non_identity!r}. Route each composite value "
                "through its own dedicated output instead."
            )
        self._objective_exprs = [
            outputs[str(expr).strip().lower()] for expr in self._objective_exprs
        ]

    def _check_study(self, study):
        """Refuse a second study. The engine keeps the state of one search, and the sampler's
        bookkeeping (_processed_trial_ids, _solutions, _proposed) is keyed by trial number,
        which is unique only within one study: the results of a second study would never reach
        the engine, with no error."""
        name = study.study_name
        with self._sync_lock:
            if self._study_name is None:
                self._study_name = name
        if name != self._study_name:
            raise RuntimeError(
                f"This OptQuestSampler searches for the study {self._study_name!r}, and an "
                f"OptQuest search keeps the state of one study. Create a new OptQuestSampler "
                f"for the study {name!r}."
            )

    def _direction_problem(self, study):
        """The error for a study whose directions are not the ones of this sampler's objectives,
        or None. study.directions does not change, so the study that the sampler searches for is
        checked once."""
        if not self._objective_directions:
            return None
        if self._directions_validated and study.study_name == self._study_name:
            return None
        directions = study.directions
        if len(directions) != len(self._objective_directions):
            return (
                f"study has {len(directions)} direction(s) but this sampler declares "
                f"{len(self._objective_directions)} objective(s); pass "
                f"directions=sampler.directions to optuna.create_study(...) instead of "
                f"hand-listing them, or fix the mismatch."
            )
        for i, (declared, actual) in enumerate(zip(self._objective_directions, directions)):
            if declared != actual:
                return (
                    f"study direction[{i}] is {actual.name} but this sampler's model "
                    f"declares {declared.name} for that objective; pass "
                    f"directions=sampler.directions to optuna.create_study(...) instead of "
                    f"hand-listing them, or fix the mismatch. If the objective was written "
                    f"for the 10.1 sampler, also return its values in the order of the "
                    f"OBJECTIVE statements, not the order of the outputs."
                )
        return None

    def infer_relative_search_space(self, study, trial):
        # The directions come first: a study that the sampler refuses must not become the study
        # that it searches for (_check_study), or the study that the error asks for would be
        # refused as a second study.
        problem = self._direction_problem(study)
        if problem is not None:
            raise ValueError(problem)
        self._check_study(study)
        self._directions_validated = True
        # A trial of the study with a number that this sampler already finished belongs to
        # another study of the same name (a study created again in a notebook, in a new
        # storage): _check_study cannot tell the two apart by name, and the sampler would drop
        # every result of the second study. Only finished trials have their number in
        # _processed_trial_ids, so a running trial of the same study never has.
        with self._sync_lock:
            finished = trial.number in self._processed_trial_ids
        if finished:
            raise RuntimeError(_another_study_message(study, trial.number))
        # A copy: a wrapper such as optuna.samplers.PartialFixedSampler removes the fixed
        # parameters from the dict it gets.
        return dict(self._space)

    def suggest_all(self, trial):
        """Make every trial.suggest_*() call of the search space, and return one natural
        Python value for each declared variable: a number for a scalar, a list for a vector or
        an ASSIGNMENT, a list of rows for a matrix, a dict keyed by item name for a PERMUTATION,
        SEQUENCE or MIXTURE, and a dict keyed by (from, to) for a DAG (see
        _variable_codec.VariableCodec for every command). A DAG of 8 nodes has 64 parameters,
        named "<name>@<from>,<to>": index the result by the names that you declared instead.
        wrap() calls this for you.

        In the plain mode every parameter is a value of its own, and the result is keyed by
        parameter name."""
        raw = {name: _suggest_one(trial, name, dist) for name, dist in self._space.items()}
        if not self._var_codecs:
            return raw
        covered = set()
        result = {}
        for stmt_name, vc in self._var_codecs.items():
            sub = {key: raw[key] for key in vc.params}
            result[stmt_name] = vc.decode(sub)
            covered.update(vc.params)
        for name, value in raw.items():
            if name not in covered:
                result[name] = value
        return result

    def wrap(self, sim):
        """Return a standard objective(trial) callable for study.optimize(sampler.wrap(sim)).

        `sim` receives the dict suggest_all() produces (natural values keyed by variable name)
        and must return:
          - with model=/statements= construction: a dict keyed by every declared output
            name (see self.directions/self._declared_outputs), or None to reject the trial.
            Shaped outputs (declared via OUTPUTVECTOR/OUTPUTMATRIX/OUTPUTNDVARIABLE) take a
            list/nested-list matching their declared shape.
          - with search_space=/directions= construction: a plain tuple/list of objective values
            in directions order (there are no output names to key a dict by in this mode).
        A raised exception from `sim` is not caught here. It propagates the same way it would
        from a hand-written objective(trial), which optuna.study.optimize() turns into a FAILed
        trial (or re-raises, depending on `catch=`)."""

        def objective(trial):
            params = self.suggest_all(trial)
            result = sim(params)
            if result is None:
                raise optuna.TrialPruned()
            if self._objective_exprs:
                return self._pack_outputs(trial, result)
            if not isinstance(result, (tuple, list)):
                raise TypeError(
                    f"With search_space=/directions= construction, sim(params) must return a "
                    f"tuple/list of objective values in directions order (there are no output "
                    f"names to key a dict by in this mode); got {type(result)}."
                )
            return tuple(result)

        return objective

    def _pack_outputs(self, trial, result):
        """Check the dict that sim() returned in wrap(), store one number for each declared
        output value with the trial, and return the objective values."""
        if not isinstance(result, dict):
            raise TypeError(
                f"With model=/statements= construction, sim(params) must return a dict "
                f"keyed by the declared outputs {self._declared_outputs!r}, or None to reject "
                f"the trial; got {type(result)}."
            )
        missing = [n for n in self._declared_outputs if n not in result]
        extra = [k for k in result if k not in self._declared_outputs]
        if missing or extra:
            raise OptQuestOutputMismatchError(
                f"trial {trial.number}: output mismatch -- missing={missing} extra={extra}; "
                f"sim(params) must return exactly {self._declared_outputs!r}."
            )
        flat = {}
        for name in self._declared_outputs:
            kind, shape = self._out_shapes.get(name, ("scalar", None))
            if kind == "scalar":
                flat[name] = _output_number(name, result[name])
            else:
                flat.update(_flatten_output(name, kind, shape, result[name]))
        # The flat {name: float} dict, and not what sim() returned: user_attrs must be JSON for
        # RDB and Journal storage, and _set_outputs gives the engine these values as they are.
        trial.set_user_attr("_optquest_outputs", flat)
        # Each objective passes one scalar output through (see _require_identity_objectives).
        return tuple(flat[expr] for expr in self._objective_exprs)

    def _finalize_trial(self, t, state, values, *, is_attach_scan, own_trial):
        """Send the result of the finished trial `t` to the engine, or reject its solution.

        after_trial() calls this for each trial of this run when the trial ends. The scan of
        sample_relative() calls it for a trial that is already in the study when the sampler
        first sees the study (`is_attach_scan`), or that study.add_trial() added during the run.
        The caller holds self._sync_lock and has added t.number to self._processed_trial_ids,
        so a trial is finalized once.

        `state` and `values` are arguments, and are not read from `t`: after_trial() gets a
        FrozenTrial from before Optuna stored the result, whose state is still RUNNING.

        `own_trial` is True from after_trial(): a trial of this run, which the sampler can stop
        with an error that names it. A trial that the scan finds only gets a warning, because an
        error there would fail the trial that asked for a point.

        Each warning of this method and of its helpers has the stacklevel of the caller of this
        method."""
        # Popped, so that the solution and the engine handle that it owns are released now.
        sol = self._solutions.pop(t.number, None)
        proposed = self._proposed.pop(t.number, None) or {}
        if state == optuna.trial.TrialState.COMPLETE:
            self._send_result(t, state, values, sol, proposed, is_attach_scan, own_trial)
        elif state in (optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED):
            self._send_rejection(t, state, sol, is_attach_scan)

    def _send_result(self, t, state, values, sol, proposed, is_attach_scan, own_trial):
        """Give the engine the result of the COMPLETE trial `t`: in `sol`, the solution that the
        engine proposed for it, or as an evaluated point when it has none (an enqueued trial, or
        a trial that the sampler did not run). `proposed` holds the values that sample_relative
        gave Optuna for the trial."""
        external = sol is None
        if external:
            sol = self._model.get_empty_solution()
            if sol is None:
                # Optuna has the trial as COMPLETE, and the engine gave no solution to record
                # it in. The engine's last error is a hint only: it can be of an earlier call.
                warnings.warn(
                    f"OptQuestSampler: trial {t.number} completed but OptQuest could "
                    f"not provide an empty solution to record it in "
                    f"({_last_error(self._model)}); "
                    f"its result was not recorded by the engine.",
                    stacklevel=3,
                )
                return
        unsendable, missing = self._set_inputs(sol, t, external, proposed)
        problem, missing_outputs = self._set_outputs(sol, t, state, values, external, unsendable)
        if problem is not None or unsendable or missing or missing_outputs:
            # Reject, and do not submit, a point whose record in the engine would be
            # incomplete: a value that is not a number cannot be marshalled, a NaN or inf
            # cannot be ranked, and an output or an input with no value would be NaN or 0.0. A
            # rejection tells the engine that this trial happened and that its result could not
            # be sent. An external point has no engine solution to reject, so it is not sent.
            # The engine is told before the warning or the error, so that the call is made in
            # every case.
            ok = True if external else sol.reject()
            self._report_incomplete(
                t,
                problem,
                unsendable,
                missing,
                missing_outputs,
                external,
                is_attach_scan,
                own_trial,
            )
        elif sol.get_iteration() == 0:
            # A solution that the sampler created goes back as an evaluated point.
            ok = sol.add_evaluated()
        else:
            ok = sol.submit()
        if not ok:
            # submit(), add_evaluated() and reject() answer False, and do not raise, when the
            # engine refuses. Without this check the trial would be COMPLETE in Optuna and
            # missing in the engine, with no sign. A warning and not an error: an error would
            # fail the trial that is being sampled or finalized, which is not always trial t.
            warnings.warn(
                f"OptQuestSampler: trial {t.number}'s solution failed to reach "
                f"OptQuest ({_last_error(sol)}); "
                f"its result was not recorded by the engine.",
                stacklevel=3,
            )

    def _set_inputs(self, sol, t, external, proposed):
        """Set the parameters of trial `t` on `sol`. Returns (the (name, value) pairs that the
        engine cannot take, the parameters of an external point that have no value).

        The return of sol.set_value() is not checked: the engine answers True for every name,
        also for one that the model does not have."""
        unsendable = []
        for param_name, param_value in t.params.items():
            to_engine = self._to_engine.get(param_name)
            if to_engine is None:
                # A parameter that the objective suggested outside the search space. The model
                # has no variable for it, and the engine would refuse the whole solution (ERROR
                # BAD VALUE NAME), so it is left out, with one warning for each name.
                self._warn_unregistered(param_name)
                continue
            if not external and self._is_proposed_choice(param_name, param_value, proposed):
                # The engine's solution holds the index that the engine proposed.
                continue
            engine_value = _converted(to_engine, param_value)
            if engine_value is None:
                unsendable.append((param_name, param_value))
                continue
            sol.set_value(self._engine_names.get(param_name, param_name), engine_value)
        # An external point needs a value for every variable. The engine takes an input with no
        # value as 0.0, even outside the variable's bounds, and the search would then use a
        # point that no one evaluated. (A solution from the engine already holds the engine's
        # value for a parameter that the objective did not suggest.) A parameter whose
        # distribution has one value (the diagonal of a DAG, an INTEGER with min == max) gets
        # that value: a hand-written objective need not suggest it.
        missing = []
        for name in self._to_engine if external else ():
            if name in t.params:
                continue
            dist = self._space.get(name)
            engine_value = None
            if dist is not None and dist.single():
                engine_value = _converted(self._to_engine[name], _single_value(dist))
            if engine_value is None:
                missing.append(name)
            else:
                sol.set_value(self._engine_names.get(name, name), engine_value)
        return unsendable, missing

    def _set_outputs(self, sol, t, state, values, external, unsendable):
        """Set the outputs of trial `t` on `sol`, and add each value that the engine cannot take
        to `unsendable`. Returns (what makes the result unusable, or None; the outputs of an
        external point that have no value)."""
        problem = None
        missing_outputs = []
        flat_outputs = t.user_attrs.get("_optquest_outputs")
        if flat_outputs is not None:
            # The trial went through wrap(): _pack_outputs checked what sim() returned and
            # stored one value for each declared output.
            if external:
                # A trial of a run with another model (a resumed study) has the outputs of that
                # model, and the engine would take an output of this model that the trial does
                # not have as 0.0.
                missing_outputs = [k for k in self._output_keys if k not in flat_outputs]
            for out_key, value in flat_outputs.items():
                if out_key not in self._output_key_set:
                    # An output that this model does not declare (a trial of a run with another
                    # model): the engine refuses a solution with a name that it does not know
                    # (ERROR BAD VALUE NAME).
                    continue
                engine_value = _engine_number(value)
                if engine_value is None:
                    unsendable.append((out_key, value))
                    continue
                sol.set_value(out_key, engine_value)
            return problem, missing_outputs
        # A hand-written objective(trial): each value that it returned goes to the output that
        # its objective passes through (self._objective_exprs, checked in __init__). Outputs and
        # objectives need not be declared in the same order, and an output that only a
        # constraint uses has no objective. In the plain mode self._objective_exprs is empty,
        # and self._out_names has one output for each direction.
        target_names = self._objective_exprs or self._out_names
        unlinked = [n for n in self._declared_outputs if n not in target_names]
        if unlinked:
            # The engine would hold NaN for such an output and mark every trial infeasible.
            problem = (
                f"the model declares output(s) {unlinked!r} that no objective "
                f"passes through (e.g. an output used only by a CONSTRAINT), and a "
                f"hand-written objective(trial) returns objective values only, so "
                f"the engine would have no value for them and would mark the trial "
                f"infeasible. Use sampler.wrap(sim), with sim(params) returning a "
                f"dict keyed by every declared output, {self._declared_outputs!r}"
            )
        elif values and len(values) != len(target_names):
            problem = (
                f"the objective returned {len(values)} value(s) but the model "
                f"declares {len(target_names)} objective(s) {target_names!r}"
            )
        elif values:
            sent = {}
            for name, obj_value in zip(target_names, values):
                engine_value = _engine_number(obj_value)
                if engine_value is None:
                    unsendable.append((name, obj_value))
                elif sent.setdefault(name, engine_value) != engine_value:
                    # Two objectives pass this output through, and the engine holds one value
                    # for each output.
                    problem = (
                        f"two objectives pass the same output, {name!r}, through, "
                        f"and the engine holds one value for each output, but the "
                        f"objective returned different values for them "
                        f"({list(values)!r}); return one value for both, or give "
                        f"each objective an output of its own"
                    )
                else:
                    sol.set_value(name, engine_value)
        else:
            # Optuna reported a COMPLETE trial with no objective value, so there is nothing to
            # give the engine. The outputs stay at the engine's defaults, and the trial is
            # still submitted.
            warnings.warn(
                f"trial {t.number} reached OptQuest write-back in state "
                f"{state!r} with no objective values ({values!r}); its outputs "
                f"stay at the engine's defaults, so the engine's view of this "
                f"trial is not the trial's real result.",
                RuntimeWarning,
                stacklevel=4,
            )
        return problem, missing_outputs

    def _report_incomplete(
        self, t, problem, unsendable, missing, missing_outputs, external, is_attach_scan, own_trial
    ):
        """Raise or warn for the trial `t`, whose result the engine did not get whole."""
        action = "did not send it to the engine" if external else "rejected it"
        if problem is not None:
            if own_trial:
                # The objective of this run cannot give this model its outputs, so every trial
                # would be rejected and the engine would search with no result. Stop at the
                # first trial instead. Optuna has already recorded the trial when after_trial()
                # raises.
                raise ValueError(f"OptQuestSampler: trial {t.number}: {problem}.")
            if self._first_time("unlinked output"):
                where = (
                    "in the study before this sampler"
                    if is_attach_scan
                    else "added to the study during this run"
                )
                warnings.warn(
                    f"OptQuestSampler: trial {t.number} ({where}): {problem}. The "
                    f"sampler {action}; it does the same for every other such trial "
                    f"without another warning.",
                    stacklevel=4,
                )
        elif missing or missing_outputs:
            if self._first_time("incomplete trial"):
                absent = []
                if missing:
                    absent.append(f"the parameter(s) {missing!r} of this sampler's search space")
                if missing_outputs:
                    absent.append(f"the output(s) {missing_outputs!r} of this sampler's model")
                warnings.warn(
                    f"OptQuestSampler: trial {t.number} has no value for "
                    f"{' or '.join(absent)} (a trial added with study.add_trial(), "
                    f"or one from a run with another search space or model). The "
                    f"engine needs a value for every variable and output, so the "
                    f"sampler {action}; it does the same for every other such trial "
                    f"without another warning. Optuna still records the trial "
                    f"normally.",
                    stacklevel=4,
                )
        else:
            detail = ", ".join(f"{n}={v!r}" for n, v in unsendable)
            warnings.warn(
                f"OptQuestSampler: trial {t.number} carried value(s) the engine "
                f"cannot accept ({detail}): a value that is not a number or not "
                f"one this model can take, or a NaN/inf the engine cannot rank. "
                f"The sampler {action} rather than send a partial point. Optuna "
                f"still records the trial normally.",
                stacklevel=4,
            )

    def _send_rejection(self, t, state, sol, is_attach_scan):
        """Tell the engine that the FAIL or PRUNED trial `t` has no result."""
        if sol is not None:
            # reject() answers False, and does not raise, when the engine refuses.
            if not sol.reject():
                warnings.warn(
                    f"OptQuestSampler: trial {t.number}'s rejection failed to reach "
                    f"OptQuest ({_last_error(sol)}); "
                    f"the engine was not told this solution failed.",
                    stacklevel=3,
                )
        elif not is_attach_scan and t.system_attrs.get("fixed_params"):
            # An enqueued trial has no solution to reject (sample_relative gives it none), so
            # the engine cannot be told that the point failed. A trial that was in the study
            # before the sampler is expected to have none (see "Enqueueing known points" in the
            # README), and so is the sampler's own pruned marker trial at the end of a search:
            # neither is enqueued, and neither gets this warning.
            warnings.warn(
                f"OptQuestSampler: trial {t.number} was enqueued via "
                f"study.enqueue_trial() but finished as {state.name} before OptQuest "
                f"ever offered it a solution to reject with; the engine was not told "
                f"this point failed.",
                stacklevel=3,
            )

    def sample_relative(self, study, trial, search_space):
        # The warnings of this method, of _stop_search() and of sample_independent() are
        # attributed to this file: Optuna calls these methods, so one frame further up is
        # Optuna's code, and a filter on Optuna's warnings would hide them.
        with self._sync_lock:  # one handshake with the engine at a time (see __init__)
            # Send the engine each finished trial that it has not seen. after_trial() sends the
            # trials of this run when they end, so this finds the trials that were in the study
            # before the sampler (a resumed study; see "Enqueueing known points" in the README)
            # and the ones that study.add_trial() added.
            all_trials = study.get_trials(deepcopy=False, states=_FINISHED_STATES)
            new_trials = [t for t in all_trials if t.number not in self._processed_trial_ids]
            is_attach_scan = not self._first_scan_done
            for t in new_trials:
                self._processed_trial_ids.add(t.number)
                self._finalize_trial(
                    t, t.state, t.values, is_attach_scan=is_attach_scan, own_trial=False
                )
            # Set before the return for an enqueued trial just below: when trial 0 is enqueued,
            # the scan above was still the first one, and a later one must not count as it.
            self._first_scan_done = True

            if trial.system_attrs.get("fixed_params"):
                # An enqueued trial. Optuna has its fixed parameters, and sample_independent
                # draws the others. It gets no engine solution, so when it completes its result
                # goes to the engine as an evaluated point (see _send_result).
                self._warn_partly_enqueued(trial.number, trial.system_attrs["fixed_params"])
                return {}

            sol = self._model.create_solution()
            while sol.is_valid() and sol.is_evaluated():
                # The engine gives back a point that add_evaluated() sent it once more, with
                # its outputs: it is submitted as it is, and no trial runs for it.
                if not sol.submit():
                    warnings.warn(
                        f"OptQuestSampler: an already-evaluated point failed to reach "
                        f"OptQuest ({_last_error(sol)}); "
                        f"its result was not recorded by the engine.",
                        stacklevel=1,
                    )
                    break
                sol = self._model.create_solution()
            if not sol.is_valid():
                self._stop_search(study)  # raises optuna.TrialPruned
            # Kept until the trial ends: its result goes back to the engine in this solution.
            self._solutions[trial.number] = sol
            self._warn_partly_fixed(trial.number, search_space)
            # The values of the parameters in the search space that Optuna passed in. A wrapper
            # such as optuna.samplers.PartialFixedSampler removes its fixed parameters from it,
            # and Optuna refuses a value for a parameter that is not in it.
            suggested_params = {}
            for var_name in search_space:
                if var_name in self._space:
                    engine_value = sol.get_value(self._engine_names.get(var_name, var_name))
                    suggested_params[var_name] = self._from_engine[var_name](engine_value)
            self._proposed[trial.number] = suggested_params
            return suggested_params

    def _stop_search(self, study):
        """End the study when the engine gives no solution: warn when the reason calls for it,
        ask the study to stop, and prune the trial that asked for a point (raises TrialPruned).

        create_solution() gives an invalid solution when the search has ended (the demo cap of
        50 iterations, an optimum found, a time or iteration limit, an exhausted space, ...) and
        when the engine failed while it made the solution; the binding raises only when it gets
        no solution object at all. The sampler never raises an error for it. It warns once for
        each sampler, also under n_jobs>1, for the demo cap, for a finite space that is
        exhausted, and for each reason of _ABNORMAL_TERMINATION_REASONS, so that a failure can
        be told apart from a search that is done. Every other reason gives no warning. Each
        warning names the study: Python shows one text once for each place in the code, and a
        later study in the same process needs its own. The warnings have the stacklevel of
        sample_relative()."""
        reason = self._model.get_termination_reason()
        if reason == optquest.OQTerminationReasonMaxDemoIterations:
            if self._first_time("demo cap"):
                warnings.warn(
                    f"OptQuestSampler: study {study.study_name!r} stopped at the "
                    f"OptQuest demo-license cap (50 iterations). Pass a license key "
                    f"with the 'license' argument to continue past it.",
                    stacklevel=2,
                )
        elif reason == _EXHAUSTED_TERMINATION_REASON:
            if self._first_time("exhausted space"):
                warnings.warn(
                    f"OptQuestSampler: in study {study.study_name!r}, OptQuest has "
                    f"generated every point its search space contains, so the sampler "
                    f"prunes this trial and stops the study. This is a normal end for "
                    f"a space with a manifold variable that has a finite set of values, "
                    f"not an error.",
                    stacklevel=2,
                )
        elif reason in _ABNORMAL_TERMINATION_REASONS:
            if self._first_time("engine failure"):
                warnings.warn(
                    f"OptQuestSampler: in study {study.study_name!r}, OptQuest stopped "
                    f"early due to "
                    f"{_ABNORMAL_TERMINATION_REASONS[reason]}, not a normal search "
                    f"conclusion "
                    f"({_engine_error(self._model)}), so the sampler prunes this trial "
                    f"and stops the study. This is not the same as convergence or "
                    f"exhausting the iteration/time budget -- check the model and the "
                    f"engine error message above.",
                    stacklevel=2,
                )
        # study.stop() takes effect between trials, and this trial still needs an end. The
        # engine gave it no values, so it is pruned, and Optuna does not choose values for it.
        # Under n_jobs>1, each worker that asks for a point after the end prunes its own trial,
        # so up to n_jobs pruned marker trials can appear.
        try:
            study.stop()
        except RuntimeError:
            # study.stop() works only inside study.optimize(). With ask-and-tell, the
            # TrialPruned of this trial's suggest_*() call is the signal.
            pass
        raise optuna.TrialPruned()

    def sample_independent(self, study, trial, param_name, param_distribution):
        proposed = self._proposed.get(trial.number)
        if proposed is not None and param_name in proposed:
            # Optuna calls sample_independent for a parameter that sample_relative gave a value
            # for only when it discarded that value: the value is not in the distribution of
            # the trial.suggest_*() call. That happens when a hand-written call uses other
            # bounds than the registered distribution (Optuna allows that), or, as a bug, when
            # the registered distribution does not hold the engine's value. The random value
            # drawn below goes back to the engine as the trial's value, so the engine is told
            # that it proposed a point it did not propose; warn once for each parameter.
            with self._sync_lock:
                first = self._first_time(("discarded", param_name))
            if first:
                warnings.warn(
                    f"OptQuestSampler: Optuna discarded the engine's value "
                    f"{proposed[param_name]!r} for parameter {param_name!r}, because it is not "
                    f"in the distribution of the trial.suggest_*() call, {param_distribution!r}, "
                    f"and sampled a random value in its place. If that distribution differs "
                    f"from the one the sampler registered, {self._space.get(param_name)!r}, "
                    f"make the call match it (sampler.suggest_all() and wrap() do). Otherwise, "
                    f"please report this at https://github.com/optuna/optunahub-registry/issues.",
                    stacklevel=1,
                )
        return self._rng.sample_independent(study, trial, param_name, param_distribution)

    def after_trial(self, study, trial, state, values):
        """Send the engine the result of a trial of this run when the trial ends. Optuna calls
        this once for each trial, just before it stores the result. sample_relative() alone
        would find a finished trial only at its next call, so the results of the last trials
        of a study.optimize() run (up to n_jobs of them) would never reach the engine."""
        if state not in _FINISHED_STATES:
            return
        problem = self._direction_problem(study)
        if problem is not None:
            # A trial of a study that the sampler refuses: its first suggest_*() call raised
            # this error, or it never asked the sampler for its parameters (every parameter was
            # enqueued). Nothing of it reaches the engine, and it does not bind the sampler to
            # that study.
            raise ValueError(problem)
        self._check_study(study)
        with self._sync_lock:
            # Optuna calls this before it writes the trial to storage, and the scan of
            # sample_relative() takes only finished trials, so the scan never has a trial of
            # this study before this call. A number that this sampler already finished is a
            # trial of another study with the same name: an enqueued trial with every parameter
            # fixed never asks the sampler for its parameters, so infer_relative_search_space()
            # does not see it. Its result would go to the engine as a point of this search.
            finished = trial.number in self._processed_trial_ids
            if not finished:
                self._processed_trial_ids.add(trial.number)
                self._finalize_trial(trial, state, values, is_attach_scan=False, own_trial=True)
        if finished:
            raise RuntimeError(_another_study_message(study, trial.number))
