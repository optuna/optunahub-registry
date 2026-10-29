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

import math
import threading
import warnings

import optuna


try:
    import optquest
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install optquest` to use `OptQuestSampler`.")

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
    refused here and the trial is rejected; RayTune and Hydra reject such a trial too.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _index_converters(labels):
    """Converters for a categorical variable backed by a 1-based engine index: the engine
    stores the index, Optuna's CategoricalDistribution holds the string label."""

    def to_engine(label):
        return float(labels.index(label) + 1)

    def from_engine(raw_index):
        return labels[int(raw_index) - 1]

    return to_engine, from_engine


def _parse_direction(token):
    normalized = str(token).strip().lower()
    if normalized == "min":
        return optuna.study.StudyDirection.MINIMIZE
    if normalized == "max":
        return optuna.study.StudyDirection.MAXIMIZE
    raise ValueError(
        f"objective direction must be 'min' or 'max' (case-insensitive), got {token!r}"
    )


def _suggest_one(trial, name, dist):
    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    if isinstance(dist, optuna.distributions.IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high, step=dist.step or 1, log=dist.log)
    if isinstance(dist, optuna.distributions.FloatDistribution):
        return trial.suggest_float(name, dist.low, dist.high, step=dist.step, log=dist.log)
    raise TypeError(f"Unsupported distribution for {name!r}: {type(dist)}")


def _flatten_output(name, kind, shape, value):
    """Inverse of _variable_codec's grid decode: a natural (nested-list) output value -> the
    engine's flat "<name>@<key>" component dict. Shaped outputs use the same "<name>@<key>"
    convention as shaped inputs (e.g. an OUTPUTVECTOR "v" of length 3 is "v@1"/"v@2"/"v@3")."""
    if kind == "vector":
        n = shape
        if len(value) != n:
            raise ValueError(f"output {name!r}: expected {n} value(s), got {len(value)}")
        return {f"{name}@{i + 1}": float(v) for i, v in enumerate(value)}
    if kind == "matrix":
        rows, cols = shape
        if len(value) != rows or any(len(row) != cols for row in value):
            raise ValueError(f"output {name!r}: expected a {rows}x{cols} matrix")
        return {
            f"{name}@{r + 1},{c + 1}": float(value[r][c]) for r in range(rows) for c in range(cols)
        }
    if kind == "nd":
        import itertools

        flat = {}
        for idx in itertools.product(*(range(d) for d in shape)):
            v = value
            for i in idx:
                v = v[i]
            flat[f"{name}@" + ",".join(str(i + 1) for i in idx)] = float(v)
        return flat
    raise AssertionError(f"unknown output kind {kind!r}")


# Termination reasons that show an engine-side problem, and not a normal end of the search,
# when create_solution() returns an invalid solution. See the exhaustion branch of
# sample_relative below. In the engine (Optimization.setTerminationReason), TERM_INFEASIBLE
# is set together with a thrown COptQuestException when the constraints of the model make it
# infeasible. Neither it nor an internal exception is a normal stop in the way that
# OptimalFound/MaxIterations/MaxTime/UserStopped/BestFound/FactoriesDone/LP are, so the
# sampler gives a warning for both.
_ABNORMAL_TERMINATION_REASONS = {
    optquest.OQTerminationReasonException: "an internal engine exception",
    optquest.OQTerminationReasonInfeasible: "the model's constraints being infeasible",
}

# CannotGenerate (13) is not in the set above, on purpose. A finite search space ends with
# this reason: a manifold variable with a finite set of values after the engine has produced
# every value, or a fully enumerated DESIGN or PERMUTATION. A warning to "check the model and
# the engine error message above" for a run that ended correctly would teach the user to
# ignore that message when it matters. This reason is reported at info level instead,
# because the remaining trials are still pruned and the user should know why they stopped.
_EXHAUSTED_TERMINATION_REASON = optquest.OQTerminationReasonCannotGenerate


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
        if seed is not None and seed <= 0:
            raise ValueError(
                f"seed must be a positive int, got {seed!r} -- OptQuest's engine only applies a "
                "random seed when it is > 0 (see OptQuestModel.SetRandomSeed()/Initialize() in "
                "the Java reference engine); seed<=0 would silently have no effect at all. Omit "
                "seed (leave it None) to let the engine pick its own."
            )
        # maps optuna categorical variable names to their choices values in optquest
        self._cat_names = {}

        # maps optuna log variables
        self._is_log = set()

        # (low, high) int bounds for each log-scale IntDistribution variable in self._is_log;
        # the sampler builds a rounding and clamping from_engine decoder from them (see
        # _log_int_decoder) instead of the plain math.exp used for log-scale FloatDistribution
        # variables.
        self._is_log_int_bounds = {}
        # (low, high) float bounds for each log-scale FloatDistribution variable, used the
        # same way: to build a clamping decoder rather than a plain math.exp, whose result can
        # be outside the distribution at a bound.
        self._is_log_float_bounds = {}

        # list of output variable names in the optquest model
        self._out_names = []

        # name -> ("scalar"|"vector"|"matrix"|"nd", shape) for every declared output; populated
        # by _record_output for every construction mode that goes through _ingest_statement
        # (statements=/model=). The plain search_space=/directions= mode never has shaped
        # outputs, so it does not populate this. Used by wrap() and the output flattening
        # (see _pack_outputs/_flatten_output).
        self._out_shapes = {}

        # for each objective (in declaration order), the output/expression name it is an
        # identity pass-through of. Only populated in the "OptQuest way" (model given).
        # see the OBJECTIVE case below for why non-identity expressions are rejected.
        self._objective_exprs = []

        # parallel to self._objective_exprs: each objective's optuna.study.StudyDirection,
        # parsed from the same OBJECTIVE statement/line. Used by the `directions` property and
        # the study-direction cross-check in infer_relative_search_space.
        self._objective_directions = []
        self._directions_validated = False

        # per-Optuna-param-name engine<->Optuna value converters, unified across every
        # construction mode (compiled from self._cat_names/self._is_log for the plain
        # search_space=/directions= mode; taken directly from _variable_codec.VariableCodec,
        # via _ingest_statement, for the statements=/model= modes) so sample_relative has
        # exactly one translation mechanism regardless of how the sampler was constructed. See
        # _index_converters and the "compile converters" step below.
        self._to_engine = {}
        self._from_engine = {}

        # statement-name -> VariableCodec for every variable declared via the declarative
        # construction mode. Reserved for the wrap()/natural-value-decode surface.
        self._var_codecs = {}

        # Solutions associated with trials. _finalize_trial pops an entry (and does not only
        # read it) once the result of the trial has been sent back to the engine. after_trial()
        # calls _finalize_trial when each trial finishes, and the attach-time scan of
        # sample_relative() calls it as a fallback. The pop releases the OptQuestSolution
        # wrapper, and the live native engine handle that it owns (should_delete=True), for GC
        # as soon as the trial is done, rather than keeping them for the life of the sampler.
        # Optuna calls after_trial() for every trial of this run, including the last one. So
        # an entry that is left here after study.optimize() returns means that the process was
        # killed during a trial (before after_trial could run for it), or that a trial is still
        # RUNNING (e.g. a call in progress on another worker under n_jobs>1).
        self._solutions = {}

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

        if isinstance(search_space, dict) and isinstance(directions, list):
            # create model from search space
            model = optquest.OptQuestModel()
            model.set_license(license)
            for var_name, var_distribution in search_space.items():
                # Create OptQuest variables based on the distribution type
                if isinstance(var_distribution, optuna.distributions.CategoricalDistribution):
                    self._cat_names[var_name] = var_distribution.choices
                    model.add_design_variable(var_name, len(var_distribution.choices))
                elif isinstance(var_distribution, optuna.distributions.IntDistribution):
                    # Optuna's IntDistribution has step=1 by default and never None, unlike
                    # FloatDistribution, whose step is None by default. So a plain
                    # `step is not None` check is always true, and every integer variable,
                    # log-scale or not, would be built as DISCRETE, and log=True would be
                    # dropped with no warning. Only a coarser step (>1) means that the user
                    # wants a discrete variable; step==1 (the default) is a plain/log integer.
                    if var_distribution.step is not None and var_distribution.step > 1:
                        model.add_discrete_variable(
                            var_name,
                            var_distribution.low,
                            var_distribution.high,
                            var_distribution.step,
                        )
                    elif var_distribution.log:
                        # A CONTINUOUS engine variable (and not add_integer_variable), as in
                        # the log-scale FloatDistribution branch below. An INTEGER engine
                        # variable bounded by log(low)..log(high) would explore only as many
                        # distinct points as there are integers in that log-space range, which
                        # is usually small (e.g. 7 for low=1/high=1000), so it would sample too
                        # few points of the real range. _log_int_decoder rounds the value back
                        # to an int when it leaves the engine.
                        model.add_continuous_variable(
                            var_name,
                            math.log(var_distribution.low),
                            math.log(var_distribution.high),
                        )
                        self._is_log.add(var_name)
                        self._is_log_int_bounds[var_name] = (
                            var_distribution.low,
                            var_distribution.high,
                        )
                    else:
                        model.add_integer_variable(
                            var_name, var_distribution.low, var_distribution.high
                        )
                elif isinstance(var_distribution, optuna.distributions.FloatDistribution):
                    if var_distribution.step is not None:
                        model.add_discrete_variable(
                            var_name,
                            var_distribution.low,
                            var_distribution.high,
                            var_distribution.step,
                        )
                    elif var_distribution.log:
                        model.add_continuous_variable(
                            var_name,
                            math.log(var_distribution.low),
                            math.log(var_distribution.high),
                        )
                        self._is_log.add(var_name)
                        self._is_log_float_bounds[var_name] = (
                            var_distribution.low,
                            var_distribution.high,
                        )
                    else:
                        model.add_continuous_variable(
                            var_name, var_distribution.low, var_distribution.high
                        )
                else:
                    raise ValueError(f"Unsupported variable type: {type(var_distribution)}")
            for i, direction in enumerate(directions):
                # create output and objective variables
                out_name = f"OQout{i:02d}"
                self._out_names.append(out_name)
                model.add_output_variable(out_name)
                self._objective_directions.append(direction)
                if direction == optuna.study.StudyDirection.MINIMIZE:
                    model.add_minimize_objective(f"OQobj{i:02d}", out_name)
                elif direction == optuna.study.StudyDirection.MAXIMIZE:
                    model.add_maximize_objective(f"OQobj{i:02d}", out_name)
                else:
                    raise ValueError(f"Unsupported study direction: {direction}")
            if seed is not None:
                model.set_random_seed(seed)
            if not model.initialize():
                raise RuntimeError(model.get_last_error())
        elif statements is not None:
            # "OptQuest way, declarative": build the model directly from emitter statements
            # (the same {"cmd": "DAG", "name": ..., ...} shape optquest.emit_statement()/
            # optquest_loop.model_doc_to_statements() use) via _variable_codec, instead of
            # decomposing a pre-built OptQuestModel's describe() text. Every command that
            # _variable_codec knows (variables of every shape, outputs, expressions,
            # constraints, objectives, groups, run settings) is supported with no per-type
            # code here. A new OptQuest command needs a new _variable_codec handler and no new
            # branch in this class.
            model = optquest.OptQuestModel()
            model.set_license(license)
            if seed is not None:
                model.set_random_seed(seed)
            search_space = {}
            for stmt in statements:
                self._ingest_statement(stmt, search_space, build_on=model)
            if not model.initialize():
                raise RuntimeError(model.get_last_error())
        elif isinstance(model, optquest.OptQuestModel):
            # "OptQuest way, typed": a pre-built OptQuestModel keeps no statement list, so
            # recover one from its describe() text with optquest.parse_model() (see its
            # docstring for the rare grammar shapes that it cannot reverse without ambiguity).
            # Then pass the statements through the same _ingest_statement() path that the
            # declarative mode uses, so both modes support the same variable types with no
            # duplicate per-type logic. build_on=None here because the model already contains
            # these statements, so there is nothing left to build.
            if seed is not None and not model.set_random_seed(seed):
                # set_random_seed returns False on a model that the caller already
                # initialized (ERROR already initialized). The seed would not apply, and
                # there would be no error.
                raise RuntimeError(f"seed={seed!r} was not applied: {model.get_last_error()}")
            if not model.initialize():
                raise RuntimeError(model.get_last_error())
            search_space = {}
            for stmt in optquest.parse_model(model.describe()):
                self._ingest_statement(stmt, search_space, build_on=None)

        if self._objective_exprs:
            # sample_relative's write-back maps each value the user's objective(trial) returns
            # onto the output its objective is an identity pass-through of. That mapping is only
            # well-defined when every objective is such a pass-through. A compound expression
            # (e.g. "out00 + out01") has no single output to receive the returned value, so
            # this is required here rather than left to fail later at write-back time.
            non_identity = [expr for expr in self._objective_exprs if expr not in self._out_names]
            if non_identity:
                raise NotImplementedError(
                    "OptQuestSampler's objective(trial) contract only supports objectives that "
                    "are an identity pass-through of exactly one declared output (e.g. "
                    "add_minimize_objective(name, 'out00')). Found compound/non-identity "
                    f"objective expression(s): {non_identity!r}. Route each composite value "
                    "through its own dedicated output instead."
                )

        # Compile self._cat_names/self._is_log (built only by the plain search_space=
        # construction mode above) into the same per-param to_engine/from_engine converters
        # that every other construction mode already populated directly from _variable_codec.
        # sample_relative then has one translation mechanism for every construction mode.
        for var_name in search_space:
            if var_name in self._to_engine:
                continue  # declarative mode already supplied real converters for this one
            if var_name in self._cat_names:
                self._to_engine[var_name], self._from_engine[var_name] = _index_converters(
                    self._cat_names[var_name]
                )
            elif var_name in self._is_log_int_bounds:
                low, high = self._is_log_int_bounds[var_name]
                self._to_engine[var_name] = math.log
                self._from_engine[var_name] = _log_int_decoder(low, high)
            elif var_name in self._is_log:
                self._to_engine[var_name] = math.log
                bounds = self._is_log_float_bounds.get(var_name)
                self._from_engine[var_name] = _log_float_decoder(*bounds) if bounds else math.exp
            else:
                self._to_engine[var_name] = _identity
                self._from_engine[var_name] = _identity

        # All declared output names, scalar and shaped. wrap() validates the dict that sim()
        # returns against this list. self._out_shapes (populated by _ingest_statement, i.e.
        # the statements-based and typed-model modes) already has every output, including
        # scalars; the plain search_space=/directions= mode never populates it, so
        # self._out_names (scalar-only there) is the complete list in that case.
        self._declared_outputs = (
            list(self._out_shapes) if self._out_shapes else list(self._out_names)
        )

        self._space = search_space
        self._model = model
        self._rng = optuna.samplers.RandomSampler(seed=seed)
        self._processed_trial_ids = set()  # Track all trials that have been sent back to OptQuest

        # Serializes the whole body of sample_relative() (the scan for newly completed or
        # failed trials, their submit or reject calls, and the request for the next solution).
        # With study.optimize(..., n_jobs>1), Optuna calls sample_relative() from a thread pool
        # with no synchronization of its own. The bookkeeping of this method (the
        # check-then-act on self._processed_trial_ids, and the shared engine handshake through
        # create_solution()) is not safe to interleave across threads. Without the lock, two
        # threads could both see the same completed trial as "new" and both submit() the same
        # OptQuestSolution, or both see the demo-cap termination and each give a warning. One
        # lock around the whole method makes each call atomic. optquest.py uses the same
        # lock-the-native-boundary pattern internally (_dll_lock/_solution_lock) at a finer
        # grain.
        self._sync_lock = threading.Lock()
        self._demo_cap_warned = False  # emit the demo-cap UserWarning at most once per sampler
        self._error_termination_warned = False  # ditto, for _ABNORMAL_TERMINATION_REASONS
        self._exhausted_warned = False  # ditto, for the normal "space is exhausted" ending
        # Names of parameters the objective suggested outside the registered search space;
        # each is warned about once (see _warn_unregistered).
        self._unregistered_warned = set()

        # Set to True when the first call of sample_relative() finishes its scan (see
        # _finalize_trial's is_attach_scan parameter). The sampler uses it to tell trials that
        # were already in study storage at attach time (no warning by design, see README's
        # "Enqueueing known points") apart from trials that after_trial() finds during this
        # run. It is set right after the scan loop, before the fixed_params early return
        # below, so that an enqueued trial 0 does not delay it to a later call.
        self._first_scan_done = False

    @property
    def directions(self):
        """This sampler's declared objective directions, in order. Pass to
        optuna.create_study(directions=sampler.directions, ...) instead of hand-duplicating the
        direction list. On first use, infer_relative_search_space() compares the study's
        directions against these and raises if they don't match."""
        return list(self._objective_directions)

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
            # The sampler applies these through the license=/seed= kwargs in every construction
            # mode; see _variable_codec.build_model_statement's docstring for why.
            return
        if build_on is not None:
            _variable_codec.build_model_statement(build_on, stmt)
        kind = _variable_codec.classify(cmd)
        if cmd == "OBJECTIVE":
            self._objective_exprs.append(stmt["expression"])
            self._objective_directions.append(_parse_direction(stmt["direction"]))
        elif kind == "variable":
            vc = _variable_codec.build(stmt)
            search_space.update(vc.params)
            self._to_engine.update(vc.to_engine)
            self._from_engine.update(vc.from_engine)
            self._var_codecs[stmt["name"]] = vc
        elif kind == "output":
            self._record_output(stmt)
        # "skip"/other non-objective "other" commands (session/run knobs, random
        # variables, EXCLUSIVE/FIXGROUP/FLOATGROUP/EXCHANGE, EXPRESSION/CONSTRAINT, ...):
        # already applied to the model (build_on branch above, when applicable); nothing
        # further needed for the Optuna-facing side.

    def _warn_unregistered(self, param_name):
        """Warn once per parameter name that the objective suggests outside this sampler's
        search space. Caller must hold self._sync_lock."""
        if param_name in self._unregistered_warned:
            return
        self._unregistered_warned.add(param_name)
        warnings.warn(
            f"OptQuestSampler: parameter {param_name!r} is not in this sampler's search "
            f"space, so OptQuest does not model it; its values are sampled independently "
            f"and are not sent to the engine. Declare it in search_space/statements/model "
            f"to let OptQuest search it.",
            stacklevel=2,
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

    def infer_relative_search_space(self, study, trial):
        if self._objective_directions and not self._directions_validated:
            # One-time check, not per-trial: cheap, and study.directions doesn't change mid-run.
            self._directions_validated = True
            if len(study.directions) != len(self._objective_directions):
                raise ValueError(
                    f"study has {len(study.directions)} direction(s) but this sampler declares "
                    f"{len(self._objective_directions)} objective(s); pass "
                    f"directions=sampler.directions to optuna.create_study(...) instead of "
                    f"hand-listing them, or fix the mismatch."
                )
            for i, (declared, actual) in enumerate(
                zip(self._objective_directions, study.directions)
            ):
                if declared != actual:
                    raise ValueError(
                        f"study direction[{i}] is {actual.name} but this sampler's model "
                        f"declares {declared.name} for that objective; pass "
                        f"directions=sampler.directions to optuna.create_study(...) instead of "
                        f"hand-listing them, or fix the mismatch."
                    )
        return self._space

    def suggest_all(self, trial):
        """Make every trial.suggest_*() call this trial's search space requires, then group
        the resulting flat (possibly "<name>@<key>"-suffixed) sub-parameters back into one
        natural Python value per originally-declared OptQuest variable: a number for a scalar,
        a list for a vector or an ASSIGNMENT, a list of rows for a matrix, a dict keyed by item
        name for a PERMUTATION, SEQUENCE or MIXTURE, and a dict keyed by (from, to) for a DAG.
        See _variable_codec.VariableCodec for the shape of every command.

        Without this method, a model with e.g. an 8-node DAG variable would require the caller
        to hand-write ~56 trial.suggest_int() calls for the generated "<name>@<from>,<to>" edge
        names, then hand-decode the flags back into an adjacency structure. Call this once per
        trial (typically as the first line of a sim() passed to wrap(), which calls it for you)
        and index the result by the variable names you declared, and not by their generated
        "<name>@<key>" names.

        Falls back to returning raw, undecoded per-parameter values (keyed by their own Optuna
        param name) only for the plain search_space=/directions= construction mode, where every
        variable is a scalar Optuna distribution to begin with and there's nothing to decode.
        Every other construction mode (statements=/model=) goes through _ingest_statement,
        which populates self._var_codecs for every variable regardless of shape (including
        manifold types recovered from a pre-built model's describe() text), so decoding is
        available for every variable in those modes."""
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
        if not isinstance(result, dict):
            raise TypeError(
                f"With model=/statements= construction, sim(params) must return a dict "
                f"keyed by output name (see self._declared_outputs), or None to reject the "
                f"trial; got {type(result)}."
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
                flat[name] = float(result[name])
            else:
                flat.update(_flatten_output(name, kind, shape, result[name]))
        # Stash the FLAT {str: float} dict rather than the natural structure sim() returned:
        # user_attrs must be JSON-serializable for RDB/Journal storages, and this is the form
        # that sample_relative needs to call sol.set_value() with no further work.
        trial.set_user_attr("_optquest_outputs", flat)
        if any(expr not in self._out_names for expr in self._objective_exprs):
            # Defensive: __init__ already rejects non-identity objectives at construction time,
            # so this line should be unreachable. If it were reached, sol.set_value() would have
            # no way to receive a composite value, so raise an error rather than guess.
            raise NotImplementedError(
                "wrap() requires every objective to be an identity pass-through of one output; "
                "this should have been rejected at construction time (see __init__)."
            )
        return tuple(float(result[expr]) for expr in self._objective_exprs)

    def _finalize_trial(self, t, state, values, *, is_attach_scan):
        """Submit/reject one finished trial's result to the engine. The two callers:

          - after_trial(): Optuna calls this synchronously, once, when each trial of this
            run reaches a terminal state. That includes the last trial of the run, when
            there may be no further sample_relative() call to find it otherwise. This is
            the primary path.
          - sample_relative()'s own scan: catches trials that were already in study
            storage before this sampler's first sample_relative() call (a resumed study, or
            one a different sampler already ran trials against; see the README's
            "Enqueueing known points" section). It is also a fallback for any
            trial after_trial() didn't run for.

        Caller must hold self._sync_lock and must have already added t.number to
        self._processed_trial_ids (so a trial is never finalized twice, regardless of which
        caller sees it first).

        `state`/`values` are taken as explicit parameters rather than read off `t.state`/
        `t.values`: after_trial() receives a FrozenTrial snapshotted by Optuna *before* the
        storage write that finalizes it (see optuna.study._tell._tell_with_warning),
        so `t.state`/`t.values` on that snapshot are stale (still RUNNING/None). There, the
        state/values Optuna passes as separate arguments are the authoritative ones.
        sample_relative()'s own scan fetches `t` fresh from storage after that write, so
        `t.state`/`t.values` would be accurate for it too, but taking the same two parameters
        from both callers keeps this method's behavior identical regardless of which one is
        calling, rather than silently depending on which path a given trial took.

        `is_attach_scan` suppresses the FAIL/PRUNED-with-no-solution warning for a trial found
        already in storage at attach time (see README). That case is documented and expected,
        and such a trial can never be rejected (there was no solution object for it).
        after_trial() always passes False: every trial it's called for was created by this
        sampler's own run, so an enqueued trial that fails during the run is a new lost
        result."""
        # Pop (not just read): once this trial's solution is looked up here, this call is done
        # with it one way or another (evaluated, submitted, or rejected below), so drop the
        # sampler's own reference now rather than keep the OptQuestSolution wrapper, and the
        # live native engine handle that it owns (should_delete=True), alive longer than
        # necessary.
        sol = self._solutions.pop(t.number, None)
        if state == optuna.trial.TrialState.COMPLETE:
            if sol is None:
                # if the solution is not attached, we create a new one and add it
                sol = self._model.get_empty_solution()
                if sol is None:
                    # get_empty_solution() can return None (see its docstring in
                    # optquest.py). This trial is already frozen COMPLETE in Optuna's
                    # storage, with no OptQuestSolution in which to send its result to the
                    # engine. Warn now rather than fall through the `if sol is not None:`
                    # test below with no signal. That would drop the result with no
                    # warning, in the same way as an unchecked submit()/reject() (see the
                    # warning below), one step earlier. get_last_error() is the same
                    # sticky-on-success accessor that submit()/reject() report through
                    # below, so it may be stale or blank here too. Its message is a
                    # best-effort hint about the cause.
                    warnings.warn(
                        f"OptQuestSampler: trial {t.number} completed but OptQuest could "
                        f"not provide an empty solution to record it in "
                        f"({self._model.get_last_error() or 'no engine error message'}); "
                        f"its result was not recorded by the engine.",
                        stacklevel=2,
                    )
            if sol is not None:
                # Update the solution with parameter values, objective/output values, and
                # send it back to the engine. Every sol.set_value() call in this block
                # (parameters below, plus outputs further down) ignores its bool return on
                # purpose: per api-behavior.md §6, OQSetValue does no validation and
                # reports success in every case, even for a misspelled or undeclared name.
                # A bad name becomes a disconnected write with no error (a key that nothing
                # reads). A check of the return here would be dead code, and would wrongly
                # imply that misspelled names are caught. submit()/add_evaluated()/reject()
                # below are different: they follow a real fail/succeed convention (see
                # their docstrings), so only those calls are checked and warned on.
                #
                # A parameter this sampler did not register (one the objective suggests
                # outside search_space; Optuna routes it to sample_independent) has no
                # variable in the engine model: set_value accepts the name, but submit()
                # then refuses the whole solution with ERROR BAD VALUE NAME. Such a
                # parameter is skipped, with one warning per name, and the trial's
                # registered values and outputs are still recorded.
                unsendable = []
                for param_name, param_value in t.params.items():
                    to_engine = self._to_engine.get(param_name)
                    if to_engine is None:
                        self._warn_unregistered(param_name)
                        continue
                    engine_value = _engine_number(to_engine(param_value))
                    if engine_value is None:
                        unsendable.append((param_name, param_value))
                        continue
                    sol.set_value(param_name, engine_value)
                # Set objective/output values. If this trial went through wrap(), sim()'s
                # dict return was already validated and flattened onto every declared
                # output (not just the objective-linked ones) in _pack_outputs, and stashed
                # as a user_attr. Apply that directly rather than re-deriving it.
                flat_outputs = t.user_attrs.get("_optquest_outputs")
                if flat_outputs is not None:
                    for out_key, value in flat_outputs.items():
                        engine_value = _engine_number(value)
                        if engine_value is None:
                            unsendable.append((out_key, value))
                            continue
                        sol.set_value(out_key, engine_value)
                else:
                    # Manual path (objective(trial) written and returned by hand, not via
                    # wrap()): map each returned value onto the output its objective is an
                    # identity pass-through of (self._objective_exprs, parsed from each
                    # OBJECTIVE line/statement and validated in __init__). Outputs and
                    # objectives need not be declared 1:1 in the same order (an output used
                    # only by a constraint has no corresponding objective), so this maps by
                    # objective rather than by output-declaration position. In the "Optuna
                    # way" (no model given) self._objective_exprs is empty, and
                    # self._out_names, which is built 1:1 with directions in that mode, is
                    # the right target list.
                    target_names = self._objective_exprs or self._out_names
                    # A declared output with no corresponding objective (e.g. one used only
                    # by a CONSTRAINT) has nothing in target_names to write it from, and this
                    # manual path (hand-written objective(trial), no wrap()) has no value
                    # for it. If the output stayed at the engine's default (0.0) with no
                    # error, every constraint that uses it would be met in every trial,
                    # whatever the real simulated value. Raise an error instead of guessing.
                    unlinked = [n for n in self._declared_outputs if n not in target_names]
                    if unlinked:
                        raise NotImplementedError(
                            f"trial {t.number}: this model declares output(s) {unlinked!r} "
                            f"that are not an identity pass-through of any objective (e.g. "
                            f"they're used only by a CONSTRAINT), so a hand-written "
                            f"objective(trial) has no value to give them -- returning "
                            f"objective values alone leaves them silently pinned at 0.0 for "
                            f"the engine's entire view of the run. Use sampler.wrap(sim) "
                            f"instead, with sim(params) returning a dict keyed by every "
                            f"name in self._declared_outputs (see the README's 'Manual / "
                            f"advanced usage' section)."
                        )
                    if values:
                        if len(values) != len(target_names):
                            raise ValueError(
                                f"trial {t.number} returned {len(values)} value(s) but "
                                f"the model declares {len(target_names)} objective(s) "
                                f"{target_names!r}; counts must match."
                            )
                        for i, obj_value in enumerate(values):
                            sol.set_value(target_names[i], obj_value)
                    else:
                        # `values` is None or empty: Optuna reported a COMPLETE trial with no
                        # objective value, so there is nothing to hand the engine. Warn and
                        # leave the solution's outputs at their defaults; the submit below
                        # still records the trial. Warn rather than raise for the
                        # misattribution reason described just below.
                        warnings.warn(
                            f"trial {t.number} reached OptQuest write-back in state "
                            f"{state!r} with no objective values ({values!r}); its outputs "
                            f"stay at the engine's defaults, so the engine's view of this "
                            f"trial is not the trial's real result.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                # Send the solution back to OptQuest. Unlike most OQ* calls, submit()/
                # add_evaluated() return False (and do not raise an exception) on an
                # engine-side failure, so the return value must be checked. If it were
                # discarded, a trial could finish COMPLETE in Optuna's storage while the
                # engine never received its result, and the engine's view of the run would
                # be wrong with no visible sign. Warn rather than raise: an exception here
                # would fail the trial that is being sampled or finalized when this runs,
                # which is not always trial t (e.g. when called from sample_relative()'s
                # scan of a batch that includes an unrelated trial), so the failure would be
                # attributed to the wrong trial.
                if unsendable:
                    # Reject, and do not submit, a point whose record in the engine would be
                    # incomplete: a value that is not a number cannot be marshalled, and a
                    # NaN or inf cannot be ranked. A rejection tells the engine that this
                    # trial happened and that its result could not be sent.
                    detail = ", ".join(f"{n}={v!r}" for n, v in unsendable)
                    warnings.warn(
                        f"OptQuestSampler: trial {t.number} carried value(s) the engine "
                        f"cannot accept ({detail}): a value that is not a number, or a "
                        f"NaN/inf the engine cannot rank. Rejecting this trial rather than "
                        f"submitting a partial point. Optuna still records the trial normally.",
                        stacklevel=2,
                    )
                    ok = sol.reject()
                elif sol.get_iteration() == 0:
                    # if the sampler created it, send it back as evaluated
                    ok = sol.add_evaluated()
                else:
                    ok = sol.submit()
                if not ok:
                    warnings.warn(
                        f"OptQuestSampler: trial {t.number}'s solution failed to reach "
                        f"OptQuest ({sol.get_last_error() or 'no engine error message'}); "
                        f"its result was not recorded by the engine.",
                        stacklevel=2,
                    )
        elif state in (optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED):
            if sol is not None:
                # send rejection to OptQuest. reject() uses the same non-exception failure
                # convention as submit()/add_evaluated() above, so check it the same way.
                if not sol.reject():
                    warnings.warn(
                        f"OptQuestSampler: trial {t.number}'s rejection failed to reach "
                        f"OptQuest ({sol.get_last_error() or 'no engine error message'}); "
                        f"the engine was not told this solution failed.",
                        stacklevel=2,
                    )
            elif not is_attach_scan and t.system_attrs.get("fixed_params"):
                # A trial enqueued via study.enqueue_trial() during this run never gets a
                # solution attached (sample_relative returns {} early for fixed_params trials,
                # see below). If it then fails or is pruned during evaluation, there is
                # nothing to reject with. Unlike the case of a trial already in storage at
                # attach time, which this branch excludes via is_attach_scan, this is a new
                # lost result of this run. The fixed_params test also keeps this warning off
                # this sampler's own pruned marker trials (see the create_solution()
                # exhaustion branch below): those were never enqueued, and they have no
                # solution because OptQuest had nothing left to offer, rather than because
                # an attachment was missed.
                warnings.warn(
                    f"OptQuestSampler: trial {t.number} was enqueued via "
                    f"study.enqueue_trial() but finished as {state.name} before OptQuest "
                    f"ever offered it a solution to reject with; the engine was not told "
                    f"this point failed.",
                    stacklevel=2,
                )
            # else: either a pre-existing FAIL/PRUNED trial found in storage at attach time
            # (documented no-op, see README's "Enqueueing known points" section), or this
            # sampler's own pruned marker trial from a normal create_solution() exhaustion
            # stop. Neither is a lost result that needs a warning.

    def sample_relative(self, study, trial, search_space):
        # See self._sync_lock's docstring in __init__: this whole method is one atomic
        # handshake with the engine and is not safe to interleave across the thread pool
        # study.optimize(..., n_jobs>1) drives sample_relative() from.
        with self._sync_lock:
            relevant_states = (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.PRUNED,
            )
            all_trials = study.get_trials(deepcopy=False, states=relevant_states)

            # Find new ones not yet returned. In normal operation this finds only trials that
            # were already in study storage before this sampler's first sample_relative() call
            # (see is_attach_scan below and the README's "Enqueueing known points" section).
            # after_trial() catches every trial of this run when it finishes (see its
            # docstring for why that override exists and why this scan alone is not enough).
            # This scan is kept as the attach-time path and as a defensive fallback for any
            # trial that after_trial did not run for. It is safe under the lock: no other
            # thread can add to self._processed_trial_ids/self._solutions at the same time, so
            # neither this loop nor after_trial() can classify a trial as "new", and finalize
            # it, twice.
            new_trials = [t for t in all_trials if t.number not in self._processed_trial_ids]
            is_attach_scan = not self._first_scan_done
            for t in new_trials:
                self._processed_trial_ids.add(t.number)
                self._finalize_trial(t, t.state, t.values, is_attach_scan=is_attach_scan)
            # Set this before the fixed_params early return just below. If trial 0 is an
            # enqueued point, this is still the attach scan for every trial found above, but a
            # later call must not be treated as one.
            self._first_scan_done = True

            if trial.system_attrs.get("fixed_params"):
                # Enqueued (full or partial), which is Optuna's version of "suggested". Suggest
                # nothing: fixed_params covers the fixed dims; any free dims go to
                # sample_independent (self._rng). Nothing is stashed in _solutions, so at
                # completion this trial goes to the sol-is-None -> add_evaluated branch (run
                # from after_trial(), via _finalize_trial) and enters OptQuest as a fully
                # external evaluated point.
                return {}

            # Every completed solution above has now been sent back to OptQuest, either via
            # add_evaluated() (loaded from the study) or submit() (created here). Get the next
            # solution from the engine.
            suggested_params = {}
            sol = self._model.create_solution()
            while sol.is_valid() and sol.is_evaluated():
                # A point fed in via add_evaluated() is re-issued once by create_solution()
                # with is_evaluated() set and its outputs intact; submit it as-is rather
                # than running a trial for it.
                if not sol.submit():
                    warnings.warn(
                        f"OptQuestSampler: an already-evaluated point failed to reach "
                        f"OptQuest ({sol.get_last_error() or 'no engine error message'}); "
                        f"its result was not recorded by the engine.",
                        stacklevel=2,
                    )
                    break
                sol = self._model.create_solution()
            if not sol.is_valid():
                # create_solution() returns an invalid solution when the engine has no more
                # solutions to offer (the demo-license 50-iteration cap, or any other terminal
                # reason: optimum found, max iterations/time, user-stopped, infeasible, ...).
                # A real failure raises OptQuestError from the binding instead. For every
                # reason, the sampler stops without an error (prune this trial, study.stop())
                # rather than crash the study. The demo cap gets its own warning below, with
                # the action to take, and so does every reason in
                # _ABNORMAL_TERMINATION_REASONS. An internal engine exception, an infeasible
                # model, or the engine's unclassified "can't generate" fallback is a real
                # problem and not a converged or exhausted search, and a licensed run that
                # stops for one needs a warning, so that the user can tell it apart from
                # success. Every other reason (OptimalFound, MaxIterations, MaxTime,
                # UserStopped, BestFound, FactoriesDone, LP) is a normal end and gives no
                # warning. (Under n_jobs>1, each worker in progress can enter here and see the
                # same exhausted or failed state, so each warning below is given only once for
                # each sampler.)
                reason = self._model.get_termination_reason()
                if reason == optquest.OQTerminationReasonMaxDemoIterations:
                    if not self._demo_cap_warned:
                        self._demo_cap_warned = True
                        warnings.warn(
                            "OptQuestSampler: stopped at the OptQuest demo-license cap (50 "
                            "iterations). Pass a real license via the 'license' kwarg to "
                            "continue past it.",
                            stacklevel=2,
                        )
                elif reason == _EXHAUSTED_TERMINATION_REASON:
                    if not self._exhausted_warned:
                        self._exhausted_warned = True
                        warnings.warn(
                            "OptQuestSampler: OptQuest has generated every point its search "
                            "space contains, so remaining trials are being pruned rather "
                            "than re-run. This is a normal ending for a finite space (a "
                            "fully-enumerated design, permutation, or manifold variable "
                            "with a finite set of values), not an error -- ask for fewer "
                            "trials to avoid the "
                            "pruned tail.",
                            stacklevel=2,
                        )
                elif reason in _ABNORMAL_TERMINATION_REASONS:
                    if not self._error_termination_warned:
                        self._error_termination_warned = True
                        warnings.warn(
                            f"OptQuestSampler: OptQuest stopped early due to "
                            f"{_ABNORMAL_TERMINATION_REASONS[reason]}, not a normal search "
                            f"conclusion "
                            f"({self._model.get_last_error() or 'no engine error message'}); "
                            f"remaining trials are being pruned rather than run against a "
                            f"search that isn't actually progressing. This is not the same as "
                            f"convergence or exhausting the iteration/time budget -- check the "
                            f"model and the engine error message above.",
                            stacklevel=2,
                        )
                # Normal stop: study.stop() takes effect only between trials (it lets any trial
                # that is already running finish), and this trial still needs a resolution.
                # OptQuest gave it no values, so prune it rather than let Optuna choose values
                # for it. Under n_jobs>1, every worker still running a trial when the cap is
                # reached comes here and prunes its own trial (a trial that another worker is
                # running cannot be resolved later from here), so up to n_jobs pruned marker
                # trials can appear. n_jobs=1 is the only case with exactly one.
                study.stop()
                raise optuna.TrialPruned()
            # save the solution object associated with the trial, we need it to submit the completed trial
            self._solutions[trial.number] = sol
            # extract parameter values from the solution
            for var_name in self._space.keys():
                suggested_params[var_name] = self._from_engine[var_name](sol.get_value(var_name))
            return suggested_params

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._rng.sample_independent(study, trial, param_name, param_distribution)

    def after_trial(self, study, trial, state, values):
        """Optuna calls this synchronously, once, when each trial finishes, immediately
        before that trial's state/values are written to study storage (see
        optuna.samplers.BaseSampler.after_trial's docstring). Optuna calls it whether or not
        sample_relative() is invoked again afterward. Without this override,
        sample_relative() alone only discovers a finished trial lazily, at the start of its
        next call: the last completed trial(s) of a study.optimize() run (up to n_jobs of
        them, under a parallel run) would finish, sample_relative() would never be asked for
        another suggestion, and their results would remain COMPLETE in Optuna's own storage
        and never reach the engine. No warning would appear, because no code attempted to
        submit them. This override handles each trial when it finishes instead of waiting
        to notice it later.

        See self._sync_lock's docstring in __init__ for why this needs the same lock
        sample_relative() takes, and _finalize_trial's docstring for why state/values are
        taken from here rather than re-read off `trial`."""
        if state not in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
        ):
            return
        with self._sync_lock:
            if trial.number in self._processed_trial_ids:
                # Already handled by sample_relative()'s own scan (e.g. this trial was already
                # in storage at attach time, and a later sample_relative() call ran before
                # Optuna's call into after_trial for it). A second finalize would submit or
                # reject the same result twice.
                return
            self._processed_trial_ids.add(trial.number)
            self._finalize_trial(trial, state, values, is_attach_scan=False)
