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

import optuna


try:
    import optquest
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install optquest` to use `OptQuestSampler`.")


class OptQuestSampler(optuna.samplers.BaseSampler):
    def __init__(self, model=None, search_space=None, directions=None, seed=None, license=""):
        # maps optuna categorical variable names to their choices values in optquest
        self._cat_names = {}

        # maps optuna log variables
        self._is_log = set()

        # list of output variable names in the optquest model
        self._out_names = []

        # store solutions associated with trials
        self._solutions = {}

        if model is not None and search_space is not None:
            raise ValueError("Provide either a model or search space, not both.")

        if model is None and search_space is None:
            raise ValueError("Either a model or search space must be provided.")

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
                    if var_distribution.step is not None:
                        model.add_discrete_variable(
                            var_name,
                            var_distribution.low,
                            var_distribution.high,
                            var_distribution.step,
                        )
                    elif var_distribution.log:
                        model.add_integer_variable(
                            var_name,
                            math.log(var_distribution.low),
                            math.log(var_distribution.high),
                        )
                        self._is_log.add(var_name)
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
        elif isinstance(model, optquest.OptQuestModel):
            # create search space from model
            if seed is not None:
                model.set_random_seed(seed)
            if not model.initialize():
                raise RuntimeError(model.get_last_error())
            search_space = {}
            for model_info in model.describe().splitlines():
                # parse each line of the model description
                var_info = model_info.split(" ")
                match var_info[0].upper():
                    case "CONTINUOUS":
                        search_space[var_info[1]] = optuna.distributions.FloatDistribution(
                            float(var_info[2]), float(var_info[3])
                        )
                    case "INTEGER":
                        search_space[var_info[1]] = optuna.distributions.IntDistribution(
                            int(var_info[2]), int(var_info[3])
                        )
                    case "DISCRETE":
                        search_space[var_info[1]] = optuna.distributions.FloatDistribution(
                            float(var_info[2]), float(var_info[3]), step=float(var_info[4])
                        )
                    case "BINARY":
                        search_space[var_info[1]] = optuna.distributions.IntDistribution(0, 1)
                    case "DESIGN":
                        self._cat_names[var_info[1]] = [
                            str(i) for i in range(1, int(var_info[2]) + 1)
                        ]
                        search_space[var_info[1]] = optuna.distributions.CategoricalDistribution(
                            self._cat_names[var_info[1]]
                        )
                    case "ENUMERATION":
                        self._cat_names[var_info[1]] = [
                            str(var_info[i]) for i in range(2, len(var_info))
                        ]
                        search_space[var_info[1]] = optuna.distributions.CategoricalDistribution(
                            self._cat_names[var_info[1]]
                        )
                    case "PERMUTATION":
                        for item in var_info[2:]:
                            name = var_info[1] + "_" + item
                            search_space[name] = optuna.distributions.IntDistribution(
                                1, len(var_info[2:])
                            )
                    case "TUPLE":
                        self._cat_names[var_info[1]] = [str(i) for i in var_info[2].split(";")]
                        search_space[var_info[1]] = optuna.distributions.CategoricalDistribution(
                            self._cat_names[var_info[1]]
                        )
                    case "GEOLOCATION":
                        self._cat_names[var_info[1]] = [str(i) for i in var_info[2].split(";")]
                        search_space[var_info[1]] = optuna.distributions.CategoricalDistribution(
                            self._cat_names[var_info[1]]
                        )
                    case "SELECTION":
                        search_space[var_info[1]] = optuna.distributions.IntDistribution(0, 1)
                    case "OUTPUT":
                        self._out_names.append(var_info[1])
                    case "OBJECTIVE":
                        pass

        self._space = search_space
        self._model = model
        self._rng = optuna.samplers.RandomSampler(seed=seed)
        self._processed_trial_ids = set()  # Track all trials that have been sent back to OptQuest

    def infer_relative_search_space(self, study, trial):
        return self._space

    def sample_relative(self, study, trial, search_space):
        relevant_states = (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
        )
        all_trials = study.get_trials(deepcopy=False, states=relevant_states)

        # Find new ones not yet returned
        new_trials = [t for t in all_trials if t.number not in self._processed_trial_ids]
        for t in new_trials:
            self._processed_trial_ids.add(t.number)
            sol = self._solutions.get(t.number)  # get the solution object from the trial
            if t.state == optuna.trial.TrialState.COMPLETE:
                if sol is None:
                    # if the solution is not attached, we create a new one and add it
                    sol = self._model.get_empty_solution()
                if sol is not None:
                    # update the solution with parameter values and objective results
                    for param_name, param_value in t.params.items():
                        if param_name in self._cat_names:
                            index = self._cat_names[param_name].index(param_value) + 1
                            sol.set_value(param_name, index)
                        elif param_name in self._is_log:
                            sol.set_value(param_name, math.log(param_value))
                        else:
                            sol.set_value(param_name, param_value)
                    # set objective values
                    if len(t.values) > 0:
                        for i, obj_value in enumerate(t.values):
                            sol.set_value(self._out_names[i], obj_value)
                    else:
                        sol.set_value(self._out_names[0], t.value)
                    # send the solution back to OptQuest
                    if sol.get_iteration() == 0:
                        # if we created it new, then we just send it back as evaluated
                        sol.add_evaluated()
                    else:
                        sol.submit()
            elif (
                t.state == optuna.trial.TrialState.FAIL
                or t.state == optuna.trial.TrialState.PRUNED
            ):
                # send rejection to OptQuest
                if sol is not None:
                    sol.reject()

        # get the next solution from the engine
        suggested_params = {}
        sol = self._model.create_solution()
        if sol.is_valid():
            # save the solution object associated with the trial, we need it to submit the completed trial
            self._solutions[trial.number] = sol
            # extract parameter values from the solution
            for var_name in self._space.keys():
                if var_name in self._cat_names:
                    suggested_params[var_name] = self._cat_names[var_name][
                        int(sol.get_value(var_name)) - 1
                    ]
                elif var_name in self._is_log:
                    suggested_params[var_name] = math.exp(sol.get_value(var_name))
                else:
                    suggested_params[var_name] = sol.get_value(var_name)
        else:
            raise RuntimeError(self._model.get_last_error())
        return suggested_params

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._rng.sample_independent(study, trial, param_name, param_distribution)
