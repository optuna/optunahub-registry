from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.sampler import _split_trials
from optuna.trial import TrialState


class TPEAcquisitionVisualizer:
    def __init__(self) -> None:
        """
        Initializes the TPEAcquisitionVisualizer.
        """
        self.log_objects: dict[int, Any] = {}

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Callback function to collect tpe sampler's acquisition information.

        Args:
            study (optuna.study.Study): The study object.
            trial (optuna.trial.FrozenTrial): The trial object for which the callback is called.
        Returns:
            None
        """
        sampler = study.sampler
        if sampler._multivariate:
            raise ValueError("This callback is not compatible with multivariate TPE sampler.")

        if sampler._constant_liar:
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not sampler._constant_liar
        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)
        trials = [t for t in trials if t.number != trial.number]

        # We divide data into below and above.
        n = sum(trial.state != TrialState.RUNNING for trial in trials)  # Ignore running trials.
        below_trials, above_trials = _split_trials(
            study,
            trials,
            sampler._gamma(n),
            sampler._constraints_func is not None,
        )
        mpe_below = {}
        mpe_above = {}
        for key, distribution in trial.distributions.items():
            search_space = {key: distribution}
            mpe_below[key] = sampler._build_parzen_estimator(
                study, search_space, below_trials, handle_below=True
            )
            mpe_above[key] = sampler._build_parzen_estimator(
                study, search_space, above_trials, handle_below=False
            )

        self.log_objects[trial.number] = {
            "below_trials": below_trials,
            "above_trials": above_trials,
            "mpe_below": mpe_below,
            "mpe_above": mpe_above,
        }

    def plot(
        self,
        study: optuna.study.Study,
        trial_number: int,
        param_name: str,
    ) -> plt.Figure:
        """
        Plots the TPE acquisition for a given trial and parameter.

        Args:
            study (optuna.study.Study): The study object.
            trial_number (int): The trial number to plot.
            param_name (str): The parameter name to plot.
        Returns:
            plt.Figure: The matplotlib figure containing the plot.
        """

        trials = study.get_trials(deepcopy=False)
        trial_number2trial = {trial.number: trial for trial in trials}
        if trial_number not in self.log_objects:
            raise ValueError(f"Trial number {trial_number} not found in log objects.")
        log_frame = self.log_objects[trial_number]
        trial = trial_number2trial[trial_number]

        trial = trial_number2trial[trial_number]

        dist = trial.distributions[param_name]
        bounds = (dist.low, dist.high)

        X = np.linspace(bounds[0], bounds[1], 10000)

        values_below = np.array(
            [trial_number2trial[t.number].params[param_name] for t in log_frame["below_trials"]]
        )
        values_above = np.array(
            [trial_number2trial[t.number].params[param_name] for t in log_frame["above_trials"]]
        )

        scores_below = np.array(
            [trial_number2trial[t.number].value for t in log_frame["below_trials"]]
        )
        scores_above = np.array(
            [trial_number2trial[t.number].value for t in log_frame["above_trials"]]
        )

        mpe_below: dict[str, _ParzenEstimator] = log_frame["mpe_below"]
        mpe_above: dict[str, _ParzenEstimator] = log_frame["mpe_above"]
        log_below_dist = mpe_below[param_name].log_pdf({param_name: X})
        log_above_dist = mpe_above[param_name].log_pdf({param_name: X})
        below_dist = np.exp(log_below_dist)
        above_dist = np.exp(log_above_dist)
        dist_max = np.max(np.concatenate([below_dist, above_dist]))

        ret = trial.params[param_name]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        ax.set_title(f"TPE (param:'{param_name}', number:{trial_number})")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Value")
        ax.set_xlim(*bounds)

        ax.plot(
            values_above,
            scores_above,
            label="Above",
            marker=".",
            linestyle="none",
            color="red",
            alpha=0.3,
        )
        ax.plot(
            values_below,
            scores_below,
            label="Below",
            marker=".",
            linestyle="none",
            color="blue",
            alpha=0.3,
        )

        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ax2.plot(
            X, above_dist / dist_max, color="red", linestyle=":", alpha=0.5, label="Above dist."
        )
        ax2.plot(
            X, below_dist / dist_max, color="blue", linestyle=":", alpha=0.5, label="Below dist."
        )

        def prob_low_func(x: np.ndarray) -> np.ndarray:
            return 1 / (
                1
                + np.exp(
                    (
                        np.log(len(log_frame["above_trials"]))
                        + mpe_above[param_name].log_pdf({param_name: x})
                    )
                    - (
                        np.log(len(log_frame["below_trials"]))
                        + mpe_below[param_name].log_pdf({param_name: x})
                    )
                )
            )

        prob_low = prob_low_func(X)
        ax2.plot(X, prob_low, color="green", label="Prob. of below")
        ax2.plot(
            [ret],
            prob_low_func(np.array([ret])),
            marker=".",
            linestyle="none",
            color="red",
            label="ret",
            markersize=10,
        )

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")

        ax.grid()
        return fig
