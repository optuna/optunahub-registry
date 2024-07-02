from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import cast
import warnings

import numpy as np
import optuna._gp.search_space as gp_search_space
from optuna._gp.search_space import sample_normalized_params
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState
from pfns import bar_distribution
from pfns import encoders
from pfns import priors
from pfns import utils
from pfns.priors.fast_gp import get_batch
from pfns.scripts.acquisition_functions import optimize_acq_w_lbfgs
from pfns.train import train
import torch


def get_vanilla_gp_config(device: str) -> dict[str, Any]:
    hps = {
        "outputscale": 1.0,
        "lengthscale": 0.1,
        "noise": 1e-4,
    }
    batch = get_batch(100000, 20, 1, hyperparameters=hps)
    ys = batch.target_y.to(device)

    config_vanilla_gp = {
        "priordataloader_class_or_get_batch": priors.fast_gp.get_batch,
        "criterion": bar_distribution.FullSupportBarDistribution(
            bar_distribution.get_bucket_limits(num_outputs=100, ys=ys)
        ),
        "encoder_generator": encoders.get_normalized_uniform_encoder(encoders.Linear),
        "emsize": 256,
        "nhead": 4,
        "nhid": 512,
        "nlayers": 4,
        "y_encoder_generator": encoders.Linear,
        "extra_prior_kwargs_dict": {
            "num_features": 1,
            "fuse_x_y": False,
            "hyperparameters": hps,
        },
        "epochs": 20,
        "warmup_epochs": 5,
        "steps_per_epoch": 100,
        "batch_size": 8,
        "lr": 0.001,
        "seq_len": 20,
        "single_eval_pos_gen": utils.get_uniform_single_eval_pos_sampler(20),
    }

    return config_vanilla_gp


def get_heboplus_config(device: str) -> dict[str, Any]:
    config = {
        "priordataloader_class_or_get_batch": priors.get_batch_to_dataloader(
            priors.get_batch_sequence(
                priors.hebo_prior.get_batch,
                priors.utils.sample_num_feaetures_get_batch,
            )
        ),
        "encoder_generator": encoders.get_normalized_uniform_encoder(
            encoders.get_variable_num_features_encoder(encoders.Linear)
        ),
        "emsize": 512,
        "nhead": 4,
        "warmup_epochs": 5,
        "y_encoder_generator": encoders.Linear,
        "batch_size": 128,
        "scheduler": utils.get_cosine_schedule_with_warmup,
        "extra_prior_kwargs_dict": {
            "num_features": 18,
            "hyperparameters": {
                "lengthscale_concentration": 1.2106559584074301,
                "lengthscale_rate": 1.5212245992840594,
                "outputscale_concentration": 0.8452312502679863,
                "outputscale_rate": 0.3993553245745406,
                "add_linear_kernel": False,
                "power_normalization": False,
                "hebo_warping": False,
                "unused_feature_likelihood": 0.3,
                "observation_noise": True,
            },
        },
        "epochs": 50,
        "lr": 0.0001,
        "seq_len": 60,
        "single_eval_pos_gen": utils.get_uniform_single_eval_pos_sampler(
            50, min_len=1
        ),  # <function utils.get_uniform_single_eval_pos_sampler.<locals>.<lambda>()>,
        "aggregate_k_gradients": 2,
        "nhid": 1024,
        "steps_per_epoch": 1024,
        "weight_decay": 0.0,
        "train_mixed_precision": False,
        "efficient_eval_masking": True,
        "nlayers": 12,
    }

    bs = 128
    all_targets = []
    for num_hps in [
        2,
        8,
        12,
    ]:  # a few different samples in case the number of features makes a difference in y dist
        b = config["priordataloader_class_or_get_batch"].get_batch_method(
            bs,
            1000,
            num_hps,
            epoch=0,
            device=device,
            hyperparameters={
                **config["extra_prior_kwargs_dict"]["hyperparameters"],
                "num_hyperparameter_samples_per_batch": -1,
            },
        )
        all_targets.append(b.target_y.flatten())
    ys = torch.cat(all_targets, 0).cpu()

    config["criterion"] = bar_distribution.FullSupportBarDistribution(
        bar_distribution.get_bucket_limits(1000, ys=ys)
    )

    return config


class PFNs4BOSampler(BaseSampler):
    """A sampler based on the Prior-data Fitted Networks (PFNs) as the surrogate model.

    This sampler is based on the PFNs, which is a neural network-based surrogate model.

    .. note::
        The default prior argument is ``"hebo"``. This trains the PFNs model in the
        init of the sampler. If you want to use a pre-trained model, you can download
        the model checkpoint from the following link:
        https://github.com/automl/PFNs/blob/main/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt
        and load it using the following code:

        .. code-block:: python
            import torch

            model = torch.load("PATH/TO/prior_diff_real_checkpoint_n_0_epoch_42.cpkt")
            sampler = PFNs4BOSampler(prior=model)

    .. note::
        The performance of PFNs4BO with the HEBO+ prior is maximized with the number of
        trials smaller than 100 or 200 in most cases. If you have a large number of trials,
        it is recommended to change the sampler to a random sampler or etc after a certain
        number of trials.

    Args:
        prior:
            A string or a torch.nn.Module object. If a string, it should be one of the following:

            - ``"vanilla gp"``: A vanilla GP model.
            - ``"hebo"``: A model based on the HEBO+ algorithm.

            If a torch.nn.Module object, it should be a trained model.
        model_path:
            A file path to save the trained model. If None, the model will not be saved.
        seed:
            Seed for random number generator.
        independent_sampler:
            A sampler instance for independent sampling. If None, :class:`~optuna.samplers.RandomSampler`
            is used.
        n_startup_trials:
            The number of initial trials that are used to fit the model.
        num_grad_steps:
            The number of gradient steps for optimization.
        num_candidates:
            The number of candidates for optimization.
        pre_sample_size:
            The number of samples for pre-sampling.
        acquisition_function_type:
            The type of acquisition function. It should be one of the following:

            - ``"ei"``: Expected improvement.
            - ``"pi"``: Probability of improvement.
            - ``"ucb"``: Upper confidence bound.
            - ``"ei_or_rand"``: Expected improvement mixed with random sampling.
            - ``"mean"``: Mean of the model.
    """

    def __init__(
        self,
        *,
        prior: str | torch.nn.Module = "hebo",
        model_path: str | None = None,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        num_grad_steps: int = 15_000,
        num_candidates: int = 100,
        pre_sample_size: int = 100_000,
        acquisition_function_type: str = "ei",
    ) -> None:
        self._num_grad_steps = num_grad_steps
        self._num_candidates = num_candidates
        self._pre_sample_size = pre_sample_size
        self._acquisition_function_type = acquisition_function_type

        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or RandomSampler(seed=seed)
        self._intersection_search_space = IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials

        self._device = utils.default_device

        if isinstance(prior, torch.nn.Module):
            trained_model = prior
        elif prior == "vanilla gp":
            _, _, trained_model, _ = train(**get_vanilla_gp_config(self._device))
        elif prior == "hebo":
            _, _, trained_model, _ = train(**get_heboplus_config(self._device))
        else:
            raise ValueError("You should specify `prior` as 'vanilla gp', 'hebo', or a model.")

        self._model = trained_model
        self._model.eval()

        if model_path is not None:
            torch.save(trained_model, model_path)

    def sample_relative(
        self, study: Study, trial: Trial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < self._n_startup_trials:
            return {}

        (
            internal_search_space,
            normalized_params,
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)

        _sign = -1.0 if study.direction == StudyDirection.MINIMIZE else 1.0
        score_vals = np.array([_sign * cast(float, trial.value) for trial in trials])

        if np.any(~np.isfinite(score_vals)):
            warnings.warn(
                "This sampler cannot handle infinite values. "
                "We clamp those values to worst/best finite value."
            )

            finite_score_vals = score_vals[np.isfinite(score_vals)]
            best_finite_score = np.max(finite_score_vals, initial=0.0)
            worst_finite_score = np.min(finite_score_vals, initial=0.0)

            score_vals = np.clip(score_vals, worst_finite_score, best_finite_score)

        standarized_score_vals = (score_vals - score_vals.mean()) / max(1e-10, score_vals.std())

        def rand_sample_func(n: int) -> torch.Tensor:
            xs = sample_normalized_params(n, internal_search_space, None)
            ret = torch.from_numpy(xs).to(torch.float32).to(self._device)
            return ret

        known_x = torch.from_numpy(normalized_params).to(torch.float32).to(self._device)
        known_y = torch.from_numpy(standarized_score_vals).to(torch.float32).to(self._device)

        with torch.enable_grad():
            _, x_options, eis, _, _ = optimize_acq_w_lbfgs(
                model=self._model,
                known_x=known_x,
                known_y=known_y,
                num_grad_steps=self._num_grad_steps,
                num_candidates=self._num_candidates,
                pre_sample_size=self._pre_sample_size,
                device=self._device,
                rand_sample_func=rand_sample_func,
                apply_power_transform=True,
                acq_function=self._acquisition_function_type,
            )

        normalized_param = x_options[torch.argmax(eis)]
        return gp_search_space.get_unnormalized_param(search_space, normalized_param)

    def infer_relative_search_space(
        self, study: Study, trial: Trial
    ) -> dict[str, BaseDistribution]:
        search_space = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def sample_independent(
        self,
        study: Study,
        trial: Trial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
