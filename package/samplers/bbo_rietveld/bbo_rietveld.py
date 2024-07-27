from dataclasses import dataclass
from logging import getLogger
from logging import INFO
from logging import StreamHandler
import os
from typing import Callable
from typing import Optional
from typing import Sequence

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.distributions import CategoricalChoiceType
from optuna.trial import FrozenTrial


# setup logger
_logger = getLogger(__name__)
_logger.setLevel(INFO)
_handler = StreamHandler()
_logger.addHandler(_handler)


@dataclass
class ProjectConfig:
    random_seed: int
    work_dir: str
    data_dir: str
    cif_file: str
    powder_histogram_file: str
    instrument_parameter_file: str
    two_theta_lower: float
    two_theta_upper: float
    two_theta_margin: float = 20
    validate_Uiso_nonnegative: bool = True


class Project:
    def __init__(self, config: ProjectConfig, trial_number: int):
        import GSASIIscriptable as G2sc

        self.gpx_path = os.path.join(
            config.work_dir, f"project_seed{config.random_seed}_trial_{trial_number}.gpx"
        )
        self.gpx = G2sc.G2Project(newgpx=self.gpx_path)

        self.hist1 = self.gpx.add_powder_histogram(
            os.path.join(config.data_dir, config.powder_histogram_file),
            os.path.join(config.data_dir, config.instrument_parameter_file),
        )

        self.phase0 = self.gpx.add_phase(
            os.path.join(config.data_dir, config.cif_file),
            phasename=config.cif_file.split(".cif")[0],
            histograms=[self.hist1],
        )

        self.hist1.data["Instrument Parameters"][0]["I(L2)/I(L1)"] = [0.5, 0.5, 0]

        # Set to use iso
        for val in self.phase0.data["Atoms"]:
            val[9] = "I"

    def refine_and_calc_Rwp(self, param_dict: dict) -> float:
        self.gpx.do_refinements([param_dict])
        for hist in self.gpx.histograms():
            Rwp = hist.get_wR()
        return Rwp


def create_objective(config: ProjectConfig) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        """
        objective function for Optuna.

        You can apply black-box optimization for other data analysis
        by modifying this function for that task.

        Parameters
        ----------
        trial : optuna.trial object

        Returns
        -------
        Rwp : float

        """

        ### define search space ###
        # Limits (acute angle)
        measured_2theta_lower = config.two_theta_lower
        measured_2theta_upper = config.two_theta_upper

        two_theta_margin = config.two_theta_margin  # 20 default

        limits_lb = trial.suggest_float(
            "Limits lower bound", measured_2theta_lower, measured_2theta_upper - two_theta_margin
        )
        limits_ub = trial.suggest_float(
            "Limits upper bound", limits_lb + two_theta_margin, measured_2theta_upper
        )
        limits_refine = trial.suggest_categorical("limits refine", [True, False])
        refdict0 = {"set": {"Limits": [limits_lb, limits_ub]}, "refine": limits_refine}

        # Background
        background_type = trial.suggest_categorical(
            "Background type",
            [
                "chebyschev",
                "cosine",
                "Q^2 power series",
                "Q^-2 power series",
                "lin interpolate",
                "inv interpolate",
                "log interpolate",
            ],
        )
        no_coeffs = trial.suggest_int("Number of coefficients", 1, 15)
        background_refine = trial.suggest_categorical("Background refine", [True, False])
        refdict0bg_h = {
            "set": {
                "Background": {
                    "type": background_type,
                    "no. coeffs": no_coeffs,
                    "refine": background_refine,
                }
            }
        }

        # Instrument parameters
        instrument_parameters1_refine = []
        for p in ["Zero"]:
            if trial.suggest_categorical("Instrument_parameters refine %s" % (p), [True, False]):
                instrument_parameters1_refine.append(p)
        refdict1_h = {
            "set": {"Cell": True, "Instrument Parameters": instrument_parameters1_refine}
        }

        sample_parameters1_refine = []
        for p in ["DisplaceX", "DisplaceY", "Scale"]:
            if trial.suggest_categorical("Sample_parameters refine %s" % (p), [True, False]):
                sample_parameters1_refine.append(p)
        refdict1_h2 = {"set": {"Sample Parameters": sample_parameters1_refine}}

        instrument_parameters2_refine = []
        for p in ["U", "V", "W", "X", "Y", "SH/L"]:
            if trial.suggest_categorical("Peakshape_parameters refine %s" % (p), [True, False]):
                instrument_parameters2_refine.append(p)
        refdict2_h = {"set": {"Instrument Parameters": instrument_parameters2_refine}}

        refdict3_h = {"set": {"Atoms": {"all": "XU"}}}

        # Limits (wide angle)
        refdict_fin_h = {
            "set": {"Limits": [measured_2theta_lower, measured_2theta_upper]},
            "refine": True,
        }

        # Evaluate
        refine_params_list = [
            refdict0,
            refdict0bg_h,
            refdict1_h,
            refdict1_h2,
            refdict2_h,
            refdict3_h,
            refdict_fin_h,
        ]

        def evaluate(
            config: ProjectConfig, trial: optuna.Trial, refine_params_list: list[dict]
        ) -> float:
            trial_number = trial.number

            ERROR_PENALTY: float = 1e9
            try:
                _logger.info(config)
                _logger.info(trial_number)
                _logger.info(refine_params_list)

                project = Project(config, trial_number)
                trial.set_user_attr("gpx_path", project.gpx_path)
                for params in refine_params_list:
                    Rwp = project.refine_and_calc_Rwp(params)
                # validate Uiso >= 0
                phase = project.gpx.phases()[0]
                u_iso_list = [atom.uiso for atom in phase.atoms()]
                if config.validate_Uiso_nonnegative and min(u_iso_list) < 0:
                    # Uiso < 0
                    Rwp = ERROR_PENALTY
                return Rwp

            except Exception as e:
                # Refinement failed
                _logger.info(e)

                return ERROR_PENALTY

        Rwp = evaluate(config, trial, refine_params_list)
        return Rwp

    return objective


class BboRietveldSampler(optuna.samplers.TPESampler):
    def __init__(
        self,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] = optuna.samplers._tpe.sampler.default_gamma,
        weights: Callable[[int], np.ndarray] = optuna.samplers._tpe.sampler.default_weights,
        seed: Optional[int] = None,
        *,
        multivariate: bool = True,
        group: bool = True,
        warn_independent_sampling: bool = True,
        constant_liar: bool = False,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
        categorical_distance_func: Optional[
            dict[str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]]
        ] = None,
    ):
        super().__init__(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            weights=weights,
            seed=seed,
            multivariate=multivariate,
            group=group,
            warn_independent_sampling=warn_independent_sampling,
            constant_liar=constant_liar,
            constraints_func=constraints_func,
            categorical_distance_func=categorical_distance_func,
        )


def rietveld_plot(config: ProjectConfig, gpx_path: str) -> matplotlib.figure.Figure:
    import GSASIIscriptable as G2sc

    gpx = G2sc.G2Project(gpx_path)

    hist = gpx.histograms()[0]

    two_theta = hist.getdata("X")
    Y_obs = hist.getdata("Yobs")
    Y_calc = hist.getdata("Ycalc")
    background = hist.getdata("Background")
    residual = hist.getdata("Residual")

    fig = plt.figure()
    gs = GridSpec(5, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:4, :])
    ax2 = fig.add_subplot(gs[4, :])
    fig.subplots_adjust(hspace=0)
    ax1.grid(color="#cccccc")

    ax1.scatter(two_theta, Y_obs, marker="P", lw=0.0001, c="Black", label="XRD (Obs)")
    ax1.plot(two_theta, Y_calc, label="XRD (Calc)")
    ax1.plot(two_theta, background, color="red", label="Background (Calc)")
    ax1.set_ylabel("Intensity")
    ax1.legend()
    ax2.plot(two_theta, residual, color="blue")
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xlabel(r"$2\theta$ (deg.)")
    ax2.set_ylabel("Residual")
    ax1.set_xlim(config.two_theta_lower, config.two_theta_upper)
    ax2.set_xlim(config.two_theta_lower, config.two_theta_upper)
    return fig
