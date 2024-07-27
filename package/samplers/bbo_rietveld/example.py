import os

import matplotlib.pyplot as plt
import optuna
import optunahub


if __name__ == "__main__":
    """
    To execute following example, please make directory `Y2O3_data` in the same directory of this code and copy `Y2O3.cif`, `Y2O3.csv`, `INST_XRY.PRM` from https://github.com/quantumbeam/BBO-Rietveld/tree/master/data/Y2O3 into `Y2O3_data`.
    """
    bbo_rietveld = optunahub.load_module(
        package="samplers/bbo_rietveld",
    )

    STUDY_NAME = "Y2O3"

    config = bbo_rietveld.ProjectConfig(
        random_seed=1024,
        work_dir="work/" + STUDY_NAME,
        data_dir="Y2O3_data/",
        cif_file="Y2O3.cif",
        powder_histogram_file="Y2O3.csv",
        instrument_parameter_file="INST_XRY.PRM",
        two_theta_lower=15,
        two_theta_upper=150,
        two_theta_margin=20,
        validate_Uiso_nonnegative=True,
    )

    objective_fn = bbo_rietveld.create_objective(config=config)
    os.makedirs(config.work_dir, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        sampler=bbo_rietveld.BboRietveldSampler(seed=config.random_seed, n_startup_trials=10),
    )

    study.optimize(objective_fn, n_trials=100)
    print(f"\n### best params:\n{study.best_params}")
    fig = bbo_rietveld.rietveld_plot(
        config=config, gpx_path=study.best_trial.user_attrs["gpx_path"]
    )
    plt.show()
