import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from optuna.samplers.nsgaii import BLXAlphaCrossover
import optuna.storages.journal
import optunahub

from nsgaii_with_initial_trials import NSGAIIwITSampler


file_path = "./journal.log"
lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)

storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock_obj),
)

storage = optuna.storages.InMemoryStorage()


def objective(trial: optuna.Trial) -> tuple[float, float]:
    # ZDT1
    n_variables = 30

    x = np.array([trial.suggest_float(f"x{i}", 0, 1) for i in range(n_variables)])
    g = 1 + 9 * np.sum(x[1:]) / (n_variables - 1)
    f1 = x[0]
    f2 = g * (1 - (f1 / g) ** 0.5)

    return f1, f2


population_size = 50
n_trials = 1000
seed = 42


samplers = [
    NSGAIISampler(
        population_size=population_size,
        seed=seed,
        crossover=BLXAlphaCrossover(),
    ),
    # NSGAIIwITSampler(
    #     population_size=population_size,
    #     seed=seed,
    #     crossover=BLXAlphaCrossover(),
    # ),
    NSGAIIwITSampler(
        population_size=population_size,
        seed=seed,
        crossover=BLXAlphaCrossover(),
    ),
]

studies = []
title = ["NSGAII", "NSGAIIwInitialTrials"]
for i, sampler in enumerate(samplers):
    study = optuna.create_study(
        sampler=sampler,
        study_name=title[i],
        directions=["minimize", "minimize"],
        storage=storage,
    )

    if i == 1:
        study.enqueue_trial(
            {
                "x0": 0,
                "x1": 1,
                "x2": 0,
                "x3": 0,
                "x4": 0,
                "x5": 0,
                "x6": 0,
                "x7": 0,
                "x8": 0,
                "x9": 0,
                "x10": 0,
                "x11": 0,
                "x12": 0,
                "x13": 0,
                "x14": 0,
                "x15": 0,
                "x16": 0,
                "x17": 0,
                "x18": 0,
                "x19": 0,
                "x20": 0,
                "x21": 0,
                "x22": 0,
                "x23": 0,
                "x24": 0,
                "x25": 0,
                "x26": 0,
                "x27": 0,
                "x28": 0,
                "x29": 0,
            }
        )
        study.enqueue_trial(
            {
                "x0": 0.5,
                "x1": 1,
                "x2": 0,
                "x3": 0,
                "x4": 0,
                "x5": 0,
                "x6": 0,
                "x7": 0,
                "x8": 0,
                "x9": 0,
                "x10": 0,
                "x11": 0,
                "x12": 0,
                "x13": 0,
                "x14": 0,
                "x15": 0,
                "x16": 0,
                "x17": 0,
                "x18": 0,
                "x19": 0,
                "x20": 0,
                "x21": 0,
                "x22": 0,
                "x23": 0,
                "x24": 0,
                "x25": 0,
                "x26": 0,
                "x27": 0,
                "x28": 0,
                "x29": 0,
            }
        )
        study.enqueue_trial(
            {
                "x0": 1,
                "x1": 1,
                "x2": 0,
                "x3": 0,
                "x4": 0,
                "x5": 0,
                "x6": 0,
                "x7": 0,
                "x8": 0,
                "x9": 0,
                "x10": 0,
                "x11": 0,
                "x12": 0,
                "x13": 0,
                "x14": 0,
                "x15": 0,
                "x16": 0,
                "x17": 0,
                "x18": 0,
                "x19": 0,
                "x20": 0,
                "x21": 0,
                "x22": 0,
                "x23": 0,
                "x24": 0,
                "x25": 0,
                "x26": 0,
                "x27": 0,
                "x28": 0,
                "x29": 0,
            }
        )

    study.optimize(objective, n_trials=n_trials)
    studies.append(study)

    optuna.visualization.plot_pareto_front(study).show()

sampler1 = optuna.samplers.QMCSampler(seed=seed, qmc_type="halton", scramble=True)
study = optuna.create_study(
    sampler=sampler1,
    study_name="Random+NSGAII",
    directions=["minimize", "minimize"],
    storage=storage,
)
study.optimize(objective, n_trials=2 * n_trials)
sampler2 = NSGAIIwITSampler(
    population_size=population_size,
    seed=seed,
    crossover=BLXAlphaCrossover(),
)
study = optuna.create_study(
    sampler=sampler2,
    study_name="Random+NSGAII",
    directions=["minimize", "minimize"],
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=n_trials // 2)
optuna.visualization.plot_pareto_front(study).show()
studies.append(study)


m = optunahub.load_module("visualization/plot_pareto_front_multi")
fig = m.plot_pareto_front(studies)
fig.show()
