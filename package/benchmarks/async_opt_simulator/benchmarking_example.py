from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import optuna
import optunahub


plot_target_over_time = optunahub.load_module(
    "visualization/plot_target_over_time"
).plot_target_over_time
Problem = optunahub.load_module("benchmarks/hpolib").Problem
AsyncOptBenchmarkSimulator = optunahub.load_module(
    "benchmarks/async_opt_simulator"
).AsyncOptBenchmarkSimulator


def simulate(n_workers: int, constant_liar: bool, seed: int) -> optuna.Study:
    sim = AsyncOptBenchmarkSimulator(n_workers=n_workers)
    problem = Problem(dataset_id=0, metric_names=["val_loss"], seed=0)
    runtime_func = Problem(dataset_id=0, metric_names=["train_time"], seed=0)
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=(n_init := 10),
        constant_liar=constant_liar,
        seed=seed,
    )
    study = optuna.create_study(directions=problem.directions, sampler=sampler)
    sim.optimize(
        study=study, problem=problem, runtime_func=lambda t: runtime_func(t)[0], n_trials=100
    )
    new_study = optuna.create_study()
    new_study.add_trials(study.trials[n_init:])
    return new_study


def plot(ax: plt.Axes, n_workers: int, constant_liar: bool) -> None:
    colors = {1: "black", 2: "blue", 4: "darkred"}
    study_list = []
    for seed in range(10):
        study_list.append(simulate(n_workers, constant_liar, seed=seed))
    plot_target_over_time(
        study_list,
        color=colors[n_workers],
        ax=ax,
        cumtime_func=lambda t: t.user_attrs["cumtime"],
        label=f"$n=${n_workers}, {constant_liar=}",
        marker="*" if constant_liar else "s",
        ls="dotted" if constant_liar else "dashed",
        markevery=10,
    )


if __name__ == "__main__":
    _, ax = plt.subplots()
    for n_workers, constant_liar in itertools.product(*([1, 2, 4], [True, False])):
        plot(ax, n_workers=n_workers, constant_liar=constant_liar)
    ax.legend()
    ax.set_yscale("log")
    plt.savefig("async-bench-example.png", bbox_inches="tight")
