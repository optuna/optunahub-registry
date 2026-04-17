from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optunahub


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["mathtext.fontset"] = "stix"


def simulate(n_workers: int, constant_liar: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sim = optunahub.load_module("benchmarks/async_opt_simulator").AsyncOptBenchmarkSimulator(
        n_workers=n_workers
    )
    Problem = optunahub.load_module("benchmarks/hpolib").Problem
    problem = Problem(dataset_id=0, metric_names=["val_loss"], seed=0)
    runtime_func = Problem(dataset_id=0, metric_names=["train_time"], seed=0)
    n_init = 10
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=n_init,
        constant_liar=constant_liar,
        seed=seed,
    )
    study = optuna.create_study(directions=problem.directions, sampler=sampler)
    sim.optimize(
        study=study, problem=problem, runtime_func=lambda t: runtime_func(t)[0], n_trials=100
    )
    results = sim.get_results_from_study(study)
    return np.array(results["cumtime"])[n_init:], np.array(results["values"])[n_init:]


def get_values_on_fixed_time_steps(
    cumtimes: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    t_min = np.min(cumtimes)
    t_max = np.max(cumtimes)
    ts = np.exp(np.linspace(np.log(t_min), np.log(t_max)))
    v_on_grid = []
    for ct, v in zip(cumtimes, values):
        i_upper = np.minimum(np.searchsorted(ct, ts, side="left"), v.size - 1)
        v_on_grid.append(v[i_upper])
    return ts, np.array(v_on_grid)


def plot(ax: plt.Axes, n_workers: int, constant_liar: bool) -> None:
    colors = {1: "black", 2: "blue", 4: "darkred"}
    markers = {True: "*", False: "s"}
    linestyles = {True: "dotted", False: "dashed"}
    label = f"$n=${n_workers}, {constant_liar=}"
    color = colors[n_workers]
    marker = markers[constant_liar]
    ls = linestyles[constant_liar]
    cumtime_list = []
    value_list = []
    for seed in range(2):
        cumtime, values = simulate(n_workers, constant_liar, seed=seed)
        cumtime_list.append(cumtime)
        value_list.append(np.minimum.accumulate(values[:, 0]))
    ts, vs = get_values_on_fixed_time_steps(np.array(cumtime_list), np.array(value_list))
    m = np.mean(vs, axis=0)
    s = np.std(vs, axis=0) / np.sqrt(len(vs))
    ax.plot(ts, m, color=color, marker=marker, markevery=10, ls=ls, label=label)
    ax.fill_between(ts, m - s, m + s, color=color, alpha=0.2)


if __name__ == "__main__":
    _, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")
    ax.set_xlabel("Wallclock Time [s]")
    ax.set_ylabel("Validation Loss")
    for n_workers, constant_liar in itertools.product(*([1, 2, 4], [True, False])):
        plot(ax, n_workers=n_workers, constant_liar=constant_liar)

    ax.legend()
    plt.savefig("async-bench-example.png", bbox_inches="tight")
