import json
import os
from problems import *
from algorithms import *
import numpy as np
import matplotlib.pyplot as plt
from joblib import delayed, Parallel

PROBLEM_SIZES = {
    SixPeaks: (20, 20 * 2, 20 * 3, 20 * 4),
    TreasureHunt: (100, 100 * 2, 100 * 3, 100 * 4),
    FlipFlop: (20, 20 * 2, 20 * 3, 20 * 4),
}


def vary_parameter(algo_cls, kwargs, key, values):
    max_steps = 120_000
    readings = []
    best = kwargs["problem"].max()

    def test_parameter(v):
        np.random.seed(0)
        random.seed(42)
        algo = algo_cls(**kwargs, **{key: v})
        _, f_best, curve = algo.run(max_steps, best)
        return (-f_best, curve.evals, v)

    readings = list(
        Parallel(n_jobs=len(values))(delayed(test_parameter)(v) for v in values)
    )
    readings.sort()
    print(readings)
    return readings[0]


def algo_by_params(algo_cls, kwargs, key, values):
    readings = {}
    for prob_cls in PROBLEM_SIZES:
        best_values = []
        best_evals = []
        best_fitness = []
        for size in PROBLEM_SIZES[prob_cls]:
            print(prob_cls.__name__, size)
            np.random.seed(1234)
            random.seed(1234)
            prob = prob_cls(size)
            kwargs.update({"problem": prob})
            f_best, n_evals, value = vary_parameter(algo_cls, kwargs, key, values)
            best_values.append(value)
            best_evals.append(n_evals)
            best_fitness.append(-f_best * 100.0 / prob.max())

        readings[prob_cls.__name__] = (
            PROBLEM_SIZES[prob_cls],
            best_values,
            best_evals,
            best_fitness,
        )
    return readings


def run_or_load_cached(path, **kwargs):
    if os.path.exists(path + ".json"):
        readings = json.load(open(path + ".json", "r"))
    else:
        readings = algo_by_params(**kwargs)
        json.dump(readings, open(path + ".json", "w"))
    return readings


def run_experiment(title, ylabel, chart_key, **kwargs):
    path = f"readings/{chart_key}"
    readings = run_or_load_cached(path, **kwargs)

    keys = ["SixPeaks", "TreasureHunt", "FlipFlop"]
    fig, ax = plt.subplots(
        ncols=3,
        nrows=3,
        figsize=(15, 6),
    )
    for i, k in enumerate(keys):
        x, y1, y2, y3 = readings[k]
        ax[0, i].plot(x, y1)
        ax[1, i].plot(x, y2)
        ax[2, i].plot(x, y3)
        ax[2, i].set_xlabel(f"{k}: Problem Size")
    ax[0, 0].set_ylabel(ylabel)
    ax[1, 0].set_ylabel("Evaluations")
    ax[2, 0].set_ylabel("Fitness (%)")
    ax[0, 1].set_title(title)
    fig.tight_layout()
    fig.savefig(path + ".png")


def run_experiment_with_groups(
    title, ylabel, chart_key, group_key, group_values, **kwargs
):
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(15, 6))
    ax[1, 0].set_ylabel("Evaluations")
    ax[2, 0].set_ylabel("Fitness (%)")
    ax[0, 0].set_ylabel(ylabel)
    ax[0, 1].set_title(title)

    for v in group_values:
        kwargs["kwargs"][group_key] = v
        path = f"readings/{chart_key}_{v}"
        label = f"{group_key}={v}"
        readings = run_or_load_cached(path, **kwargs)

        for i, k in enumerate(["SixPeaks", "TreasureHunt", "FlipFlop"]):
            x, y1, y2, y3 = readings[k]
            ax[0, i].plot(x, y1, label=label)
            ax[1, i].plot(x, y2, label=label)
            ax[2, i].plot(x, y3, label=label)
            ax[2, i].set_xlabel(f"{k}: Problem Size")

    for i in ax.reshape(-1):
        i.legend(loc="upper center", ncol=2)

    path = f"readings/{chart_key}"
    fig.tight_layout()
    fig.savefig(path + ".png")


if __name__ == "__main__":
    plt.style.use("ggplot")

    run_experiment(
        "Hill Climbing: max_attempts vs problem_size",
        "Optimal max_attempts",
        "hill_climbing_attempts_by_size",
        algo_cls=HillClimbing,
        kwargs={"soft": False},
        key="max_attempts",
        values=[20, 40, 60, 80, 100],
    )

    run_experiment(
        "Soft Hill Climbing: max_attempts vs problem_size",
        "Optimal max_attempts",
        "soft_hill_climbing_attempts_by_size",
        algo_cls=HillClimbing,
        kwargs={"soft": True},
        key="max_attempts",
        values=[20, 40, 60, 80, 100],
    )

    run_experiment_with_groups(
        "Simulated Annealing: temperature vs problem_size",
        "Optimal temperature",
        "annealing_temperature_by_size",
        group_key="decay",
        group_values=[0.9, 0.99, 0.999],
        algo_cls=Annealing,
        kwargs={"min": 0.01, "max_attempts": 30},
        key="T",
        values=[0.5, 1, 1.5, 2, 3, 4],
    )

    run_experiment_with_groups(
        "Genetic Algorithm with Uniform Cross-over: population vs problem_size",
        "Optimal population size",
        "ga_population_vs_size",
        algo_cls=Genetic,
        group_key="keep_pct",
        group_values=[0.2, 0.4, 0.6],
        kwargs={"combine_fn": bitstrings.splice_combine},
        key="K",
        values=[10, 25, 50, 100, 200],
    )

    run_experiment_with_groups(
        "Genetic Algorithm with Cut Cross-over: population vs problem_size",
        "Optimal population size",
        "cut_ga_population_vs_size",
        algo_cls=Genetic,
        group_key="keep_pct",
        group_values=[0.2, 0.4, 0.6],
        kwargs={"combine_fn": bitstrings.cut_combine},
        key="K",
        values=[10, 25, 50, 100, 200],
    )

    run_experiment_with_groups(
        "MIMIC: population vs problem_size",
        "Optimal population size",
        "mimic_population_vs_size",
        algo_cls=Mimic,
        group_key="keep_pct",
        group_values=[0.2, 0.4, 0.6],
        kwargs={"clip": 0},
        key="K",
        values=[100, 200, 300, 400],
    )

    run_experiment_with_groups(
        "Clipped MIMIC: population vs problem_size",
        "Optimal population size",
        "mimic_population_vs_size",
        algo_cls=Mimic,
        group_key="keep_pct",
        group_values=[0.2, 0.4, 0.6],
        kwargs={"clip": 0},
        key="K",
        values=[100, 200, 300, 400],
    )

    run_experiment_with_groups(
        "Clipped MIMIC: population vs problem_size",
        "Optimal population size",
        "clipped_mimic_population_vs_size",
        algo_cls=Mimic,
        group_key="keep_pct",
        group_values=[0.2, 0.4, 0.6],
        kwargs={"clip": 0.01},
        key="K",
        values=[100, 200, 300, 400],
    )
