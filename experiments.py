import json
import os
from problems import *
from algorithms import *
import numpy as np
import matplotlib.pyplot as plt

PROBLEM_SIZES = {
    Reverse: (32, 32 * 2, 32 * 3, 32 * 4),
    TreasureHunt: (256, 256 * 2, 256 * 3, 256 * 4),
    FlipFlop: (32, 32 * 2, 32 * 3, 32 * 4),
}


def vary_parameter(algo_cls, kwargs, key, values):
    max_steps = 12_000
    readings = []
    for v in values:
        np.random.seed(0)
        random.seed(42)
        algo = algo_cls(**kwargs, **{key: v})
        best = kwargs["problem"].max()
        _, f_best, curve = algo.run(max_steps, best)
        readings.append(curve.evals)
    return readings


def algo_by_params(algo_cls, kwargs, key, values):
    algo_cls = HillClimbing
    readings = {}
    for prob_cls in PROBLEM_SIZES:
        best_values = []
        best_fitness = []
        for size in PROBLEM_SIZES[prob_cls]:
            prob = prob_cls(size)
            kwargs.update({"problem": prob})
            r = vary_parameter(algo_cls, kwargs, key, values)
            best = np.argmin(r)
            best_values.append(values[best])
            best_fitness.append(r[best])

        readings[prob_cls.__name__] = (
            PROBLEM_SIZES[prob_cls],
            best_values,
            best_fitness,
        )
    return readings


def plot_all(readings):
    keys = ["Reverse", "TreasureHunt", "FlipFlop"]
    fig, ax = plt.subplots(
        ncols=3,
        nrows=2,
        figsize=(15, 5),
    )
    for i, k in enumerate(keys):
        x, y1, y2 = readings[k]
        ax[0, i].plot(x, y1)
        ax[1, i].plot(x, y2)
        ax[1, i].set_xlabel(f"{k}: Problem Size")
    return fig, ax


def run_experiment(title, ylabel, key, *args, **kwargs):
    path = f"readings/{key}"
    if os.path.exists(path + ".json"):
        readings = json.load(open(path + ".json", "r"))
    else:
        readings = algo_by_params(*args, **kwargs)
        json.dump(readings, open(path + ".json", "w"))

    fig, ax = plot_all(readings)
    ax[0, 0].set_ylabel(ylabel)
    ax[1, 0].set_ylabel("Fitness")
    ax[0, 1].set_title(title)
    fig.tight_layout()
    fig.savefig(path + ".png")


if __name__ == "__main__":
    plt.style.use("ggplot")

    run_experiment(
        "Hill Climbing: max_attempts vs problem_size",
        "Optimal max_attempts",
        "hill_climbing_attempts_by_size",
        HillClimbing,
        {"soft": False},
        "max_attempts",
        values=[20, 40, 60, 80, 100],
    )

    run_experiment(
        "Soft Hill Climbing: max_attempts vs problem_size",
        "Optimal max_attempts",
        "soft_hill_climbing_attempts_by_size",
        HillClimbing,
        {"soft": True},
        "max_attempts",
        values=[20, 40, 60, 80, 100],
    )
