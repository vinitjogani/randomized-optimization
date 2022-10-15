import json
import os
from problems import *
from algorithms import *
from experiments import PROBLEM_SIZES
from collections import defaultdict
import matplotlib.pyplot as plt


def update(key, algo, prob, readings, timings):
    _, f, logs = algo.run(100_000, prob.max())
    print(key, f, logs.evals)
    readings[key][prob.k] = logs.evals
    timings[key][prob.k] = logs.elapsed


def sixpeaks():
    path = "readings/best_sixpeaks.json"
    if os.path.exists(path):
        return json.load(open(path))

    readings = defaultdict(dict)
    timings = defaultdict(dict)

    for p_size in PROBLEM_SIZES[SixPeaks]:
        print("SixPeaks", p_size)
        prob = SixPeaks(p_size)

        np.random.seed(0)
        random.seed(42)
        rhc = HillClimbing(40 if p_size > 20 else 20, True, prob)
        update("RHC", rhc, prob, readings, timings)

        np.random.seed(0)
        random.seed(42)
        sa = Annealing(1 if p_size > 20 else 0.5, 0.999, 0.01, 30, prob)
        update("SA", sa, prob, readings, timings)

        np.random.seed(0)
        random.seed(42)
        ga = Genetic(50, 0.2, bitstrings.splice_combine, prob)
        update("GA", ga, prob, readings, timings)

        np.random.seed(0)
        random.seed(42)
        mimic = Mimic(200, 0.25, 0.02, prob)
        update("MIMIC", mimic, prob, readings, timings)

    json.dump({"readings": readings, "timings": timings}, open(path, "w"))


def treasure():
    path = "readings/best_treasure.json"
    if os.path.exists(path):
        return json.load(open(path))

    readings = defaultdict(dict)
    timings = defaultdict(dict)

    for p_size in PROBLEM_SIZES[TreasureHunt]:
        print("Treasure", p_size)
        prob = TreasureHunt(p_size)

        np.random.seed(0)
        random.seed(42)
        rhc = HillClimbing(20, True, prob)
        update("RHC", rhc, prob, readings, timings)

        np.random.seed(0)
        random.seed(42)
        sa = Annealing(0.5, 0.999, 0.01, 30, prob)
        update("SA", sa, prob, readings, timings)

        np.random.seed(0)
        random.seed(42)
        ga = Genetic(10, 0.2, bitstrings.splice_combine, prob)
        update("GA", ga, prob, readings, timings)

        np.random.seed(0)
        random.seed(42)
        mimic = Mimic(100, 0.4, 0.00, prob)
        update("MIMIC", mimic, prob, readings, timings)

    json.dump({"readings": readings, "timings": timings}, open(path, "w"))


def flipflop():
    path = "readings/best_flipflop.json"
    if os.path.exists(path):
        return json.load(open(path))

    readings = defaultdict(dict)
    timings = defaultdict(dict)

    for p_size in PROBLEM_SIZES[FlipFlop]:
        print("FlipFlop", p_size)
        prob = FlipFlop(p_size)

        np.random.seed(0)
        random.seed(42)
        rhc = HillClimbing(p_size, True, prob)
        update("RHC", rhc, prob, readings, timings)

        np.random.seed(0)
        random.seed(42)
        sa = Annealing(1, 0.999, 0.01, 30, prob)
        update("SA", sa, prob, readings, timings)

        np.random.seed(0)
        random.seed(42)
        ga = Genetic(100, 0.4, bitstrings.cut_combine, prob)
        update("GA", ga, prob, readings, timings)

        np.random.seed(0)
        random.seed(42)
        mimic = Mimic(300 if p_size < 50 else 500, 0.25, 0.02, prob)
        update("MIMIC", mimic, prob, readings, timings)

    json.dump({"readings": readings, "timings": timings}, open(path, "w"))


if __name__ == "__main__":
    sixpeaks()
    treasure()
    flipflop()

    s_ = json.load(open("readings/best_sixpeaks.json"))
    t_ = json.load(open("readings/best_treasure.json"))
    f_ = json.load(open("readings/best_flipflop.json"))
    plt.style.use("ggplot")

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 7))
    axs[0, 1].set_title("Comparative Evaluation")
    for i in range(3):
        axs[1, i].set_yscale("log", base=2)
    for i, x in enumerate(["readings", "timings"]):
        s, t, f = s_[x], t_[x], f_[x]
        ax = axs[i, 0]
        for k in s:
            ax.plot(list(map(int, s[k].keys())), s[k].values(), label=k)
        ax.legend()
        ax.set_xlabel("Six Peaks: Problem Size")
        if x == "readings":
            ax.set_ylabel("Function Evaluations to reach Optima")
        else:
            ax.set_ylabel("Time (s) to reach Optima")

        ax = axs[i, 1]
        for k in t:
            ax.plot(list(map(int, t[k].keys())), t[k].values(), label=k)
        ax.legend()
        ax.set_xlabel("Treasure Hunt: Problem Size")

        ax = axs[i, 2]
        for k in f:
            ax.plot(list(map(int, f[k].keys())), f[k].values(), label=k)
        ax.legend()
        ax.set_xlabel("Flip Flop: Problem Size")
    fig.tight_layout()
    fig.savefig("readings/best.png")
