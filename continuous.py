import json
from datasets import load_term_deposits
import numpy as np
import mlrose_hiive as mlrose
import pickle
import time
from sklearn.metrics import auc, precision_recall_curve
import glob
import matplotlib.pyplot as plt


def plot_key(algo, key, ax, evals=False):
    nets = glob.glob(f"neuralnets/1.0_{algo}_{key}_*")
    best = None
    for n in nets:
        nn = pickle.load(open(n, "rb"))
        v = n.replace(f"neuralnets/1.0_{algo}_{key}_", "").split("_")[0]
        k = key.replace("schedule", "temperature")
        if best is None or nn.fitness_curve[:, 0].min() < best:
            best = nn.fitness_curve[:, 0].min()
        nn.fitness_curve[:, 1] -= nn.fitness_curve[0, 1]

        if evals:
            ax.plot(nn.fitness_curve[:, 1], nn.fitness_curve[:, 0], label=f"{k}={v}")
        else:
            ax.plot(nn.fitness_curve[:, 0], label=f"{k}={v}")
    ax.legend()
    return best


def pr_auc_score(y_true, y_score):
    aucs = []
    counts = []
    for class_ in range(2):
        if class_ == 1:
            t, s = y_true, y_score
        else:
            t, s = 1 - y_true, 1 - y_score
        precision, recall, _ = precision_recall_curve(t, s)
        aucs.append(auc(recall, precision))
        counts.append((y_true == class_).sum())

    return sum(aucs) / len(aucs)


def train_neuralnet(algo, X_train, y_train, key, values):
    for v in values:
        nn = mlrose.NeuralNetwork(
            hidden_nodes=(256, 128),
            activation="relu",
            algorithm=algo,
            max_iters=100,
            max_attempts=20,
            early_stopping=True,
            curve=True,
            random_state=0,
            learning_rate=0.5 if algo != "gradient_descent" else 0.1,
            **{key: v},
        )
        start = time.time()
        nn.fit(X_train, y_train)
        end = time.time()
        elapsed = round(end - start, 2)
        print(algo, key, v, elapsed)
        path = f"neuralnets/{1.0}_{algo}_{key}_{str(v)}_{elapsed}.pkl"
        pickle.dump(nn, open(path, "wb"))


def eval_neuralnet(dataset, kwargs):
    (X_train, y_train), (X_test, y_test) = dataset
    out = {}
    for problem_size in [0.2, 0.4, 0.6, 0.8]:
        N = X_train.shape[0]
        sample = np.random.randint(0, N, size=int(N * problem_size))
        X, y = X_train[sample], y_train.iloc[sample]
        nn = mlrose.NeuralNetwork(
            hidden_nodes=(256, 128),
            activation="relu",
            max_iters=100,
            max_attempts=20,
            early_stopping=True,
            curve=True,
            random_state=0,
            learning_rate=0.5,
            **kwargs,
        )
        start = time.time()
        nn.fit(X, y)
        end = time.time()
        elapsed = round(end - start, 2)

        nn.predict(X)
        p = nn.predicted_probs.reshape(-1)
        train_auprc = pr_auc_score(y, p)

        nn.predict(X_test)
        p = nn.predicted_probs.reshape(-1)
        test_loss = (y_test * np.log(p) + (1 - y_test) * np.log(p)).mean()
        test_auprc = pr_auc_score(y_test, p)

        out[problem_size] = {
            "train_time": elapsed,
            "train_loss": nn.loss,
            "test_loss": -test_loss,
            "train_auprc": train_auprc,
            "test_auprc": test_auprc,
        }
    return out


def plot_iters_and_evals():
    fig, axs = plt.subplots(ncols=4, figsize=(15, 4))

    plot_key("random_hill_climb", "restarts", axs[0])
    axs[0].set_xlabel("RHC: Iterations")
    axs[0].set_ylabel("Loss")

    plot_key("simulated_annealing", "schedule", axs[1])
    axs[1].set_xlabel("SA: Iterations")

    plot_key("genetic_alg", "pop_size", axs[2])
    axs[2].set_xlabel("GA: Iterations")

    nn = pickle.load(open("neuralnets/gradient_descent_pop_size_10_68.6.pkl", "rb"))
    axs[-1].plot(-nn.fitness_curve)
    axs[-1].set_xlabel("Gradient Descent: Iterations")

    fig.tight_layout()
    fig.savefig("readings/cont_iterations.png")

    fig, axs = plt.subplots(ncols=3, figsize=(11.5, 4))

    plot_key("random_hill_climb", "restarts", axs[0], True)
    axs[0].set_xlabel("RHC: Evaluations")
    axs[0].set_ylabel("Loss")

    plot_key("simulated_annealing", "schedule", axs[1], True)
    axs[1].set_xlabel("SA: Evaluations")

    plot_key("genetic_alg", "pop_size", axs[2], True)
    axs[2].set_xlabel("GA: Evaluations")

    fig.tight_layout()
    fig.savefig("readings/cont_evals.png")


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_term_deposits()

    # train_neuralnet(
    #     "random_hill_climb",
    #     X_train,
    #     y_train,
    #     "restarts",
    #     [5, 10, 15, 20],
    # )

    # train_neuralnet(
    #     "simulated_annealing",
    #     X_train,
    #     y_train,
    #     "schedule",
    #     [
    #         mlrose.GeomDecay(2),
    #         mlrose.GeomDecay(4),
    #         mlrose.GeomDecay(6),
    #         mlrose.GeomDecay(8),
    #     ],
    # )

    # train_neuralnet(
    #     "genetic_alg",
    #     X_train,
    #     y_train,
    #     "pop_size",
    #     [20, 50, 100, 200],
    # )

    # train_neuralnet(
    #     "gradient_descent",
    #     X_train,
    #     y_train,
    #     "pop_size",
    #     [10],
    # )

    # plot_iters_and_evals()

    dataset = load_term_deposits()
    outputs = {
        "RHC": eval_neuralnet(
            dataset,
            dict(algorithm="random_hill_climb", restarts=5),
        ),
        "SA": eval_neuralnet(
            dataset,
            dict(algorithm="simulated_annealing", schedule=mlrose.GeomDecay(8)),
        ),
        "GA": eval_neuralnet(
            dataset,
            dict(algorithm="genetic_alg", pop_size=50),
        ),
        "GD": eval_neuralnet(
            dataset,
            dict(algorithm="gradient_descent"),
        ),
    }
    json.dump(outputs, open("readings/cont.json", "w"))
