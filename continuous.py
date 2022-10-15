from datasets import load_term_deposits
import numpy as np
import mlrose_hiive as mlrose
import pickle
import time


def train_neuralnet(algo, X_train, y_train, key, values, problem_size):
    N = X_train.shape[0]
    sample = np.random.randint(0, N, size=int(N * problem_size))
    X_train, y_train = X_train[sample], y_train[sample]
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
        path = f"neuralnets/{problem_size}_{algo}_{key}_{str(v)}_{elapsed}.pkl"
        pickle.dump(nn, open(path, "wb"))


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_term_deposits()

    for problem_size in [1.0, 0.8, 0.6, 0.4, 0.2]:
        train_neuralnet(
            "random_hill_climb",
            X_train,
            y_train,
            "restarts",
            [1, 2, 5, 10],
            problem_size,
        )

        train_neuralnet(
            "simulated_annealing",
            X_train,
            y_train,
            "schedule",
            [
                mlrose.GeomDecay(0.5),
                mlrose.GeomDecay(1),
                mlrose.GeomDecay(2),
                mlrose.GeomDecay(3),
            ],
            problem_size,
        )

        train_neuralnet(
            "genetic_alg",
            X_train,
            y_train,
            "pop_size",
            [20, 50, 100, 200],
            problem_size,
        )

        train_neuralnet(
            "gradient_descent",
            X_train,
            y_train,
            "pop_size",
            [10],
            problem_size,
        )
