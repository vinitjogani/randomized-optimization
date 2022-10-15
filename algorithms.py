import numpy as np
from problems import *
import bitstrings
import random
import matplotlib.pyplot as plt
import unionfind
from collections import defaultdict


class Logger:
    def __init__(self):
        self.steps = []
        self.fitness = []
        self.evaluations = []
        self.evals = 0
        self.last_print = 0

    def log(self, step, fitness, evals):
        self.steps.append(step)
        self.fitness.append(fitness)
        self.last_print += evals
        self.evals += evals
        self.evaluations.append(self.evals)

        if self.last_print > 25_00:
            self.last_print = 0
            print(
                len(self.steps),
                "\t",
                self.steps[-1],
                "\t",
                self.fitness[-1],
                "\t",
                self.evaluations[-1],
            )


class Algorithm:
    def __init__(self, problem):
        self.problem = problem
        self.steps = 1
        self.logger = Logger()

        self.x = problem.sample()
        self.f = self.problem.fitness(self.x)
        self.logger.log(self.steps, self.f, 1)

        self.x_best, self.f_best = self.x, self.f

    def run(self, n_evals, stop_at=None):
        step = 0
        while n_evals >= 0:
            step += 1
            self.x, self.f, evals = self.step()

            if self.f > self.f_best:
                self.x_best, self.f_best = self.x, self.f

            self.logger.log(step, self.f_best, evals)
            n_evals -= evals

            if evals == 0 or self.f_best == stop_at:
                break
        return self.x_best, self.f_best, self.logger

    def step(self):
        raise NotImplementedError()


class HillClimbing(Algorithm):
    def __init__(self, max_attempts, soft, problem):
        super().__init__(problem)
        self.max_attempts = max_attempts
        self.soft = soft

    def step(self):
        attempts = 0

        while attempts < self.max_attempts:
            attempts += 1
            next_state = self.problem.sample_neighbor(self.x)
            next_fitness = self.problem.fitness(next_state)

            if (next_fitness == self.f and self.soft) or (next_fitness > self.f):
                x = next_state
                f = next_fitness
                break

        if attempts == self.max_attempts:
            # Local minima, random restart
            x = self.problem.sample()
            f = self.problem.fitness(x)

        return x, f, attempts


class Annealing(Algorithm):
    def __init__(self, T, decay, min, max_attempts, problem):
        super().__init__(problem)
        self.T = T
        self.decay = decay
        self.min = min
        self.max_attempts = max_attempts

    def step(self):
        evals = 0
        while True:
            x = self.problem.sample_neighbor(self.x)
            f = self.problem.fitness(x)
            evals += 1

            if f > self.f or evals > self.max_attempts:
                break

            prob = np.exp((f - self.f) / self.T)
            if np.random.random() < prob:
                break
        self.T = max(self.T * self.decay, self.min)
        return x, f, evals


class Genetic(Algorithm):
    def __init__(self, K, keep_pct, combine_fn, problem):
        super().__init__(problem)
        self.keep_pct = keep_pct
        self.combine_fn = combine_fn
        self.P = [problem.sample() for _ in range(K)]
        self.F = list(map(self.problem.fitness, self.P))
        self.logger.evals += len(self.F) - 1

    def step(self):
        P = np.array(self.P)
        F = np.array(self.F)
        n = int(len(P) * self.keep_pct)
        N = len(P) - n

        top = np.argsort(-F)[:n]
        P = list(P[top])
        F = list(F[top])

        new = [
            self.combine_fn(
                np.random.choice(P),
                np.random.choice(P),
            )
            for _ in range(N)
        ]
        P.extend(new)
        F.extend(map(self.problem.fitness, new))
        self.P, self.F = P, F

        best = np.argmax(F)
        return P[best], F[best], N


class Mimic(Algorithm):
    def __init__(self, K, keep_pct, clip, problem):
        super().__init__(problem)
        self.K = K
        self.keep_pct = keep_pct
        self.clip = clip

        # Initialize uniform distribution
        N = len(problem.sample())
        self.prob = {}
        self.parent = {}
        for i in range(N):
            self.prob[i] = 0.5
            self.parent[i] = None

    def print_tree(self, k=0, tab=0):
        if tab == 0:
            self.children = defaultdict(list)
            for k_, v in self.parent.items():
                self.children[v].append(k_)

        print("-" * tab + f"{k}")
        for x in self.children[k]:
            self.print_tree(x, tab + 1)

    def sample(self):
        samples = {}
        queue = list(self.parent)
        while queue:
            i = queue.pop(0)
            p = self.parent[i]
            if p is not None and p not in samples:
                queue.append(i)
                continue

            if p is None:
                prob = self.prob[i]
            else:
                prob = self.prob[i][samples[p]]
            s = np.random.random() < np.clip(prob, self.clip, 1 - self.clip)
            samples[i] = int(s)

        out = ""
        for i in range(len(self.parent)):
            out += "1" if samples[i] == 1 else "0"
        return out

    def sample_many(self):
        P = np.array([self.sample() for _ in range(self.K)])
        F = np.array(list(map(self.problem.fitness, P)))
        return P, F

    def keep_top(self, P, F):
        N = len(P)
        keep = int(N * self.keep_pct)
        best = np.argsort(-F)[:keep]
        return P[best], F[best]

    def entropy(self, p):
        entropy = 0
        if p > 0:
            entropy -= p * np.log2(p)
        if p < 1:
            entropy -= (1 - p) * np.log2(1 - p)
        return entropy

    def conditional_prob(self, v1, v2):
        return [v1[v2 == 0].mean(), v1[v2 == 1].mean()]

    def conditional_entropy(self, v1, v2):
        entropy = 0
        for i, p1 in enumerate(self.conditional_prob(v1, v2)):
            p2 = (v2 == i).mean()
            entropy += p2 * self.entropy(p1)
        return entropy

    def mutual_information(self, v1, v2):
        return self.entropy(v1.mean()) - self.conditional_entropy(v1, v2)

    def information_matrix(self, P):
        P = np.array([[int(p[i]) for i in range(len(p))] for p in P])
        I = []
        for i in range(P.shape[1]):
            for j in range(P.shape[1]):
                if j > i:
                    continue
                mi = self.mutual_information(P[:, i], P[:, j])
                I.append((-mi, (i, j)))
        return P, I

    def mst(self, I):
        N = len(self.parent)
        uf = unionfind.unionfind(N)
        chosen = []
        I.sort()

        while len(chosen) < N - 1:
            _, (i, j) = I.pop(0)
            if uf.issame(i, j):
                continue

            uf.unite(i, j)
            chosen.append((i, j))

        return chosen

    def estimate_distribution(self, P, I):
        chosen = self.mst(I)

        adj = defaultdict(list)
        for i, j in chosen:
            adj[i].append(j)
            adj[j].append(i)

        visited = {}

        def root(curr=0):
            if curr in visited:
                return
            visited[curr] = True
            for a in adj[curr]:
                if a in visited:
                    continue
                self.parent[a] = curr
                self.prob[a] = self.conditional_prob(P[:, a], P[:, curr])
                root(a)

        self.parent, self.prob = {}, {}
        root(0)

        for i in range(len(chosen) + 1):
            if i in self.parent:
                continue
            self.parent[i] = None
            self.prob[i] = P[:, i].mean()

        for i in self.parent:
            if self.parent[i] is not None:
                assert i != self.parent[self.parent[i]]

    def step(self):
        P, F = self.sample_many()
        topP, topF = self.keep_top(P, F)
        x, f, evals = topP[0], topF[0], len(F)
        P, I = self.information_matrix(topP)
        self.estimate_distribution(P, I)
        return x, f, evals


if __name__ == "__main__":
    np.random.seed(1234)
    random.seed(1234)
    problem = FlipFlop(60)

    n_evals = 1_500_0

    print("Hill Climbing")
    np.random.seed(0)
    random.seed(42)
    algo = HillClimbing(100, True, problem)
    x_best, f_best, hc = algo.run(n_evals, problem.max())
    print(x_best, f_best / problem.max(), hc.evals)
    print()

    print("Annealing")
    np.random.seed(0)
    random.seed(42)
    algo = Annealing(1, 0.99, 0.05, 100, problem)
    x_best, f_best, annealing = algo.run(n_evals, problem.max())
    print(x_best, f_best / problem.max(), annealing.evals)
    print()

    print("Genetic")
    np.random.seed(0)
    random.seed(42)
    algo = Genetic(10, 0.4, bitstrings.cut_combine, problem)
    x_best, f_best, genetic = algo.run(n_evals, problem.max())
    print(x_best, f_best / problem.max(), genetic.evals)
    print()

    print("MIMIC")
    np.random.seed(0)
    random.seed(42)
    algo = Mimic(500, 0.25, 0.01, problem)
    x_best, f_best, mimic = algo.run(n_evals, problem.max())
    print(x_best, f_best / problem.max(), mimic.evals)
    print()

    plt.figure()
    plt.plot(hc.evaluations, hc.fitness, label="Hill Climbing")
    plt.plot(annealing.evaluations, annealing.fitness, label="Annealing")
    plt.plot(genetic.evaluations, genetic.fitness, label="Genetic")
    plt.plot(mimic.evaluations, mimic.fitness, label="MIMIC")
    plt.legend()
    plt.xscale("log", base=2)
    # plt.xlim(1, 2**12)
    plt.tight_layout()
    plt.savefig("by_iterations.png")

    plt.figure()
    plt.plot(hc.steps, hc.fitness, label="Hill Climbing")
    plt.plot(annealing.steps, annealing.fitness, label="Annealing")
    plt.plot(genetic.steps, genetic.fitness, label="Genetic")
    plt.plot(mimic.steps, mimic.fitness, label="MIMIC")
    plt.legend()
    plt.xscale("log", base=2)
    # plt.xlim(, 2**12)
    plt.tight_layout()
    plt.savefig("by_steps.png")
