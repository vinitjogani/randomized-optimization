from math import ceil
import bitstrings
import numpy as np
import mlrose_hiive as mlrose


class BitStringProblem:
    def __init__(self, k):
        self.k = k

    def max(self):
        return 1e6

    def neighbors(self, x):
        return bitstrings.neighbors(x)

    def sample_neighbor(self, x):
        return bitstrings.random_flip(x)

    def sample(self):
        return bitstrings.random_generate(self.k)


class SixPeaks(BitStringProblem):
    def __init__(self, k):
        super().__init__(k)
        self.fn = mlrose.SixPeaks()

    def max(self):
        n = len(self.sample())
        t = ceil(n * 0.1)
        return 2 * n - (t + 1)

    def fitness(self, x):
        return self.fn.evaluate(np.array(list(map(int, list(x)))))


class TreasureHunt(BitStringProblem):
    def __init__(self, k):
        super().__init__(k)
        t_pct = 0.1
        n = len(self.sample())
        self.t = np.random.choice(list(range(n)), size=int(n * t_pct), replace=False)
        self.keys = np.random.randint(0, 2, size=len(self.t))

    def max(self):
        return len(self.t)

    def fitness(self, x):
        out = 0
        for i, t in enumerate(self.t):
            if self.keys[i] == int(x[t]):
                out += 1
        return out


class FlipFlop(BitStringProblem):
    def __init__(self, k):
        super().__init__(k)
        self.fn = mlrose.SixPeaks()

    def fitness(self, x):
        out = 0
        for i, c in enumerate(x[1:]):
            if c != x[i]:
                out += 1
        return out

    def max(self):
        return len(self.sample()) - 1
