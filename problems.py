from math import ceil
import random

from regex import D
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


class Reverse(BitStringProblem):
    def max(self):
        return len(self.sample())

    def fitness(self, x):
        out = 0
        for i in range(len(x)):
            out += int(x[i] == x[-i - 1])
        return out


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
    def fitness(self, x):
        out = 0
        for i, c in enumerate(x[1:]):
            if c != x[i]:
                out += 1
        return out

    def max(self):
        return len(self.sample()) - 1
