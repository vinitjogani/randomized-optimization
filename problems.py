import random
import bitstrings
import numpy as np
import mlrose_hiive as mlrose


class Problem1:
    def __init__(self):
        self.key = bitstrings.random_generate()
        self.key_int = int(self.key, 2)
        self.fn = mlrose.SixPeaks()

    def max(self):
        n = len(self.sample())
        t = int(n * 0.1)
        return 2 * n - (t + 1)

    def fitness(self, x):
        state = np.array(list(map(int, x)))
        return self.fn.evaluate(state)
        state = x

        def head(i, state):
            out = 0
            for c in state:
                if c == str(i):
                    out += 1
                else:
                    break
            return out

        def tail(i, state):
            return head(i, state[::-1])

        _n = len(state)
        _t = np.ceil(0.1 * _n)

        # Calculate head and tail values
        head_0 = head(0, state)
        tail_0 = tail(0, state)
        head_1 = head(1, state)
        tail_1 = tail(1, state)

        # Calculate R(X, T)
        _r = 0
        _max_score = max(tail_0, head_1)
        if tail_0 > _t and head_1 > _t:
            _r = _n
        elif tail_1 > _t and head_0 > _t:
            _r = _n
            _max_score = max(tail_1, head_0)

        # Evaluate function
        return _max_score + _r

        score = 0
        for i in range(len(x) // 2):
            if x[i] != x[-i - 1]:
                score -= 1
            # score += int(s[0]) ^ int(s[1]) == int(s[2])
            # score += int(s[3])
            # score += (int(s[0]) ^ int(s[1])) == int(s[2])
            # score += (int(s[2]) ^ int(s[3])) == int(s[3])
        return score
        # return sum(a == b for a, b in zip(self.key, x))
        # x_int = int(x, 2)
        # xor = x_int ^ self.key_int
        # return xor

    def neighbors(self, x):
        return bitstrings.neighbors(x)

    def sample_neighbor(self, x):
        return bitstrings.random_flip(x)

    def sample(self):
        return bitstrings.random_generate()


class Problem2:
    def __init__(self):
        size = len(self.sample())
        self.weights = np.random.random(size)
        self.values = np.random.random(size)

    def fitness(self, x):
        w = np.array(list(map(int, list(x))))
        if np.dot(w, self.weights) <= 16:
            return np.dot(w, self.values)
        return 0

    def neighbors(self, x):
        return bitstrings.neighbors(x)

    def sample_neighbor(self, x):
        return bitstrings.random_flip(x)

    def sample(self):
        return bitstrings.random_generate()


class FlipFlop:
    def fitness(self, x):
        out = 0
        for i, c in enumerate(x[1:]):
            if c != x[i]:
                out += 1
        return out

    def max(self):
        return len(self.sample()) - 1

    def neighbors(self, x):
        return bitstrings.neighbors(x)

    def sample_neighbor(self, x):
        return bitstrings.random_flip(x)

    def sample(self):
        return bitstrings.random_generate()


class Problem4:
    def __init__(self):
        N = len(self.sample())
        self.p1 = np.random.random(size=N)
        self.p2 = np.random.random(size=N)

    def fitness(self, x):
        x = list(map(int, list(x)))
        p_ = 0
        ps = [self.p1, self.p2]
        for i, _ in enumerate(x[1:]):
            parent = x[i // 2]
            p_ += np.log(ps[parent][i])
        return p_

    def neighbors(self, x):
        return bitstrings.neighbors(x)

    def sample_neighbor(self, x):
        return bitstrings.random_flip(x)

    def sample(self):
        return bitstrings.random_generate()
