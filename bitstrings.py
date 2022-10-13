import random
import numpy as np


def random_generate(length=1024):
    return "".join([str(random.randint(0, 1)) for _ in range(length)])


def flip(c):
    return "0" if c == "1" else "1"


def random_flip(x):
    y = list(x)
    i = random.randint(0, len(x) - 1)
    y[i] = flip(x[i])
    return "".join(y)


def neighbors(x):
    for i, c in enumerate(x):
        y = list(x)
        y[i] = flip(c)
        yield "".join(y)


def splice_combine(x1, x2, p=0.1):
    def combine_bit(x1, x2):
        if np.random.random() < p:
            return str(random.randint(0, 1))

        if np.random.random() < 0.5:
            return x1

        return x2

    return "".join([combine_bit(b1, b2) for b1, b2 in zip(x1, x2)])


def cut_combine(x1, x2, p=0.1):
    cut = np.random.randint(len(x1) - 1)
    x = list(x1[:cut] + x2[cut:])
    for i in np.where(np.random.random(size=len(x)) < p)[0]:
        x[i] = str(1 - int(x[i]))
    return "".join(x)
