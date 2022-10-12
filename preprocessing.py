import re
import pandas as pd
import numpy as np


def interval_to_months(x):
    if pd.isnull(x):
        return 0
    y, m = re.findall(r"([0-9]+) Years and ([0-9]+) Months", x)[0]
    return 12 * int(y) + int(m)


def elements(x):
    if pd.isnull(x):
        return []
    x = x.replace(" and ", "")
    return [i.strip() for i in x.split(",")]


class ElementsEncoder:
    def __init__(self):
        self.vocab = {}

    def fit(self, X):
        for col in X.columns:
            values = X[col].map(elements)
            vocab = set(np.hstack(values))
            self.vocab[col] = list(sorted(list(vocab)))
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.vocab:
            vocab = self.vocab[col]
            values = X[col].map(elements).map(set)
            for item in vocab:
                item = item.lower().replace(" ", "_")
                X[f"{col}_{item}"] = values.map(lambda x: item in x)
        return X.drop(columns=list(self.vocab))

    def fit_transform(self, X):
        return self.fit(X).transform(X)
