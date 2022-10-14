import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer


def load_term_deposits():
    df = pd.read_csv("term_deposits.csv")
    label_idx = {"yes": 1, "no": 0}
    df["y"] = df["y"].map(label_idx.get)

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

    # Encode
    onehot_features = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]
    numeric_features = [
        "age",
        "campaign",
        "pdays",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]

    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    power_transformer = PowerTransformer()

    onehot_encoder.fit(train_df[onehot_features])
    power_transformer.fit(train_df[numeric_features])

    def transform(df):
        x1 = onehot_encoder.transform(df[onehot_features])
        x2 = power_transformer.transform(df[numeric_features])
        x2[pd.isnull(x2)] = 0
        return np.concatenate([x1, x2], axis=1)

    train = transform(train_df), train_df["y"]
    test = transform(test_df), test_df["y"]

    return train, test
