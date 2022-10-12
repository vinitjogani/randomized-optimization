import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from preprocessing import ElementsEncoder, interval_to_months


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


def load_credit_score():
    df = pd.read_csv("credit_score.csv", low_memory=False)

    # Basic cleanup
    clean_numbers = [
        "Num_of_Loan",
        "Outstanding_Debt",
        "Num_of_Delayed_Payment",
        "Amount_invested_monthly",
        "Annual_Income",
        "Age",
        "Monthly_Balance",
        "Changed_Credit_Limit",
    ]
    for f in clean_numbers:
        df[f] = df[f].map(lambda x: float(str(x).replace("_", "")) if x != "_" else 0)
    df["Credit_History_Age"] = df["Credit_History_Age"].map(interval_to_months)
    label_idx = {"Good": 2, "Standard": 1, "Poor": 0}
    df["Credit_Score"] = df["Credit_Score"].map(label_idx.get)

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

    # Encode
    element_features = ["Type_of_Loan"]
    onehot_features = [
        "Occupation",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
    ]
    numeric_features = [
        "Age",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Monthly_Balance",
        "Credit_History_Age",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
    ]

    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    elements_encoder = ElementsEncoder()
    power_transformer = PowerTransformer()

    onehot_encoder.fit(train_df[onehot_features])
    elements_encoder.fit(train_df[element_features])
    power_transformer.fit(train_df[numeric_features])

    def transform(df):
        x1 = onehot_encoder.transform(df[onehot_features])
        x2 = elements_encoder.transform(df[element_features])
        x3 = power_transformer.transform(df[numeric_features])
        x3[pd.isnull(x3)] = 0
        return np.concatenate([x1, x2, x3], axis=1)

    train = transform(train_df), train_df["Credit_Score"]
    test = transform(test_df), test_df["Credit_Score"]

    return train, test


def load_dataset(name):
    if name == "credit_score":
        return load_credit_score()
    elif name == "term_deposits":
        return load_term_deposits()
    raise ValueError("Dataset not found.")
