#!/usr/bin/env python3
"""
Generate synthetic credit scoring dataset and save to CSV.
Usage:
    python src/data_generation.py --n 5000 --out data/dataset.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate(n_samples=10000, out_path="data/dataset.csv", random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.75, 0.25],  # imbalanced: more non-defaults
        class_sep=1.0,
        random_state=random_state,
    )
    columns = [
        "income",
        "age",
        "employment_years",
        "num_credit_lines",
        "delinquencies",
        "credit_utilization",
        "months_since_last_delinq",
        "debt_to_income_ratio",
        "savings",
        "loan_amount",
        "payment_history_score",
        "other_expenses",
    ]
    df = pd.DataFrame(X, columns=columns)
    rng = np.random.RandomState(random_state)

    # realistic scaling/transforms
    df["income"] = (np.abs(df["income"]) * 20000 + 15000).astype(int)
    df["age"] = (np.abs(df["age"]) * 12 + 25).astype(int).clip(18, 80)
    df["employment_years"] = (np.abs(df["employment_years"]) * 3).astype(int)
    df["num_credit_lines"] = (np.abs(df["num_credit_lines"]) * 2 + 1).astype(int)
    df["delinquencies"] = np.clip((np.abs(df["delinquencies"]) * 2).astype(int), 0, 10)
    df["credit_utilization"] = np.clip(np.abs(df["credit_utilization"]) * 100, 0, 200)
    df["months_since_last_delinq"] = np.abs(df["months_since_last_delinq"] * 12).astype(int)
    df["debt_to_income_ratio"] = np.clip(np.abs(df["debt_to_income_ratio"]) * 0.5, 0, 1)
    df["savings"] = (np.abs(df["savings"]) * 5000).astype(int)
    df["loan_amount"] = (np.abs(df["loan_amount"]) * 20000).astype(int)
    # normalize payment_history_score to 0-100
    ph = df["payment_history_score"]
    df["payment_history_score"] = ((ph - ph.min()) / (ph.max() - ph.min()) * 100).astype(int)
    df["other_expenses"] = (np.abs(df["other_expenses"]) * 2000).astype(int)

    # target: 1 = default / high risk, 0 = good
    df["target"] = y
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000, help="number of samples")
    parser.add_argument("--out", type=str, default="data/dataset.csv", help="output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    generate(n_samples=args.n, out_path=args.out, random_state=args.seed)

