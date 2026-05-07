#!/usr/bin/env python3
"""
Preprocess credit card fraud dataset from data/creditcard.csv
- Standardize numeric features (z-score) using training stats
- Train/test split (80/20)
- Partition training data into K clients (IID by default) and save CSVs only

Usage:
  python scripts/preprocess_data.py --input data/creditcard.csv --clients 3 \
      --out-root data --seed 42 --test-size 0.2

Outputs:
  data/processed/train.csv
  data/processed/test.csv
  data/clients/client_1/train.csv ... client_K/train.csv
  data/clients/client_*/stats.csv (feature means/stds for reference)

Note: Uses pandas for efficient CSV handling, but outputs remain CSV only.
"""
import argparse
import os
from pathlib import Path
from typing import Tuple

import pandas as pd


def train_test_split_df(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    n_test = int(n * test_size)
    test_df = df_shuffled.iloc[:n_test].reset_index(drop=True)
    train_df = df_shuffled.iloc[n_test:].reset_index(drop=True)
    return train_df, test_df


def standardize_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [c for c in train_df.columns if c != label_col]
    means = train_df[features].mean()
    stds = train_df[features].std().replace(0, 1.0)
    train_df_std = train_df.copy()
    test_df_std = test_df.copy()
    train_df_std[features] = (train_df_std[features] - means) / stds
    test_df_std[features] = (test_df_std[features] - means) / stds
    stats = pd.DataFrame({"feature": features, "mean": means.values, "std": stds.values})
    return train_df_std, test_df_std, stats


def partition_clients(train_df: pd.DataFrame, clients: int, seed: int) -> Tuple[pd.DataFrame, ...]:
    # Simple IID split
    parts = []
    df_shuffled = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    sizes = [n // clients] * clients
    for i in range(n % clients):
        sizes[i] += 1
    start = 0
    for sz in sizes:
        parts.append(df_shuffled.iloc[start:start+sz].reset_index(drop=True))
        start += sz
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/creditcard.csv")
    ap.add_argument("--label", default="Class", help="Name of the label column (0/1)")
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default="data")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not in CSV. Columns: {list(df.columns)}")

    # Ensure label is last column for consistency
    feature_cols = [c for c in df.columns if c != args.label]
    df = df[feature_cols + [args.label]]

    train_df, test_df = train_test_split_df(df, test_size=args.test_size, seed=args.seed)
    train_df, test_df, stats = standardize_train_test(train_df, test_df, label_col=args.label)

    processed_dir = Path(args.out_root) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    # Partition for clients (training data only)
    client_parts = partition_clients(train_df, clients=args.clients, seed=args.seed)
    clients_root = Path(args.out_root) / "clients"
    clients_root.mkdir(parents=True, exist_ok=True)

    for i, part in enumerate(client_parts, start=1):
        cdir = clients_root / f"client_{i}"
        cdir.mkdir(parents=True, exist_ok=True)
        part.to_csv(cdir / "train.csv", index=False)
        stats.to_csv(cdir / "stats.csv", index=False)

    print(f"Saved processed CSVs to {processed_dir} and per-client splits to {clients_root}")


if __name__ == "__main__":
    main()
