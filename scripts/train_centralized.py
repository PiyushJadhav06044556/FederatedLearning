"""
Train centralized baseline using Logistic Regression on CSV data only.
Reads data/processed/train.csv and test.csv with label column 'Class' by default.
Computes AUC and precision, then saves results under outputs/.

Usage:
    python scripts/train_centralized.py --train data/processed/train.csv --test data/processed/test.csv \
            --label Class --epochs 5 --lr 0.1 --batch-size 256 --reg 0.0
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.logistic_regression import LogisticRegression, load_csv_xy


def auc_roc(y_true: List[int], y_score: List[float]) -> float:
    pairs = list(zip(y_score, y_true))
    pairs.sort(key=lambda t: t[0])
    n0 = sum(1 for _, yt in pairs if yt == 0)
    n1 = sum(1 for _, yt in pairs if yt == 1)
    if n0 == 0 or n1 == 0:
        return 0.0
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        rank += (j - i)
        i = j
    U1 = rank_sum - n1 * (n1 + 1) / 2.0
    return U1 / (n0 * n1)


def precision_only(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--test", default="data/processed/test.csv")
    ap.add_argument("--label", default="Class")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--reg", type=float, default=0.0)
    args = ap.parse_args()

    X_train, y_train, feat_names = load_csv_xy(args.train, label_col=args.label)
    X_test, y_test, _ = load_csv_xy(args.test, label_col=args.label)

    model = LogisticRegression(n_features=len(feat_names), lr=args.lr, reg=args.reg)
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=True)

    proba = model.predict_proba(X_test)
    auc = auc_roc(y_test, proba)

    preds = [1 if p >= 0.5 else 0 for p in proba]
    precision = precision_only(y_test, preds)

    # Save outputs
    outputs_dir = Path("outputs")
    (outputs_dir / "models").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics").mkdir(parents=True, exist_ok=True)

    model_path = outputs_dir / "models" / "logreg.csv"
    model.save_csv(str(model_path))

    metrics_path = outputs_dir / "metrics" / "centralized_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["AUC", f"{auc:.6f}"])
        writer.writerow(["Precision", f"{precision:.6f}"])

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
