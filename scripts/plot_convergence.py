#!/usr/bin/env python3
"""
Read per-round server metrics from server/state/metrics_log.csv and produce a convergence plot.
Outputs: outputs/plots/convergence.png
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Metrics log not found: {path}. Ensure EVAL_TEST_CSV is set on the server.")
    rounds = []
    aucs = []
    precs = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            aucs.append(float(row["AUC"]))
            precs.append(float(row["Precision"]))  # fixed: moved inside the loop
    return rounds, aucs, precs


def main():
    metrics_path = Path("server/state/metrics_log.csv")
    rounds, aucs, precs = load_metrics(metrics_path)

    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "convergence.png"

    plt.figure(figsize=(8, 5))
    if aucs:
        plt.plot(rounds, aucs, label="AUC")
    if precs:
        plt.plot(rounds, precs, label="Precision")
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title("Federated Convergence (Server-side evaluation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()