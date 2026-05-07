#!/usr/bin/env python3
"""Compare centralized vs federated metrics (AUC, Precision) and plot overlay."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def read_centralized(path: Path):
    vals = {}
    if not path.exists():
        raise FileNotFoundError(f"Centralized metrics not found: {path}")
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            vals[row["metric"]] = float(row["value"]) 
    return vals


def read_fl_metrics(path: Path):
    rounds, aucs, precs = [], [], []
    if not path.exists():
        raise FileNotFoundError(f"Federated metrics log not found: {path}")
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            aucs.append(float(row["AUC"]))
            precs.append(float(row["Precision"]))
    return rounds, aucs, precs


def compare_and_plot():
    cen_path = Path("outputs/metrics/centralized_metrics.csv")
    fl_path = Path("server/state/metrics_log.csv")
    cen = read_centralized(cen_path)
    rounds, aucs, precs = read_fl_metrics(fl_path)

    # latest federated
    auc_last = aucs[-1] if aucs else 0
    prec_last = precs[-1] if precs else 0

    # write CSV
    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_path = out_dir / "baseline_vs_fl.csv"
    with cmp_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "centralized", "federated_last_round"]) 
        w.writerow(["AUC", f"{cen.get('AUC', 0):.6f}", f"{auc_last:.6f}"])
        w.writerow(["Precision", f"{cen.get('Precision', 0):.6f}", f"{prec_last:.6f}"])
    print(f"Saved comparison to {cmp_path}")

    # plot overlay
    out_plots = Path("outputs/plots")
    out_plots.mkdir(parents=True, exist_ok=True)
    out_path = out_plots / "centralized_vs_fl.png"

    plt.figure(figsize=(8, 5))
    if rounds:
        plt.plot(rounds, aucs, label="FL AUC")
        plt.plot(rounds, precs, label="FL Precision")
    plt.axhline(cen.get('AUC', 0), color='C0', linestyle='--', alpha=0.6, label="Centralized AUC")
    plt.axhline(cen.get('Precision', 0), color='C1', linestyle='--', alpha=0.6, label="Centralized Precision")
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title("Centralized vs Federated (server-side evaluation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    compare_and_plot()
