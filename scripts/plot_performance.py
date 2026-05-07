#!/usr/bin/env python3
"""Plot server latency and client upload bytes; write a short summary."""
from __future__ import annotations
import csv
from pathlib import Path
import statistics as stats
import matplotlib.pyplot as plt


def read_server_perf(path: Path):
    rounds = []
    duration = []
    cpu = []
    bytes_total = []
    if not path.exists():
        raise FileNotFoundError(f"Server perf log not found: {path}")
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            duration.append(float(row.get("duration_s", 0)))
            cpu.append(float(row.get("cpu_time_s", 0)))
            bytes_total.append(int(float(row.get("total_masked_bytes", 0))))
    return rounds, duration, cpu, bytes_total


def read_client_perf(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"Client perf log not found: {path}")
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "client_id": row["client_id"],
                "round": int(row["round"]),
                "secagg": int(row.get("secagg", 0)),
                "local_train_time_s": float(row.get("local_train_time_s", 0)),
                "cpu_time_s": float(row.get("cpu_time_s", 0)),
                "upload_bytes": int(float(row.get("upload_bytes", 0))),
            })
    return rows


def summarize_and_plot():
    server_path = Path("server/state/perf_log.csv")
    client_path = Path("server/state/client_perf.csv")
    rounds, duration, cpu, bytes_total = read_server_perf(server_path)
    client_rows = read_client_perf(client_path)

    # client bytes per round
    round_bytes_per_client = {}
    for row in client_rows:
        round_bytes_per_client.setdefault(row["round"], []).append(row["upload_bytes"])

    # plots
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "performance.png"

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)

    axes[0].plot(rounds, duration, marker="o", label="Server latency (s)")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # bytes per client per round
    x = sorted(round_bytes_per_client.keys())
    y = [round_bytes_per_client[r] for r in x]
    axes[1].boxplot(y, labels=[str(r) for r in x], showfliers=False)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Upload bytes per client")
    axes[1].set_title("Client upload size distribution per round")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Performance: latency and communication")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")

    # summary CSV
    out_metrics = Path("outputs/metrics")
    out_metrics.mkdir(parents=True, exist_ok=True)
    summary_path = out_metrics / "performance_summary.csv"
    avg_latency = stats.mean(duration) if duration else 0
    avg_cpu = stats.mean(cpu) if cpu else 0
    total_bytes = sum(bytes_total)
    # avg bytes per client
    all_client_bytes = [b for arr in round_bytes_per_client.values() for b in arr]
    avg_bytes_per_client = stats.mean(all_client_bytes) if all_client_bytes else 0
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["avg_round_latency_s", f"{avg_latency:.6f}"])
        w.writerow(["avg_server_cpu_time_s", f"{avg_cpu:.6f}"])
        w.writerow(["total_masked_bytes", total_bytes])
        w.writerow(["avg_upload_bytes_per_client", int(avg_bytes_per_client)])
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    summarize_and_plot()
