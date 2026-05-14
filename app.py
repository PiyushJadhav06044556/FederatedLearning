#!/usr/bin/env python3
"""
FedShield - Federated Fraud Detection Platform
Flask backend for the UI dashboard.
"""
from __future__ import annotations
import os
import sys
import json
import subprocess
import threading
import queue
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Project root is where app.py lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Global log queues for SSE streaming
log_queues: dict[str, queue.Queue] = {}
process_status: dict[str, str] = {}  # 'running', 'done', 'error'


# ─── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Check which steps are already completed."""
    return jsonify({
        "dataset_exists": (PROJECT_ROOT / "data" / "creditcard.csv").exists(),
        "processed_exists": (PROJECT_ROOT / "data" / "processed" / "train.csv").exists(),
        "baseline_exists": (PROJECT_ROOT / "outputs" / "metrics" / "centralized_metrics.csv").exists(),
        "fl_exists": (PROJECT_ROOT / "server" / "state" / "metrics_log.csv").exists(),
        "plots_exist": (PROJECT_ROOT / "outputs" / "plots" / "convergence.png").exists(),
    })


@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """Handle creditcard.csv upload."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400
    if not file.filename.endswith(".csv"):
        return jsonify({"status": "error", "message": "File must be a CSV"}), 400

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    save_path = data_dir / "creditcard.csv"
    file.save(str(save_path))
    size_mb = save_path.stat().st_size / (1024 * 1024)
    return jsonify({
        "status": "ok",
        "message": f"Dataset uploaded successfully ({size_mb:.1f} MB)",
        "size_mb": round(size_mb, 1)
    })


@app.route("/api/preprocess", methods=["POST"])
def preprocess():
    """Run preprocessing script and stream logs."""
    task_id = "preprocess"
    cmd = [
        sys.executable, "-m", "scripts.preprocess_data",
        "--input", "data/creditcard.csv",
        "--clients", "3",
        "--out-root", "data",
        "--seed", "42",
        "--test-size", "0.2"
    ]
    _run_task(task_id, cmd)
    return jsonify({"status": "started", "task_id": task_id})


@app.route("/api/train_baseline", methods=["POST"])
def train_baseline():
    """Run centralized training and stream logs."""
    task_id = "train_baseline"
    cmd = [
        sys.executable, "-m", "scripts.train_centralized",
        "--train", "data/processed/train.csv",
        "--test", "data/processed/test.csv",
        "--label", "Class",
        "--epochs", "2",
        "--lr", "0.1",
        "--batch-size", "512"
    ]
    _run_task(task_id, cmd)
    return jsonify({"status": "started", "task_id": task_id})


@app.route("/api/run_federated", methods=["POST"])
def run_federated():
    """Run the federated demo (docker) and stream logs."""
    task_id = "run_federated"

    def federated_task():
        q = log_queues[task_id]
        process_status[task_id] = "running"
        try:
            # Step 1: clean server state
            import shutil
            state_dir = PROJECT_ROOT / "server" / "state"
            if state_dir.exists():
                shutil.rmtree(str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)
            q.put("[FedShield] Cleaned server state\n")

            # Step 2: build docker image
            q.put("[FedShield] Building Docker image...\n")
            build_cmd = [
                "docker", "compose",
                "-f", str(PROJECT_ROOT / "docker" / "docker-compose.yml"),
                "--project-name", "fedavg_demo",
                "build", "server"
            ]
            _stream_process(build_cmd, q, cwd=str(PROJECT_ROOT))

            # Step 3: run docker compose up
            q.put("[FedShield] Starting federated demo (server + 3 clients)...\n")
            up_cmd = [
                "docker", "compose",
                "-f", str(PROJECT_ROOT / "docker" / "docker-compose.yml"),
                "--project-name", "fedavg_demo",
                "up", "--no-build", "--abort-on-container-exit"
            ]
            _stream_process(up_cmd, q, cwd=str(PROJECT_ROOT))

            q.put("[FedShield] ✓ Federated demo complete!\n")
            process_status[task_id] = "done"
        except Exception as e:
            q.put(f"[ERROR] {e}\n")
            process_status[task_id] = "error"
        finally:
            q.put("__DONE__")

    log_queues[task_id] = queue.Queue()
    process_status[task_id] = "pending"
    t = threading.Thread(target=federated_task, daemon=True)
    t.start()
    return jsonify({"status": "started", "task_id": task_id})


@app.route("/api/generate_plots", methods=["POST"])
def generate_plots():
    """Run all 3 plot scripts."""
    task_id = "generate_plots"

    def plots_task():
        q = log_queues[task_id]
        process_status[task_id] = "running"
        try:
            for script in [
                "scripts.plot_convergence",
                "scripts.plot_performance",
                "scripts.compare_baseline_vs_fl"
            ]:
                q.put(f"[FedShield] Running {script}...\n")
                cmd = [sys.executable, "-m", script]
                _stream_process(cmd, q, cwd=str(PROJECT_ROOT))
            q.put("[FedShield] ✓ All plots generated!\n")
            process_status[task_id] = "done"
        except Exception as e:
            q.put(f"[ERROR] {e}\n")
            process_status[task_id] = "error"
        finally:
            q.put("__DONE__")

    log_queues[task_id] = queue.Queue()
    process_status[task_id] = "pending"
    t = threading.Thread(target=plots_task, daemon=True)
    t.start()
    return jsonify({"status": "started", "task_id": task_id})


@app.route("/api/logs/<task_id>")
def stream_logs(task_id):
    """SSE endpoint - streams logs for a given task."""
    def generate():
        if task_id not in log_queues:
            yield f"data: Task not found\n\n"
            return
        q = log_queues[task_id]
        while True:
            try:
                msg = q.get(timeout=30)
                if msg == "__DONE__":
                    yield f"data: __DONE__\n\n"
                    break
                # Escape for SSE
                for line in msg.splitlines():
                    yield f"data: {line}\n\n"
            except queue.Empty:
                yield f"data: [ping]\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/metrics/baseline")
def get_baseline_metrics():
    """Return centralized metrics as JSON."""
    path = PROJECT_ROOT / "outputs" / "metrics" / "centralized_metrics.csv"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    import csv
    metrics = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["metric"]] = float(row["value"])
    return jsonify(metrics)


@app.route("/api/metrics/federated")
def get_federated_metrics():
    """Return per-round federated metrics as JSON."""
    path = PROJECT_ROOT / "server" / "state" / "metrics_log.csv"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    import csv
    rounds, aucs, precs = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            aucs.append(float(row["AUC"]))
            precs.append(float(row["Precision"]))
    return jsonify({"rounds": rounds, "aucs": aucs, "precisions": precs})


@app.route("/api/metrics/comparison")
def get_comparison_metrics():
    """Return baseline vs FL comparison."""
    path = PROJECT_ROOT / "outputs" / "metrics" / "baseline_vs_fl.csv"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    import csv
    data = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["metric"]] = {
                "centralized": float(row["centralized"]),
                "federated": float(row["federated_last_round"])
            }
    return jsonify(data)


@app.route("/api/metrics/performance")
def get_performance_metrics():
    """Return performance summary."""
    path = PROJECT_ROOT / "outputs" / "metrics" / "performance_summary.csv"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    import csv
    data = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["metric"]] = row["value"]
    return jsonify(data)


@app.route("/api/plot/<name>")
def serve_plot(name):
    """Serve plot images."""
    allowed = ["convergence", "performance", "centralized_vs_fl"]
    if name not in allowed:
        return jsonify({"error": "Not found"}), 404
    path = PROJECT_ROOT / "outputs" / "plots" / f"{name}.png"
    if not path.exists():
        return jsonify({"error": "Plot not generated yet"}), 404
    return send_file(str(path), mimetype="image/png")


@app.route("/api/task_status/<task_id>")
def task_status(task_id):
    return jsonify({"status": process_status.get(task_id, "unknown")})


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def _run_task(task_id: str, cmd: list):
    """Run a command in a background thread, streaming output to a queue."""
    log_queues[task_id] = queue.Queue()
    process_status[task_id] = "running"

    def worker():
        try:
            _stream_process(cmd, log_queues[task_id], cwd=str(PROJECT_ROOT))
            process_status[task_id] = "done"
        except Exception as e:
            log_queues[task_id].put(f"[ERROR] {e}\n")
            process_status[task_id] = "error"
        finally:
            log_queues[task_id].put("__DONE__")

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def _stream_process(cmd: list, q: queue.Queue, cwd: str = None):
    """Run a subprocess and put its stdout/stderr lines into a queue."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=cwd or str(PROJECT_ROOT),
        bufsize=1
    )
    for line in iter(proc.stdout.readline, ""):
        q.put(line)
    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Process exited with code {proc.returncode}")


if __name__ == "__main__":
    # Ensure templates folder exists
    (PROJECT_ROOT / "templates").mkdir(exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)