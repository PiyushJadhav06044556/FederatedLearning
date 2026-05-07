#!/usr/bin/env python3
"""FL server with FedAvg and secure aggregation."""
from __future__ import annotations
import os
import csv
from pathlib import Path
from typing import List

from flask import Flask, jsonify, request
from threading import Lock

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.fedavg import FedAvgState
from server.secure_aggregation.ecdh import generate_keypair
from server.secure_aggregation.bonawitz import make_mask_vector
from scripts.encoding import encode_vector_to_int, decode_int_to_float

app = Flask(__name__)
STATE = None
# SecAgg
SECAGG_ROUND = {
    'round_id': 0,
    'pubkeys': {},
    'ready_issued': set(),
    'submitted': set(),
    'masked_updates': [],
    't_start': None,
    'byte_count': 0,
    'cpu_start': None,
    'finalized': False,
}
SECAGG_LOCK = Lock()
Q = 2**61 - 1
S = 2**20


def infer_n_features(csv_path: Path, label_col: str = "Class") -> int:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if label_col in header:
            return len(header) - 1
        return len(header) - 1 


def get_state() -> FedAvgState:
    global STATE
    if STATE is None:
        clients_per_round = int(os.environ.get("CLIENTS_PER_ROUND", "3"))
        processed_path = Path(os.environ.get("DATA_PROCESSED", "data/processed/train.csv"))
        n_features = infer_n_features(processed_path)
        state_dir = Path(os.environ.get("STATE_DIR", "server/state"))
        STATE = FedAvgState(n_features=n_features, save_dir=state_dir, clients_per_round=clients_per_round)
    return STATE


@app.get("/global_model")
def get_global_model():
    return jsonify(get_state().get_global())


@app.post("/submit_update")
def submit_update():
    payload = request.get_json(force=True)
    delta_w = payload.get("delta_w")
    delta_b = payload.get("delta_b")
    num_samples = payload.get("num_samples", 0)
    if not isinstance(delta_w, list) or not isinstance(delta_b, (float, int)):
        return jsonify({"status": "error", "message": "invalid payload"}), 400
    res = get_state().submit_update([float(v) for v in delta_w], float(delta_b), int(num_samples))
    # log when applied
    if res and res.get("status") == "ok":
        _log_round_metrics()
    code = 200 if res and res.get("status") in ("ok", "queued") else 400
    return jsonify(res), code


@app.post("/register_round")
def register_round():
    global SECAGG_ROUND
    data = request.get_json(force=True)
    client_id = str(data.get('client_id'))
    pubkey_b64 = str(data.get('pubkey'))
    if not client_id or not pubkey_b64:
        return jsonify({"status": "error", "message": "missing client_id/pubkey"}), 400
    import base64
    with SECAGG_LOCK:
        # register pubkey
        if client_id not in SECAGG_ROUND['pubkeys']:
            SECAGG_ROUND['pubkeys'][client_id] = base64.b64decode(pubkey_b64)
        # enough clients: READY once
        if len(SECAGG_ROUND['pubkeys']) >= get_state().clients_per_round and not SECAGG_ROUND['finalized']:
            # start timing
            import time as _time
            import resource as _resource
            if SECAGG_ROUND.get('t_start') is None:
                SECAGG_ROUND['t_start'] = _time.time()
                ru = _resource.getrusage(_resource.RUSAGE_SELF)
                SECAGG_ROUND['cpu_start'] = float(ru.ru_utime + ru.ru_stime)
            # set round id
            SECAGG_ROUND['round_id'] = get_state().round
            if client_id in SECAGG_ROUND['ready_issued']:
                # already ready
                return jsonify({"status": "waiting", "message": "already_ready", "round_id": SECAGG_ROUND['round_id'], "registered": len(SECAGG_ROUND['pubkeys'])})
            SECAGG_ROUND['ready_issued'].add(client_id)
            all_pub = {cid: base64.b64encode(pk).decode() for cid, pk in SECAGG_ROUND['pubkeys'].items()}
            return jsonify({"status": "ready", "round_id": SECAGG_ROUND['round_id'], "pubkeys": all_pub})
        else:
            return jsonify({"status": "waiting", "registered": len(SECAGG_ROUND['pubkeys'])})


@app.post("/submit_masked_update")
def submit_masked_update():
    global SECAGG_ROUND
    data = request.get_json(force=True)
    masked_w = data.get('masked_w')
    masked_b = data.get('masked_b')
    num_samples = int(data.get('num_samples', 0))
    client_id = data.get('client_id')
    round_id = data.get('round_id')
    if not isinstance(masked_w, list) or not isinstance(masked_b, (int, float)) or num_samples <= 0 or not client_id:
        return jsonify({"status": "error", "message": "invalid payload"}), 400
    with SECAGG_LOCK:
        # round finalizing
        if SECAGG_ROUND.get('finalized'):
            return jsonify({"status": "round_closed", "round": SECAGG_ROUND.get('round_id', -1)})
        # round must match
        if round_id is None or int(round_id) != int(SECAGG_ROUND.get('round_id', 0)):
            return jsonify({"status": "wrong_round", "expected_round": SECAGG_ROUND.get('round_id', 0)}), 409
        # dedupe
        if client_id in SECAGG_ROUND['submitted']:
            return jsonify({"status": "duplicate", "received": len(SECAGG_ROUND['masked_updates'])})
        SECAGG_ROUND['submitted'].add(client_id)
        SECAGG_ROUND['masked_updates'].append((masked_w, int(masked_b), num_samples))
        # approx bytes
        SECAGG_ROUND['byte_count'] = SECAGG_ROUND.get('byte_count', 0) + 8 * (len(masked_w) + 1)
        received = len(SECAGG_ROUND['masked_updates'])
        if received < get_state().clients_per_round:
            return jsonify({"status": "queued", "received": received})

        # finalize and aggregate
        SECAGG_ROUND['finalized'] = True
        updates = list(SECAGG_ROUND['masked_updates'])

        # aggregate and apply
        n_features = get_state().n_features
        total = sum(ns for _vec, _b, ns in updates)
        agg_w = [0] * n_features
        agg_b = 0
        for vec, b_enc, _ns in updates:
            for i in range(n_features):
                agg_w[i] = (agg_w[i] + int(vec[i]) % Q) % Q
            agg_b = (agg_b + int(b_enc) % Q) % Q
        avg_w = [x / total for x in decode_int_to_float(agg_w, S=S, q=Q)]
        avg_b = decode_int_to_float([agg_b], S=S, q=Q)[0] / total
        st = get_state()
        for i in range(n_features):
            st.w[i] += avg_w[i]
        st.b += avg_b
        st.round += 1
        st.save()
        # log
        _log_round_metrics()
        _log_round_perf(total_bytes=SECAGG_ROUND.get('byte_count', 0))
        SECAGG_ROUND = {
            'round_id': get_state().round,
            'pubkeys': {},
            'ready_issued': set(),
            'submitted': set(),
            'masked_updates': [],
            't_start': None,
            'byte_count': 0,
            'cpu_start': None,
            'finalized': False,
        }
        return jsonify({"status": "ok", "applied_round": st.round})


# Logging / eval
def _round_files():
    state_dir = Path(os.environ.get("STATE_DIR", "server/state"))
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "round_log.csv", state_dir / "metrics_log.csv"


def _perf_file():
    state_dir = Path(os.environ.get("STATE_DIR", "server/state"))
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "perf_log.csv"


def _write_csv_row(path: Path, header: list, row: list):
    exists = path.exists()
    with path.open("a", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def _auc_roc(y_true, y_score) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda t: t[0])
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


def _precision_recall_f1(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _log_round_metrics():
    st = get_state()
    round_log, metrics_log = _round_files()
    # Round log
    from datetime import datetime
    _write_csv_row(round_log, ["round", "timestamp"], [st.round, datetime.utcnow().isoformat() + "Z"])
    # Optional evaluation
    test_csv = os.environ.get("EVAL_TEST_CSV")
    if test_csv and Path(test_csv).exists():
        from models.logistic_regression import LogisticRegression, load_csv_xy
        X_test, y_test, _ = load_csv_xy(test_csv, label_col=os.environ.get("LABEL_COL", "Class"))
        # current global model
        model = LogisticRegression(n_features=st.n_features)
        model.set_params(st.w, st.b)
        proba = model.predict_proba(X_test)
        auc = _auc_roc(y_test, proba)
        preds = [1 if p >= 0.5 else 0 for p in proba]
        prec, _rec, _f1 = _precision_recall_f1(y_test, preds)
        _write_csv_row(metrics_log, ["round", "AUC", "Precision"],
                       [st.round, f"{auc:.6f}", f"{prec:.6f}"])


def _log_round_perf(total_bytes: int = 0):
    perf_path = _perf_file()
    import time as _time
    import resource as _resource
    dur = None
    if SECAGG_ROUND.get('t_start') is not None:
        dur = _time.time() - SECAGG_ROUND['t_start']
    cpu = None
    try:
        ru = _resource.getrusage(_resource.RUSAGE_SELF)
        cpu_total = float(ru.ru_utime + ru.ru_stime)
        cpu_start = SECAGG_ROUND.get('cpu_start')
        cpu = (cpu_total - cpu_start) if cpu_start is not None else None
    except Exception:
        cpu = None
    st = get_state()
    _write_csv_row(perf_path,
                   ["round", "secagg", "clients", "total_masked_bytes", "duration_s", "cpu_time_s"],
                   [st.round, 1, st.clients_per_round, total_bytes, f"{(dur or 0):.6f}", f"{(cpu or 0):.6f}"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
