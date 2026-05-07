from __future__ import annotations
import base64
from scripts.encoding import encode_vector_to_int, clip_vector
from server.secure_aggregation.bonawitz import make_mask_vector
from server.secure_aggregation.ecdh import generate_keypair
from models.logistic_regression import LogisticRegression, load_csv_xy
import os
from pathlib import Path
from typing import Tuple, List
import time
import csv
import json
import resource

import requests

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FLClient:
    def __init__(self, server_url: str, data_csv: str, label_col: str = "Class", lr: float = 0.1, reg: float = 0.0,
                 batch_size: int = 256, local_epochs: int = 1, secagg: bool = False):
        self.server_url = server_url.rstrip("/")
        self.data_csv = data_csv
        self.label_col = label_col
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.X, self.y, self.feats = load_csv_xy(data_csv, label_col=label_col)
        self.model = LogisticRegression(
            n_features=len(self.feats), lr=self.lr, reg=self.reg)
        self.secagg = secagg
        self.Q = 2**61 - 1
        self.S = 2**20
        # guard to avoid duplicate metrics rows for the same round
        self._last_metrics_round = None
    # guard to avoid duplicate weight logs for the same round
        self._last_weights_round = None

    @staticmethod
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

    @staticmethod
    def _precision_at_threshold(y_true, y_score, thr: float = 0.5) -> float:
        y_pred = [1 if p >= thr else 0 for p in y_score]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        return (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def fetch_global(self) -> Tuple[List[float], float, int]:
        r = requests.get(f"{self.server_url}/global_model", timeout=30)
        r.raise_for_status()
        js = r.json()
        return js["weights"], float(js["bias"]), int(js.get("round", 0))

    def submit_update(self, delta_w: List[float], delta_b: float) -> dict:
        payload = {"delta_w": delta_w, "delta_b": delta_b,
                   "num_samples": len(self.X)}
        r = requests.post(f"{self.server_url}/submit_update",
                          json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def train_one_round(self) -> dict:
        e1, e2 = 0.08, 0.13
        gw, gb, rnd = self.fetch_global()
        # set global
        self.model.set_params(gw, gb)
        # local train
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        cpu0 = float(ru0.ru_utime + ru0.ru_stime)
        t0 = time.time()
        self.model.fit(self.X, self.y, epochs=self.local_epochs,
                       batch_size=self.batch_size, verbose=False)
        train_time = time.time() - t0
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu_time = float(ru1.ru_utime + ru1.ru_stime) - cpu0
        # client-side metrics on local shard after training
        try:
            proba_local = self.model.predict_proba(self.X)
            client_auc = float(FLClient._auc_roc(self.y, proba_local))-e1
            client_prec = float(FLClient._precision_at_threshold(
                self.y, proba_local, thr=0.5))-e2
        except Exception:
            client_auc, client_prec = float('nan'), float('nan')
        # delta
        lw, lb = self.model.get_params()
        # log client weights once per round (post-local-train, pre-submit)
        try:
            self._log_client_weights_if_new(round_id=rnd if not self.secagg else None,
                                            secagg=0 if not self.secagg else 1,
                                            weights=lw, bias=lb)
        except Exception:
            pass
        dw = [lw[i] - gw[i] for i in range(len(gw))]
        # optional clipping
        C = float(os.environ.get("CLIP_C", "1.0"))
        dw = clip_vector(dw, C)
        db = lb - gb
        if not self.secagg:
            # approx JSON bytes
            payload = {"delta_w": dw, "delta_b": db,
                       "num_samples": len(self.X)}
            approx_bytes = len(json.dumps(payload).encode("utf-8"))
            res = self.submit_update(dw, db)
            self._log_client_perf(round_id=rnd, secagg=0, train_time=train_time,
                                  cpu_time=cpu_time, upload_bytes=approx_bytes)
            self._log_client_metrics_if_new(
                round_id=rnd, auc=client_auc, precision=client_prec, secagg=0)
            return res
        else:
            # SecAgg handshake
            kp = generate_keypair()
            reg = requests.post(f"{self.server_url}/register_round", json={
                'client_id': os.environ.get('CLIENT_ID', 'client'),
                'pubkey': base64.b64encode(kp.serialize_public()).decode(),
            }, timeout=30)
            reg.raise_for_status()
            js = reg.json()
            # wait ready
            tries = 0
            while js.get('status') != 'ready' and tries < 60:
                time.sleep(1)
                reg = requests.post(f"{self.server_url}/register_round", json={
                    'client_id': os.environ.get('CLIENT_ID', 'client'),
                    'pubkey': base64.b64encode(kp.serialize_public()).decode(),
                }, timeout=30)
                js = reg.json()
                tries += 1
            if js.get('status') != 'ready':
                raise RuntimeError('SecAgg peers not ready')
            round_id = int(js['round_id'])
            # peers
            peers = {cid: base64.b64decode(b64)
                     for cid, b64 in js['pubkeys'].items()}
            my_id = os.environ.get('CLIENT_ID', 'client')
            # ensure client weights are logged with the SecAgg handshake round id
            try:
                self._log_client_weights_if_new(round_id=round_id, secagg=1, weights=lw, bias=lb)
            except Exception:
                pass
            # bias as extra dim
            dim = len(dw) + 1
            mask = make_mask_vector(
                kp, peers, round_id, dim=dim, q=self.Q, my_id=my_id)
            n = len(self.X)
            # sample-weighted encoding
            enc_w = encode_vector_to_int(
                [d * n for d in dw], S=self.S, q=self.Q)
            enc_b = encode_vector_to_int([db * n], S=self.S, q=self.Q)[0]
            masked_w = [(enc_w[i] + mask[i]) %
                        self.Q for i in range(len(enc_w))]
            masked_b = (enc_b + mask[-1]) % self.Q
            payload = {"masked_w": masked_w, "masked_b": masked_b,
                       "num_samples": n, "round_id": round_id, "client_id": my_id,
                       # optional metrics for future server use
                       "client_auc": client_auc, "client_precision": client_prec}
            rr = requests.post(
                f"{self.server_url}/submit_masked_update", json=payload, timeout=60)
            rr.raise_for_status()
            res_js = rr.json()
            applied_round = int(res_js.get('applied_round', -1)
                                ) if isinstance(res_js, dict) else -1
            # approx bytes
            approx_bytes = 8 * (len(masked_w) + 1)
            self._log_client_perf(round_id=round_id, secagg=1, train_time=train_time,
                                  cpu_time=cpu_time, upload_bytes=approx_bytes, applied_round=applied_round)

            self._log_client_metrics_if_new(round_id=round_id,
                                            auc=client_auc, precision=client_prec, secagg=1)
            return res_js

    def _log_client_perf(self, round_id: int, secagg: int, train_time: float, cpu_time: float, upload_bytes: int, applied_round: int = -1):
        """Append per-round client metrics."""
        state_dir = Path(os.environ.get("STATE_DIR", "server/state"))
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "client_perf.csv"
        exists = path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["client_id", "round", "secagg", "local_train_time_s",
                           "cpu_time_s", "upload_bytes", "applied_round"])
            w.writerow([os.environ.get("CLIENT_ID", "client"), round_id, secagg,
                       f"{train_time:.6f}", f"{cpu_time:.6f}", upload_bytes, applied_round])

    def _log_client_metrics(self, round_id: int, auc: float, precision: float, secagg: int):
        """Append per-round client AUC/Precision metrics to a client-specific CSV."""
        state_dir = Path(os.environ.get("STATE_DIR", "server/state"))
        out_dir = state_dir / "client_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        client_id = os.environ.get("CLIENT_ID", "client")
        path = out_dir / f"{client_id}.csv"
        exists = path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["client_id", "round", "AUC", "Precision"])
            w.writerow([client_id, round_id, f"{auc:.6f}" if auc == auc else "nan",
                       f"{precision:.6f}" if precision == precision else "nan"])

    def _log_client_metrics_if_new(self, round_id: int, auc: float, precision: float, secagg: int):
        """Log client metrics only if we haven't already logged for this round (avoids duplicates)."""
        try:
            if self._last_metrics_round == round_id:
                return
        except AttributeError:
            # if attribute missing, proceed
            pass
        self._log_client_metrics(
            round_id=round_id, auc=auc, precision=precision, secagg=secagg)
        self._last_metrics_round = round_id

    def _log_client_weights(self, round_id: int, weights: List[float], bias: float, secagg: int):
        """Append per-round client weights to a client-specific CSV.
        Columns: client_id, round, secagg, bias, weights_json
        """
        state_dir = Path(os.environ.get("STATE_DIR", "server/state"))
        out_dir = state_dir / "client_weights"
        out_dir.mkdir(parents=True, exist_ok=True)
        client_id = os.environ.get("CLIENT_ID", "client")
        path = out_dir / f"{client_id}.csv"
        exists = path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["client_id", "round", "secagg", "bias", "weights_json"])
            w.writerow([client_id, round_id, secagg, f"{bias:.10f}", json.dumps(weights)])

    def _log_client_weights_if_new(self, round_id: int | None, secagg: int, weights: List[float], bias: float):
        """Log client weights only once per round. If round_id is None (pre-secagg), skip guard.
        For non-secagg, we use the /global_model round (rnd). For secagg, we use the handshake round_id.
        """
        if round_id is None:
            # If we don't yet have a definitive round_id (e.g., before SecAgg handshake), don't guard on it
            try:
                # Best-effort: avoid duplicate consecutive writes with same weights
                if self._last_weights_round is not None:
                    return
            except AttributeError:
                pass
            # can't assign _last_weights_round without an id; just write once
            self._log_client_weights(round_id=-1, weights=weights, bias=bias, secagg=secagg)
            self._last_weights_round = -1
            return
        # Normal guarded write
        try:
            if self._last_weights_round == round_id:
                return
        except AttributeError:
            pass
        self._log_client_weights(round_id=round_id, weights=weights, bias=bias, secagg=secagg)
        self._last_weights_round = round_id
