#!/usr/bin/env python3
"""FedAvg state and aggregation for logistic regression."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Optional


class FedAvgState:
    def __init__(self, n_features: int, save_dir: Path, clients_per_round: int):
        self.n_features = n_features
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.round: int = 0
        self.clients_per_round = clients_per_round
        self.w: List[float] = [0.0] * n_features
        self.b: float = 0.0
        self._pending_updates: List[tuple] = []
        self._round_samples: int = 0
        self._load_if_exists()

    def _model_path(self) -> Path:
        return self.save_dir / "global_model.csv"

    def _load_if_exists(self) -> None:
        p = self._model_path()
        if p.exists():
            with p.open("r", newline="") as f:
                reader = csv.reader(f)
                first = next(reader)
                if first[0] == "round":
                    self.round = int(first[1])
                second = next(reader)
                if second[0] != "bias":
                    raise ValueError("Invalid model file format")
                self.b = float(second[1])
                third = next(reader)
                if third[0] != "weights":
                    raise ValueError("Invalid model file format")
                w = [float(v) for v in third[1:]]
                if len(w) != self.n_features:
                    # resize if needed
                    self.n_features = len(w)
                self.w = w

    def save(self) -> None:
        p = self._model_path()
        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", self.round])
            writer.writerow(["bias", self.b])
            writer.writerow(["weights"] + self.w)

    def get_global(self) -> dict:
        return {"round": self.round, "weights": self.w, "bias": self.b}

    def submit_update(self, delta_w: List[float], delta_b: float, num_samples: int) -> Optional[dict]:
        if len(delta_w) != self.n_features:
            return {"status": "error", "message": "delta_w length mismatch"}
        self._pending_updates.append((delta_w, delta_b, num_samples))
        self._round_samples += num_samples
        if len(self._pending_updates) >= self.clients_per_round:
            self._apply_aggregate()
            self.round += 1
            self.save()
            return {"status": "ok", "applied_round": self.round}
        return {"status": "queued", "received": len(self._pending_updates)}

    def _apply_aggregate(self) -> None:
        # weighted by num_samples
        total = sum(n for _, _, n in self._pending_updates)
        if total <= 0:
            self._pending_updates.clear()
            self._round_samples = 0
            return
        agg_dw = [0.0] * self.n_features
        agg_db = 0.0
        for dw, db, n in self._pending_updates:
            coef = n / total
            for i in range(self.n_features):
                agg_dw[i] += coef * dw[i]
            agg_db += coef * db
        # apply
        for i in range(self.n_features):
            self.w[i] += agg_dw[i]
        self.b += agg_db
        self._pending_updates.clear()
        self._round_samples = 0
