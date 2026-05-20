"""
Hand-written Federated Averaging (FedAvg) server.

No external FL framework: this module implements McMahan et al. 2017 directly so
every line is auditable. It exposes:

    federated_average(client_weights, client_sizes) -> list[np.ndarray]
    class FederationServer

The server orchestrates one FedAvg round at a time:
    initialize_global_model() -> distribute_weights_to_clients() ->
    (clients train locally) -> collect_and_aggregate(updates) ->
    evaluate_global_model(X_val, y_val) -> save_global_model(round) ->
    log_round(round, metrics, n_clients)
"""

from __future__ import annotations

import csv
import copy
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from models.hybrid_model import build_model


_LOG_COLUMNS = [
    "timestamp",
    "round",
    "n_clients",
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


def federated_average(
    client_weights: list[list[np.ndarray]],
    client_sizes: list[int],
) -> list[np.ndarray]:
    """Weighted average of per-client model weights.

    w_global[l] = sum_k (n_k / N) * w_k[l]   where N = sum_k n_k

    Raises ValueError on empty input, mismatched lengths, non-positive totals,
    or layer-shape disagreements between clients.
    """
    if not client_weights:
        raise ValueError("client_weights is empty; nothing to aggregate")
    if len(client_weights) != len(client_sizes):
        raise ValueError(
            f"client_weights ({len(client_weights)}) and client_sizes "
            f"({len(client_sizes)}) length mismatch"
        )
    if any(n < 0 for n in client_sizes):
        raise ValueError("client_sizes must be non-negative")

    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("sum of client_sizes must be > 0")

    reference = client_weights[0]
    num_layers = len(reference)
    for ci, w in enumerate(client_weights):
        if len(w) != num_layers:
            raise ValueError(
                f"client {ci} has {len(w)} layers, expected {num_layers}"
            )
        for li, (a, b) in enumerate(zip(w, reference)):
            if a.shape != b.shape:
                raise ValueError(
                    f"client {ci} layer {li} shape {a.shape} != {b.shape}"
                )

    fractions = [n / total for n in client_sizes]

    averaged: list[np.ndarray] = []
    for layer_idx in range(num_layers):
        acc = np.zeros_like(reference[layer_idx], dtype=np.float32)
        for ci, frac in enumerate(fractions):
            acc += frac * client_weights[ci][layer_idx].astype(np.float32)
        averaged.append(acc)
    return averaged


class FederationServer:
    """Central FedAvg coordinator."""

    def __init__(
        self,
        build_model_fn: Callable[[], "object"] = build_model,
        model_dir: str | Path = "models/global",
        log_path: str | Path = "logs/training_log.csv",
    ) -> None:
        self.build_model_fn = build_model_fn
        self.model_dir = Path(model_dir)
        self.log_path = Path(log_path)
        self.global_weights: list[np.ndarray] | None = None
        self._model = None

    def initialize_global_model(self) -> list[np.ndarray]:
        """Build the architecture, snapshot its initial weights, prep log file."""
        self._model = self.build_model_fn()
        self.global_weights = [w.copy() for w in self._model.get_weights()]

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(_LOG_COLUMNS)

        return self.global_weights

    def distribute_weights_to_clients(self) -> list[np.ndarray]:
        """Return a deep copy so client-side mutations cannot leak back."""
        if self.global_weights is None:
            raise RuntimeError(
                "global model not initialised; call initialize_global_model() first"
            )
        return [w.copy() for w in self.global_weights]

    def collect_and_aggregate(
        self, client_updates: list[dict]
    ) -> list[np.ndarray]:
        """Aggregate a round's client updates via weighted FedAvg.

        client_updates: list of {"weights": [np.ndarray, ...], "n_samples": int,
                                  "client_id": str}
        """
        if not client_updates:
            raise ValueError("No client updates received for this round")

        weights = [u["weights"] for u in client_updates]
        sizes = [int(u["n_samples"]) for u in client_updates]
        self.global_weights = federated_average(weights, sizes)
        return self.global_weights

    def evaluate_global_model(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> dict:
        """Score current global weights on a held-out set."""
        if self.global_weights is None:
            raise RuntimeError("global model not initialised")
        if self._model is None:
            self._model = self.build_model_fn()
        self._model.set_weights(self.global_weights)

        y_prob = self._model.predict(X_val, verbose=0)
        y_true = np.asarray(y_val).astype(int)
        y_pred = (y_prob >= 0.5).astype(int)

        eps = 1e-7
        p = np.clip(y_prob, eps, 1.0 - eps)
        loss = float(
            -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        )

        return {
            "loss": loss,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }

    def save_global_model(self, round_number: int) -> Path:
        if self.global_weights is None:
            raise RuntimeError("global model not initialised")
        if self._model is None:
            self._model = self.build_model_fn()
        self._model.set_weights(self.global_weights)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        out = self.model_dir / f"global_round_{round_number}.weights.h5"
        self._model.save_weights(str(out))
        return out

    def log_round(
        self, round_number: int, metrics: dict, n_clients: int
    ) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(_LOG_COLUMNS)

        row = [
            datetime.utcnow().isoformat(timespec="seconds"),
            round_number,
            n_clients,
            metrics.get("loss", ""),
            metrics.get("accuracy", ""),
            metrics.get("precision", ""),
            metrics.get("recall", ""),
            metrics.get("f1", ""),
        ]
        with self.log_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)
