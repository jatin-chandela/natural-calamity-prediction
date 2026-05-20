"""
LocalClient — simulates a single weather-station federated node.

Each node owns its city CSV and never exposes raw data. Only model weights
(and the training-set size needed for weighted FedAvg) leave the node.

Usage:
    client = LocalClient("mumbai", "clients/node_mumbai.csv")
    client.load_local_data()
    client.receive_global_weights(global_weights)
    client.local_train(epochs=5)
    weights, n = client.get_model_update()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.fetch_data import clean, load_imd_csv, make_sequences, normalize
from models.hybrid_model import build_model


class LocalClient:
    """A single federated weather-station node."""

    def __init__(self, node_id: str, data_path: str | Path) -> None:
        self.node_id = node_id
        self.data_path = Path(data_path)

        self._model = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._X_val: np.ndarray | None = None
        self._y_val: np.ndarray | None = None
        self._n_train: int = 0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_local_data(self, window: int = 30, val_fraction: float = 0.2) -> None:
        """Load and preprocess the city CSV; build the local model."""
        df = load_imd_csv(self.data_path)
        df = clean(df)
        df, _ = normalize(df)

        X, y_flood, y_cyclone = make_sequences(df, window=window)
        y = np.column_stack([y_flood, y_cyclone]).astype(np.float32)

        split = int(len(X) * (1.0 - val_fraction))
        self._X_train, self._X_val = X[:split], X[split:]
        self._y_train, self._y_val = y[:split], y[split:]
        self._n_train = len(self._X_train)

        self._model = build_model()

    # ------------------------------------------------------------------
    # Federated interface
    # ------------------------------------------------------------------

    def receive_global_weights(self, weights: list[np.ndarray]) -> None:
        """Overwrite local model weights with the server's global weights."""
        if self._model is None:
            raise RuntimeError(f"[{self.node_id}] call load_local_data() first")
        self._model.set_weights(weights)

    def local_train(self, epochs: int = 5) -> None:
        """Train on local data only — raw data never leaves this method."""
        if self._model is None or self._X_train is None:
            raise RuntimeError(f"[{self.node_id}] call load_local_data() first")
        self._model.fit(
            self._X_train,
            self._y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0,
        )

    def get_model_update(self) -> tuple[list[np.ndarray], int]:
        """Return updated weights and training-set size.

        Only weights leave the node — never raw data.
        """
        if self._model is None:
            raise RuntimeError(f"[{self.node_id}] call load_local_data() first")
        return self._model.get_weights(), self._n_train
