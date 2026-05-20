"""
Unit tests for Phase 11.

Coverage:
  - federated_average()       (server/fedavg.py)
  - LocalClient.local_train() (clients/local_client.py)
  - Flask routes /  /predict  /dashboard  (dashboard/app.py)

Run:
    cd "C:\\Users\\Lenovo\\Desktop\\natural calamity prediction"
    python -m pytest tests/test_fedavg.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the project root is on sys.path so all internal imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from server.fedavg import federated_average  # noqa: E402
from clients.local_client import LocalClient  # noqa: E402

MUMBAI_CSV = PROJECT_ROOT / "clients" / "node_mumbai.csv"


# ── federated_average() ───────────────────────────────────────────────────────

class TestFederatedAverage:
    """Tests for the core FedAvg aggregation function."""

    def test_equal_weights_unchanged(self):
        """Two clients with identical weights and equal sizes → output equals input."""
        w = [np.array([1.0, 2.0], dtype=np.float32),
             np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
        result = federated_average([w, w], [100, 100])
        for got, expected in zip(result, w):
            np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_weighted_average_correct_value(self):
        """Numerical check: (200 * 0 + 100 * 3) / 300 = 1.0."""
        w0 = [np.array([0.0], dtype=np.float32)]
        w1 = [np.array([3.0], dtype=np.float32)]
        result = federated_average([w0, w1], [200, 100])
        np.testing.assert_allclose(result[0], np.array([1.0], dtype=np.float32), rtol=1e-5)

    def test_mismatched_layer_shapes_raises(self):
        """Clients with incompatible layer shapes must raise ValueError."""
        w0 = [np.array([1.0, 2.0], dtype=np.float32)]
        w1 = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]  # different shape
        with pytest.raises(ValueError):
            federated_average([w0, w1], [100, 100])


# ── LocalClient ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not MUMBAI_CSV.exists(),
    reason="node_mumbai.csv not present; skipping local client tests",
)
class TestLocalClient:
    """Tests for LocalClient.local_train() using the real Mumbai dataset."""

    def setup_method(self):
        self.client = LocalClient("mumbai", MUMBAI_CSV)
        self.client.load_local_data()
        # Seed the local model with its own initial weights so
        # receive_global_weights() is exercised before training.
        init_weights = self.client._model.get_weights()
        self.client.receive_global_weights(init_weights)

    def test_local_train_runs_without_error(self):
        """local_train() with epochs=1 must complete without raising."""
        self.client.local_train(epochs=1)
        weights, _ = self.client.get_model_update()
        assert isinstance(weights, list)
        assert len(weights) > 0

    def test_get_model_update_returns_correct_types(self):
        """get_model_update() must return (list[np.ndarray], positive int)."""
        self.client.local_train(epochs=1)
        weights, n = self.client.get_model_update()
        assert n > 0, "n_train must be positive"
        assert all(isinstance(w, np.ndarray) for w in weights), \
            "every weight tensor must be a numpy array"


# ── Flask routes ──────────────────────────────────────────────────────────────

class TestFlaskRoutes:
    """Tests that each Flask route returns HTTP 200."""

    def setup_method(self):
        from dashboard.app import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_index_returns_200(self):
        """`GET /` renders the landing page without requiring a model."""
        response = self.client.get("/")
        assert response.status_code == 200

    def test_predict_returns_200_with_city_keys(self):
        """`GET /predict` returns JSON with an entry for every city."""
        mock_model = MagicMock()
        # predict_city() calls model.predict() on a (7, 30, 5) batch → expects (7, 2)
        mock_model.predict.return_value = np.full((7, 2), 0.2, dtype=np.float32)
        with patch("dashboard.app.load_latest_model", return_value=mock_model):
            response = self.client.get("/predict")
        assert response.status_code == 200
        data = response.get_json()
        expected_cities = {"mumbai", "chennai", "kolkata", "bhubaneswar", "visakhapatnam"}
        assert expected_cities == set(data.keys())

    def test_dashboard_returns_200(self):
        """`GET /dashboard` renders the risk map and bar chart page."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.full((7, 2), 0.2, dtype=np.float32)
        with patch("dashboard.app.load_latest_model", return_value=mock_model):
            response = self.client.get("/dashboard")
        assert response.status_code == 200
