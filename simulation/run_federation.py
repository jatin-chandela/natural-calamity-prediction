"""
Federated learning simulation — 5 city nodes, 20 rounds of FedAvg.

Run from the project root:
    python -m simulation.run_federation

Outputs:
    outputs/training_curve.png   — accuracy & F1 across rounds
    logs/training_log.csv        — per-round metrics (appended)
    models/global/               — weight checkpoints per round
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clients.local_client import LocalClient
from server.fedavg import FederationServer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CITIES: dict[str, str] = {
    "bhubaneswar":   "clients/node_bhubaneswar.csv",
    "chennai":       "clients/node_chennai.csv",
    "kolkata":       "clients/node_kolkata.csv",
    "mumbai":        "clients/node_mumbai.csv",
    "visakhapatnam": "clients/node_visakhapatnam.csv",
}

NUM_ROUNDS: int = 20
LOCAL_EPOCHS: int = 5
WINDOW: int = 30
OUTPUT_DIR = _ROOT / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header() -> None:
    print(
        f"\n{'Round':>6}  {'Accuracy':>10}  {'Precision':>10}  "
        f"{'Recall':>8}  {'F1':>8}  {'Loss':>10}"
    )
    print("-" * 62)


def _print_row(rnd: int, m: dict) -> None:
    print(
        f"{rnd:>6}  {m['accuracy']:>10.4f}  {m['precision']:>10.4f}  "
        f"{m['recall']:>8.4f}  {m['f1']:>8.4f}  {m['loss']:>10.4f}"
    )


def _save_plot(rounds: list[int], accuracies: list[float], f1s: list[float]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, accuracies, marker="o", linewidth=2, label="Accuracy")
    ax.plot(rounds, f1s, marker="s", linewidth=2, linestyle="--", label="F1 (macro)")
    ax.set_xlabel("Federation Round")
    ax.set_ylabel("Score")
    ax.set_title("Federated Learning — Global Model Performance Across Rounds")
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    out = OUTPUT_DIR / "training_curve.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTraining curve saved -> {out}")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 62)
    print("  Natural Calamity Prediction - Federated Simulation")
    print(f"  Nodes: {len(CITIES)}   Rounds: {NUM_ROUNDS}   Local epochs: {LOCAL_EPOCHS}")
    print("=" * 62)

    # --- 1. Initialise clients -------------------------------------------
    print("\n[Setup] Loading local data for each node...")
    clients: list[LocalClient] = []
    for node_id, csv_rel in CITIES.items():
        csv_path = _ROOT / csv_rel
        c = LocalClient(node_id, csv_path)
        c.load_local_data(window=WINDOW)
        clients.append(c)
        print(f"  {node_id:<18} train={c._n_train}")

    # --- 2. Build cross-city validation pool (held-out, never trained on) --
    X_val_pool = np.concatenate([c._X_val for c in clients], axis=0)
    y_val_pool = np.concatenate([c._y_val for c in clients], axis=0)
    print(f"\n[Setup] Validation pool: {len(X_val_pool)} sequences from all nodes")

    # --- 3. Initialise server --------------------------------------------
    server = FederationServer(
        model_dir=str(_ROOT / "models" / "global"),
        log_path=str(_ROOT / "logs" / "training_log.csv"),
    )
    global_weights = server.initialize_global_model()
    print("[Setup] Global model initialised\n")

    # --- 4. Federation rounds --------------------------------------------
    round_nums: list[int] = []
    round_accuracies: list[float] = []
    round_f1s: list[float] = []

    _print_header()

    for rnd in range(1, NUM_ROUNDS + 1):
        global_weights = server.distribute_weights_to_clients()

        client_updates: list[dict] = []
        for c in clients:
            c.receive_global_weights(global_weights)
            c.local_train(epochs=LOCAL_EPOCHS)
            weights, n_samples = c.get_model_update()
            client_updates.append(
                {"weights": weights, "n_samples": n_samples, "client_id": c.node_id}
            )

        server.collect_and_aggregate(client_updates)
        metrics = server.evaluate_global_model(X_val_pool, y_val_pool)
        server.log_round(rnd, metrics, n_clients=len(clients))
        server.save_global_model(rnd)

        _print_row(rnd, metrics)
        round_nums.append(rnd)
        round_accuracies.append(metrics["accuracy"])
        round_f1s.append(metrics["f1"])

    # --- 5. Summary & plot -----------------------------------------------
    print("-" * 62)
    print(f"\nBest accuracy : {max(round_accuracies):.4f}  (round {round_accuracies.index(max(round_accuracies)) + 1})")
    print(f"Best F1       : {max(round_f1s):.4f}  (round {round_f1s.index(max(round_f1s)) + 1})")

    _save_plot(round_nums, round_accuracies, round_f1s)
    print(f"Round logs    → {_ROOT / 'logs' / 'training_log.csv'}")
    print(f"Model weights → {_ROOT / 'models' / 'global'}/")
    print("\nDone.")


if __name__ == "__main__":
    run()
