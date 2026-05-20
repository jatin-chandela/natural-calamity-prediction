"""
Phase 10 — Final Evaluation Script.

Loads the final federated global model, evaluates it against a centralized
ML baseline, and produces all artifacts needed for the project report:
  - outputs/confusion_matrix_flood.png
  - outputs/confusion_matrix_cyclone.png
  - outputs/roc_curves.png
  - outputs/final_report.txt

Run from the project root:
    python -m evaluation.evaluate
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.fetch_data import clean, load_imd_csv, make_sequences, normalize
from models.hybrid_model import build_model

CITIES = ["bhubaneswar", "chennai", "kolkata", "mumbai", "visakhapatnam"]
WINDOW = 30
VAL_FRACTION = 0.2
CENTRALIZED_EPOCHS = 10
OUTPUTS_DIR = ROOT / "outputs"
WEIGHTS_PATH = ROOT / "models" / "global" / "global_round_20.weights.h5"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_city_data(city: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) for one city using the same
    80/20 time-ordered split as LocalClient.load_local_data()."""
    csv_path = ROOT / "clients" / f"node_{city}.csv"
    df = load_imd_csv(csv_path)
    df = clean(df)
    df, _ = normalize(df)
    X, y_flood, y_cyclone = make_sequences(df, window=WINDOW)
    y = np.column_stack([y_flood, y_cyclone]).astype(np.float32)

    split = int(len(X) * (1.0 - VAL_FRACTION))
    return X[:split], y[:split], X[split:], y[split:]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Step 1 & 2 — Load global model + per-city held-out data
# ---------------------------------------------------------------------------

def load_global_model():
    model = build_model()
    model.load_weights(str(WEIGHTS_PATH))
    return model


def load_all_cities() -> dict[str, dict]:
    city_data = {}
    for city in CITIES:
        X_tr, y_tr, X_te, y_te = load_city_data(city)
        city_data[city] = {
            "X_train": X_tr, "y_train": y_tr,
            "X_test":  X_te, "y_test":  y_te,
        }
    return city_data


# ---------------------------------------------------------------------------
# Step 3 — Evaluate federated model per city + pooled
# ---------------------------------------------------------------------------

def evaluate_federated(model, city_data: dict[str, dict]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    per_city_metrics: dict[str, dict] = {}
    all_X_test = []
    all_y_test = []

    for city, d in city_data.items():
        X_te, y_te = d["X_test"], d["y_test"]
        y_prob = model.predict(X_te, verbose=0)
        y_pred = (y_prob >= 0.5).astype(int)
        per_city_metrics[city] = compute_metrics(y_te, y_pred)
        all_X_test.append(X_te)
        all_y_test.append(y_te)

    X_pool = np.concatenate(all_X_test, axis=0)
    y_pool = np.concatenate(all_y_test, axis=0)
    y_prob_pool = model.predict(X_pool, verbose=0)
    return per_city_metrics, y_pool, y_prob_pool, (y_prob_pool >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Step 4 — Centralized baseline
# ---------------------------------------------------------------------------

def train_centralized_baseline(city_data: dict[str, dict]) -> tuple[dict, np.ndarray, np.ndarray]:
    X_train_all = np.concatenate([d["X_train"] for d in city_data.values()], axis=0)
    y_train_all = np.concatenate([d["y_train"] for d in city_data.values()], axis=0)
    X_test_all  = np.concatenate([d["X_test"]  for d in city_data.values()], axis=0)
    y_test_all  = np.concatenate([d["y_test"]  for d in city_data.values()], axis=0)

    model = build_model()
    model.fit(X_train_all, y_train_all, epochs=CENTRALIZED_EPOCHS, batch_size=32, verbose=0)

    y_prob = model.predict(X_test_all, verbose=0)
    y_pred = (y_prob >= 0.5).astype(int)
    return compute_metrics(y_test_all, y_pred), y_test_all, y_prob


# ---------------------------------------------------------------------------
# Step 6 — Confusion matrices
# ---------------------------------------------------------------------------

def save_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    task_names = ["Flood", "Cyclone"]
    file_names = ["confusion_matrix_flood.png", "confusion_matrix_cyclone.png"]

    for i, (task, fname) in enumerate(zip(task_names, file_names)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {task} Prediction\n(Federated Global Model, Pooled Test Set)")
        fig.tight_layout()
        out_path = OUTPUTS_DIR / fname
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Step 8 — ROC curves
# ---------------------------------------------------------------------------

def save_roc_curves(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "darkorange"]
    task_names = ["Flood", "Cyclone"]

    for i, (task, color) in enumerate(zip(task_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{task} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Federated Global Model (Pooled Test Set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = OUTPUTS_DIR / "roc_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Step 5 & 7 — Report
# ---------------------------------------------------------------------------

def build_report(
    fed_metrics_per_city: dict[str, dict],
    fed_metrics_pooled: dict,
    central_metrics: dict,
    y_true_pool: np.ndarray,
    y_pred_pool: np.ndarray,
) -> str:
    buf = StringIO()

    def p(line: str = "") -> None:
        print(line)
        buf.write(line + "\n")

    separator = "=" * 72

    p(separator)
    p("NATURAL CALAMITY PREDICTION — FINAL EVALUATION REPORT")
    p("Delhi Technological University | B.Tech Project-I | AY 2025-26")
    p(separator)

    # Comparison table
    p()
    p("COMPARISON TABLE: Centralized ML Baseline vs Federated Global Model")
    p("-" * 72)
    header = f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    p(header)
    p("-" * 72)

    def fmt_row(name: str, m: dict) -> str:
        return (
            f"{name:<30} {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
            f" {m['recall']:>10.4f} {m['f1']:>10.4f}"
        )

    p(fmt_row("Centralized ML Baseline", central_metrics))
    p(fmt_row("Federated Global Model", fed_metrics_pooled))
    p("-" * 72)

    # Per-city breakdown
    p()
    p("PER-CITY RESULTS — Federated Global Model (Held-Out 20% per node)")
    p("-" * 72)
    p(header)
    p("-" * 72)
    for city, m in fed_metrics_per_city.items():
        p(fmt_row(city.capitalize(), m))
    p("-" * 72)

    # Full sklearn classification report — Flood
    p()
    p("CLASSIFICATION REPORT — Flood Prediction (Federated, Pooled Test Set)")
    p("-" * 72)
    p(classification_report(
        y_true_pool[:, 0], y_pred_pool[:, 0],
        target_names=["No Flood", "Flood"],
        zero_division=0,
    ))

    # Full sklearn classification report — Cyclone
    p()
    p("CLASSIFICATION REPORT — Cyclone Prediction (Federated, Pooled Test Set)")
    p("-" * 72)
    p(classification_report(
        y_true_pool[:, 1], y_pred_pool[:, 1],
        target_names=["No Cyclone", "Cyclone"],
        zero_division=0,
    ))

    p(separator)
    p("END OF REPORT")
    p(separator)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("\n[1/6] Loading city data ...")
    city_data = load_all_cities()
    for city, d in city_data.items():
        print(f"  {city}: train={len(d['X_train'])} | test={len(d['X_test'])}")

    print(f"\n[2/6] Loading global model from {WEIGHTS_PATH.name} ...")
    global_model = load_global_model()

    print("\n[3/6] Evaluating federated model per city ...")
    fed_per_city, y_true_pool, y_prob_pool, y_pred_pool = evaluate_federated(global_model, city_data)
    fed_pooled = compute_metrics(y_true_pool, y_pred_pool)

    print("\n[4/6] Training centralized baseline (10 epochs) ...")
    central_metrics, y_true_central, y_prob_central = train_centralized_baseline(city_data)

    print("\n[5/6] Saving confusion matrices ...")
    save_confusion_matrices(y_true_pool, y_pred_pool)

    print("\n[5/6] Saving ROC curves ...")
    save_roc_curves(y_true_pool, y_prob_pool)

    print("\n[6/6] Writing final report ...")
    report = build_report(fed_per_city, fed_pooled, central_metrics, y_true_pool, y_pred_pool)
    report_path = OUTPUTS_DIR / "final_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")

    print("\n" + "=" * 72)
    print(report)


if __name__ == "__main__":
    main()
