"""Run the trained model once and cache predictions to JSON.

Run this LOCALLY (where TensorFlow is installed) before deploying:

    python dashboard/precompute_predictions.py

It writes dashboard/predictions.json. The deployed Flask app reads that
file instead of running TensorFlow, because Render's free tier does not
have enough RAM to load TF. Re-run this script whenever the model weights
or the input CSVs change, then commit the updated predictions.json.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.fetch_data import FEATURE_COLS, clean, load_imd_csv, normalize  # noqa: E402

MODELS_DIR = ROOT / "models" / "global"
OUT_PATH = ROOT / "dashboard" / "predictions.json"

CITIES = {
    "mumbai":        ROOT / "clients" / "node_mumbai.csv",
    "chennai":       ROOT / "clients" / "node_chennai.csv",
    "kolkata":       ROOT / "clients" / "node_kolkata.csv",
    "bhubaneswar":   ROOT / "clients" / "node_bhubaneswar.csv",
    "visakhapatnam": ROOT / "clients" / "node_visakhapatnam.csv",
}

WINDOW = 30
N_DAYS = 7


def _round_num(path: Path) -> int:
    m = re.search(r"global_round_(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def main() -> None:
    from models.hybrid_model import build_model  # imports TensorFlow

    checkpoints = sorted(MODELS_DIR.glob("global_round_*.weights.h5"), key=_round_num)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {MODELS_DIR}")

    model = build_model()
    model.load_weights(str(checkpoints[-1]))
    fed_round = _round_num(checkpoints[-1])

    city_risks: dict[str, dict] = {}
    for city, csv in CITIES.items():
        df = load_imd_csv(csv)
        df = clean(df)
        df, _ = normalize(df)
        tail = df.tail(WINDOW + N_DAYS - 1).reset_index(drop=True)
        features = tail[FEATURE_COLS].values.astype("float32")
        seqs = np.stack([features[i : i + WINDOW] for i in range(N_DAYS)])
        probs = model.predict(seqs, verbose=0)
        city_risks[city] = {
            "flood_risk":   round(float(probs[:, 0].mean()), 4),
            "cyclone_risk": round(float(probs[:, 1].mean()), 4),
        }
        print(f"  {city:14s} flood={city_risks[city]['flood_risk']:.4f} "
              f"cyclone={city_risks[city]['cyclone_risk']:.4f}")

    payload = {
        "fed_round":    fed_round,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city_risks":   city_risks,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH}  (round {fed_round})")


if __name__ == "__main__":
    main()
