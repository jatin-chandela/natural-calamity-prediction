"""Phase-2 data acquisition + preprocessing for the federated calamity-prediction project.

Downloads public NOAA archives, loads an IMD-style CSV, cleans + normalizes it,
builds 30-day sliding-window sequences, and writes .npy artifacts under
``data/processed/`` for the CNN+LSTM model.

Run from the project root::

    python -m data.fetch_data --skip-noaa --csv data/raw/sample_imd.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_data")

FEATURE_COLS = [
    "rainfall_mm",
    "temperature_c",
    "wind_speed_kmh",
    "humidity_percent",
    "pressure_hpa",
]
LABEL_COLS = ["flood_occurred", "cyclone_occurred"]
REQUIRED_COLS = ["date"] + FEATURE_COLS + LABEL_COLS

NOAA_SOURCES = {
    "ibtracs_north_indian.csv": (
        "https://www.ncei.noaa.gov/data/"
        "international-best-track-archive-for-climate-stewardship-ibtracs/"
        "v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
    ),
    # GHCN-Daily by-station CSV (Mumbai Santa Cruz – IN012010100). Public, no key.
    "ghcn_mumbai.csv": (
        "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/"
        "access/IN012010100.csv"
    ),
}


def download_noaa(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in NOAA_SOURCES.items():
        dest = raw_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            log.info("NOAA: %s already present, skipping.", fname)
            continue
        try:
            log.info("NOAA: downloading %s ...", url)
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            log.info("NOAA: saved -> %s (%d bytes)", dest, dest.stat().st_size)
        except requests.RequestException as exc:
            log.warning("NOAA: failed to fetch %s (%s). Continuing without it.", url, exc)


def load_imd_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"IMD CSV is missing required columns: {missing}. "
            f"Expected schema: {REQUIRED_COLS}"
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[REQUIRED_COLS]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill()
    for col in FEATURE_COLS:
        if df[col].isna().all():
            raise ValueError(f"Column '{col}' is entirely NaN after forward-fill.")
        df[col] = df[col].fillna(df[col].median())
    df[LABEL_COLS] = df[LABEL_COLS].fillna(0).astype(int)
    return df


def normalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    # NOTE: scaler fit on full series for simplicity; refit on train split during modeling.
    scaler = MinMaxScaler()
    scaled = df.copy()
    scaled[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS].values)
    return scaled, scaler


def make_sequences(
    df: pd.DataFrame, window: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = df[FEATURE_COLS].values.astype(np.float32)
    flood = df["flood_occurred"].values.astype(np.int8)
    cyclone = df["cyclone_occurred"].values.astype(np.int8)

    n_samples = len(df) - window
    if n_samples <= 0:
        raise ValueError(f"Not enough rows ({len(df)}) for window={window}.")

    X = np.empty((n_samples, window, len(FEATURE_COLS)), dtype=np.float32)
    y_flood = np.empty(n_samples, dtype=np.int8)
    y_cyclone = np.empty(n_samples, dtype=np.int8)
    for i in range(n_samples):
        X[i] = features[i : i + window]
        y_flood[i] = flood[i + window]
        y_cyclone[i] = cyclone[i + window]
    return X, y_flood, y_cyclone


def save_processed(
    X: np.ndarray,
    y_flood: np.ndarray,
    y_cyclone: np.ndarray,
    scaler: MinMaxScaler,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y_flood.npy", y_flood)
    np.save(out_dir / "y_cyclone.npy", y_cyclone)
    with (out_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    log.info(
        "Saved X=%s, y_flood=%s, y_cyclone=%s, scaler.pkl -> %s",
        X.shape, y_flood.shape, y_cyclone.shape, out_dir,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fetch + preprocess calamity data.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_root / "data" / "raw" / "sample_imd.csv",
        help="Path to IMD-style CSV with the required schema.",
    )
    parser.add_argument("--skip-noaa", action="store_true", help="Skip NOAA downloads.")
    parser.add_argument("--window", type=int, default=30, help="Sliding-window size.")
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "data" / "processed",
        help="Output directory for .npy artifacts.",
    )
    args = parser.parse_args()

    raw_dir = project_root / "data" / "raw"
    if not args.skip_noaa:
        download_noaa(raw_dir)

    log.info("Loading IMD CSV: %s", args.csv)
    df = load_imd_csv(args.csv)
    log.info("Loaded %d rows (%s to %s)", len(df), df["date"].min(), df["date"].max())

    df = clean(df)
    df, scaler = normalize(df)
    X, y_flood, y_cyclone = make_sequences(df, window=args.window)

    assert X.ndim == 3 and X.shape[1] == args.window and X.shape[2] == len(FEATURE_COLS)
    assert not np.isnan(X).any(), "NaNs leaked into X"
    assert X.min() >= 0.0 - 1e-6 and X.max() <= 1.0 + 1e-6, "MinMax range violated"

    save_processed(X, y_flood, y_cyclone, scaler, args.out)
    log.info(
        "Class balance — flood: %d/%d (%.2f%%), cyclone: %d/%d (%.2f%%)",
        int(y_flood.sum()), len(y_flood), 100 * y_flood.mean(),
        int(y_cyclone.sum()), len(y_cyclone), 100 * y_cyclone.mean(),
    )


if __name__ == "__main__":
    main()
