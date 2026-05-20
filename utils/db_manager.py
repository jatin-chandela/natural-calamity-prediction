"""SQLite persistence for predictions and federation round history."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                city             TEXT    NOT NULL,
                flood_risk       REAL    NOT NULL,
                cyclone_risk     REAL    NOT NULL,
                confidence_score REAL    NOT NULL,
                federation_round INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS federation_rounds (
                round_id        INTEGER PRIMARY KEY,
                timestamp       TEXT    NOT NULL,
                global_accuracy REAL,
                global_f1       REAL,
                num_clients     INTEGER,
                avg_loss        REAL
            );
        """)


def insert_prediction(
    city: str,
    flood_risk: float,
    cyclone_risk: float,
    confidence_score: float,
    federation_round: int,
    db_path: Path,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.execute(
            "INSERT INTO predictions "
            "(timestamp, city, flood_risk, cyclone_risk, confidence_score, federation_round) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, city, flood_risk, cyclone_risk, confidence_score, federation_round),
        )


def insert_round(
    round_id: int,
    global_accuracy: float,
    global_f1: float,
    num_clients: int,
    avg_loss: float,
    db_path: Path,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO federation_rounds "
            "(round_id, timestamp, global_accuracy, global_f1, num_clients, avg_loss) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (round_id, ts, global_accuracy, global_f1, num_clients, avg_loss),
        )


def get_latest_predictions(n: int = 50, db_path: Path = None) -> list[dict]:
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_round_history(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM federation_rounds ORDER BY round_id"
        ).fetchall()
    return [dict(r) for r in rows]


def export_to_csv(table_name: str, output_path: Path, db_path: Path) -> None:
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)  # noqa: S608
    df.to_csv(output_path, index=False)
