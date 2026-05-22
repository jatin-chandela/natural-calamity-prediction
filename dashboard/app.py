"""Flask dashboard for the federated calamity-prediction project.

Routes
------
GET /           index page -- plain Folium map with 5 node markers
GET /predict    JSON -- flood_risk + cyclone_risk per city (last 7 sliding windows)
GET /dashboard  HTML -- color-coded map + Plotly bar chart + readings table
GET /history    HTML -- Plotly line chart of accuracy & F1 across federation rounds

Run from project root:
    python dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template

# project root on sys.path so 'utils.*' resolves
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.db_manager import init_db, insert_prediction  # noqa: E402

app = Flask(__name__)

# ── static config ─────────────────────────────────────────────────────────────
LOG_PATH         = ROOT / "logs" / "training_log.csv"
PREDICTIONS_PATH = ROOT / "dashboard" / "predictions.json"
DB_PATH          = ROOT / "logs" / "calamity_predictions.db"
init_db(DB_PATH)

CITIES: dict[str, dict] = {
    "mumbai":        {"lat": 19.0760, "lon": 72.8777, "csv": ROOT / "clients" / "node_mumbai.csv"},
    "chennai":       {"lat": 13.0827, "lon": 80.2707, "csv": ROOT / "clients" / "node_chennai.csv"},
    "kolkata":       {"lat": 22.5726, "lon": 88.3639, "csv": ROOT / "clients" / "node_kolkata.csv"},
    "bhubaneswar":   {"lat": 20.2961, "lon": 85.8245, "csv": ROOT / "clients" / "node_bhubaneswar.csv"},
    "visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "csv": ROOT / "clients" / "node_visakhapatnam.csv"},
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_cached_predictions() -> dict:
    """Read precomputed predictions from predictions.json.

    The model runs offline (TensorFlow is not installed on the server);
    regenerate this file with dashboard/precompute_predictions.py.
    """
    with open(PREDICTIONS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def risk_color(v: float) -> str:
    if v < 0.30:
        return "green"
    if v < 0.60:
        return "orange"
    return "red"


def risk_class(v: float) -> str:
    if v < 0.30:
        return "risk-green"
    if v < 0.60:
        return "risk-orange"
    return "risk-red"


def build_map(city_risks: dict | None = None) -> str:
    """Return Folium map HTML string. Pass city_risks for color-coded markers."""
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
    for city, cfg in CITIES.items():
        if city_risks:
            fr = city_risks[city]["flood_risk"]
            cr = city_risks[city]["cyclone_risk"]
            color = risk_color(max(fr, cr))
            popup = (
                f"<b>{city.title()}</b><br>"
                f"Flood Risk: {fr:.1%}<br>"
                f"Cyclone Risk: {cr:.1%}"
            )
        else:
            color = "blue"
            popup = f"<b>{city.title()}</b><br>Weather Station Node"
        folium.CircleMarker(
            location=[cfg["lat"], cfg["lon"]],
            radius=13,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=220),
            tooltip=city.title(),
        ).add_to(m)
    return m._repr_html_()


def latest_readings() -> dict:
    rows = {}
    for city, cfg in CITIES.items():
        last = pd.read_csv(cfg["csv"]).iloc[-1]
        rows[city] = {
            "date":             str(last["date"]),
            "rainfall_mm":      round(float(last["rainfall_mm"]),      2),
            "temperature_c":    round(float(last["temperature_c"]),    2),
            "wind_speed_kmh":   round(float(last["wind_speed_kmh"]),   2),
            "humidity_percent": round(float(last["humidity_percent"]), 2),
            "pressure_hpa":     round(float(last["pressure_hpa"]),     2),
        }
    return rows


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    map_html = build_map()
    return render_template("index.html", map_html=map_html)


@app.route("/predict")
def predict():
    try:
        data = load_cached_predictions()
    except FileNotFoundError:
        return jsonify({"error": "predictions.json not found - run dashboard/precompute_predictions.py"}), 500

    city_risks = data["city_risks"]
    fed_round = data.get("fed_round", 0)
    for city, risk in city_risks.items():
        try:
            insert_prediction(city, risk["flood_risk"], risk["cyclone_risk"], 0.0, fed_round, DB_PATH)
        except Exception:
            pass
    return jsonify(city_risks)


@app.route("/dashboard")
def dashboard():
    try:
        data = load_cached_predictions()
    except Exception as exc:
        return render_template("error.html", message=str(exc)), 500

    city_risks = data["city_risks"]

    map_html = build_map(city_risks)

    names     = [c.title() for c in CITIES]
    flood_v   = [city_risks[c]["flood_risk"]   for c in CITIES]
    cyclone_v = [city_risks[c]["cyclone_risk"] for c in CITIES]

    fig = go.Figure(data=[
        go.Bar(
            name="Flood Risk", x=names, y=flood_v,
            marker_color=[risk_color(v) for v in flood_v],
        ),
        go.Bar(
            name="Cyclone Risk", x=names, y=cyclone_v,
            marker_color=[risk_color(v) for v in cyclone_v],
        ),
    ])
    fig.update_layout(
        barmode="group",
        title="Risk Levels by City (mean of last 7 predictions)",
        yaxis=dict(title="Risk Probability", range=[0, 1]),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    bar_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    readings     = latest_readings()
    last_updated = data.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return render_template(
        "dashboard.html",
        map_html=map_html,
        bar_html=bar_html,
        city_risks=city_risks,
        risk_class=risk_class,
        readings=readings,
        last_updated=last_updated,
    )


@app.route("/history")
def history():
    df = pd.read_csv(LOG_PATH)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["round"], y=df["accuracy"],
        mode="lines+markers", name="Accuracy",
        line=dict(color="#0d6efd", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df["round"], y=df["f1"],
        mode="lines+markers", name="F1 Score",
        line=dict(color="#fd7e14", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df["round"], y=df["loss"],
        mode="lines+markers", name="Loss",
        line=dict(color="#dc3545", width=2, dash="dot"),
        marker=dict(size=5),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Federated Model Performance Across Rounds",
        xaxis=dict(title="Federation Round", dtick=1),
        yaxis=dict(title="Score", range=[0, 1], side="left"),
        yaxis2=dict(title="Loss", overlaying="y", side="right", showgrid=False),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    rounds = df.to_dict("records")
    return render_template("history.html", chart_html=chart_html, rounds=rounds)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
