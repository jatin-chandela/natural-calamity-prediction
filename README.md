# Natural Calamity Prediction (Flood & Cyclone) using Federated Learning

B.Tech Project-I — Delhi Technological University, Department of Computer Science Engineering, AY 2025-26.

## Overview
A federated deep learning system that predicts floods and cyclones from distributed meteorological and IoT sensor data. Each local node (weather station / IoT cluster) trains a CNN+LSTM hybrid model on its private dataset and shares only model parameters with a central aggregator. The **Federated Averaging (FedAvg)** algorithm combines updates to form a global model, preserving privacy while reducing communication cost.

A Flask-based web dashboard visualizes real-time predictions and high-risk regions on an interactive map.

## Architecture
```
                   ┌─────────────────────────┐
                   │   Central Aggregator    │  (server/aggregator.py)
                   │   FedAvg via Flower     │
                   └────────┬──────┬─────────┘
                            │      │       (only model weights)
        ┌───────────────────┘      └───────────────────┐
        ▼                                              ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Client 1   │   │   Client 2   │…  │   Client N   │   (clients/client.py)
│  Local data  │   │  Local data  │   │  Local data  │
│  CNN+LSTM    │   │  CNN+LSTM    │   │  CNN+LSTM    │   (models/cnn_lstm.py)
└──────────────┘   └──────────────┘   └──────────────┘
                            │
                            ▼
                   ┌─────────────────────────┐
                   │   Flask Dashboard       │   (dashboard/app.py)
                   │   Folium + Plotly       │
                   └─────────────────────────┘
```

## Project Structure
```
.
├── server/         # Central aggregator (Flower ServerApp running FedAvg)
├── clients/        # Local node simulators (Flower NumPyClient)
├── models/         # CNN+LSTM hybrid model definitions
├── data/           # Preprocessing scripts; raw datasets under data/raw/
├── dashboard/      # Flask web app for live predictions & risk maps
├── utils/          # Shared helpers (logging, config, metrics)
├── requirements.txt
└── README.md
```

## Tech Stack
- **Language:** Python 3.10 / 3.11
- **ML / DL:** TensorFlow, Keras (CNN + LSTM hybrid)
- **Federated framework:** Flower (`flwr`) implementing FedAvg
- **Data:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Folium, Plotly
- **Web:** Flask
- **Storage:** SQLite (Python stdlib), MongoDB (`pymongo`) — optional, Phase 2+

> **Note on `sqlite3`:** it is part of Python's standard library and is **not** installed via pip. It is intentionally absent from `requirements.txt`. Just `import sqlite3` after installing Python.

## Data Sources (Phase 2+)
- Indian Meteorological Department (IMD) datasets
- NASA / NOAA open climate archives
- IoT sensor data: rainfall, temperature, wind speed, humidity

Place raw CSVs under `data/raw/` (gitignored).

## Setup
```powershell
# 1. Clone / open the project folder
cd "natural calamity prediction"

# 2. (Recommended) rename the folder to remove the space:
#    Rename-Item ..\"natural calamity prediction" natural_calamity_prediction

# 3. Create and activate a virtual environment (Python 3.10 or 3.11)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

### Platform notes
- **TensorFlow on Windows:** TF ≥ 2.11 has no native GPU support on Windows. Use CPU build, or run under WSL2 for GPU.
- **Python version:** stick to 3.10 or 3.11 for best TF + Flower compatibility.

## Running (Phase 2+ stubs — not yet functional)
```powershell
# Start the federated server
python -m server.aggregator

# Start clients (in separate terminals)
python -m clients.client --cid 0
python -m clients.client --cid 1

# Launch the dashboard
python -m dashboard.app
```

## Phase Roadmap
- **Phase 1 (current):** Project scaffold, dependency manifest, README. ✅
- **Phase 2:** Data ingestion + preprocessing pipeline; CNN+LSTM model definition.
- **Phase 3:** Flower client/server implementation; FedAvg training loop.
- **Phase 4:** Flask dashboard with Folium maps and Plotly charts.
- **Phase 5:** Evaluation (accuracy, precision, recall, F1) and final report.

## Team
| S.No | Name             | Roll No.     |
|------|------------------|--------------|
| 1    | Jatin            | 2K22/CO/224  |
| 2    | Jatin            | 2K22/CO/225  |
| 3    | Jay Dinesh Nimje | 2K22/CO/226  |

**Project Guide:** Dr. Moirangthem Biken Singh, Department of Computer Science Engineering, DTU.
