# System Architecture — Natural Calamity Prediction (Flood & Cyclone) Using Federated Learning

**Project:** B.Tech Project-I, Delhi Technological University, AY 2025-26  
**Team:** Jatin (2K22/CO/224), Jatin (2K22/CO/225), Jay Dinesh Nimje (2K22/CO/226)  
**Guide:** Dr. Moirangthem Biken Singh

---

## 1. System Architecture

The system follows a **client-server federated learning** topology. Five geographically distributed nodes act as local clients; a single central server performs model aggregation. Raw meteorological data never leaves a node.

### Components

#### Local Clients (5 Nodes)
Each node simulates a coastal weather station. It owns its private dataset, trains a local CNN+LSTM model, and shares only model weights with the server.

| Node | City | Location |
|------|------|----------|
| Node 1 | Mumbai | 19.08°N, 72.88°E |
| Node 2 | Chennai | 13.08°N, 80.27°E |
| Node 3 | Kolkata | 22.57°N, 88.36°E |
| Node 4 | Bhubaneswar | 20.30°N, 85.82°E |
| Node 5 | Visakhapatnam | 17.69°N, 83.22°E |

Each node ingests a time-series CSV with six meteorological features:
`rainfall_mm`, `temperature_c`, `wind_speed_kmh`, `humidity_percent`, `pressure_hpa`, plus binary labels `flood_occurred` and `cyclone_occurred`.

#### Central Aggregation Server
Implemented in `server/fedavg.py` as `FederationServer`. Responsibilities:
- Build and distribute the initial global model
- Collect per-client weight updates after each local training round
- Aggregate updates using the FedAvg algorithm
- Evaluate the global model on a pooled held-out validation set
- Persist model checkpoints and training metrics

#### Flask Dashboard
Implemented in `dashboard/app.py`. Loads the latest global model checkpoint and provides four HTTP routes:

| Route | Purpose |
|-------|---------|
| `GET /` | Landing page with Folium node map |
| `GET /predict` | JSON endpoint — flood & cyclone risk per city |
| `GET /dashboard` | Color-coded risk map + Plotly bar chart + sensor readings |
| `GET /history` | Training curves (accuracy, F1, loss) across federation rounds |

#### Persistence
- **SQLite** (`logs/calamity_predictions.db`): prediction history and per-round federation metrics
- **CSV** (`logs/training_log.csv`): round-level training log (loss, accuracy, precision, recall, F1)
- **HDF5** (`models/global/global_round_N.weights.h5`): model checkpoints, one per round

---

## 2. Data Flow Diagram

```
  ╔══════════════════════════════════════════════════════════════════╗
  ║                     FEDERATED TRAINING LOOP                     ║
  ╚══════════════════════════════════════════════════════════════════╝

  CENTRAL SERVER
  ┌─────────────────────────────────────────────────────────────┐
  │  FederationServer                                           │
  │  1. initialize_global_model()  →  w_global (round 0)       │
  │                                                             │
  │  For each round r = 1 … 20:                                 │
  │    a. distribute_weights_to_clients() ──────────────────┐  │
  │                                                          │  │
  │    e. collect_and_aggregate(updates) ◄──────────────┐   │  │
  │       federated_average(weights, sizes)             │   │  │
  │       → w_global (round r)                          │   │  │
  │                                                     │   │  │
  │    f. evaluate_global_model(X_val_pool, y_val_pool) │   │  │
  │    g. save_global_model(r)                          │   │  │
  │    h. log_round(r, metrics, n_clients)              │   │  │
  └─────────────────────────────────────────────────────┼───┘  │
             global weights (w_global) broadcast        │      │
             ─────────────────────────────────────      │      │
             ↓              ↓              ↓            │      │
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │      │
  │  Node:       │ │  Node:       │ │  Node:       │   │      │
  │  Mumbai      │ │  Chennai     │ │  Kolkata … 5 │   │      │
  │              │ │              │ │              │   │      │
  │  Private CSV │ │  Private CSV │ │  Private CSV │   │      │
  │  (raw data)  │ │  (raw data)  │ │  (raw data)  │   │      │
  │   ↓          │ │   ↓          │ │   ↓          │   │      │
  │  Preprocess  │ │  Preprocess  │ │  Preprocess  │   │      │
  │  (normalize, │ │  (normalize, │ │  (normalize, │   │      │
  │   window)    │ │   window)    │ │   window)    │   │      │
  │   ↓          │ │   ↓          │ │   ↓          │   │      │
  │  b. receive  │ │  b. receive  │ │  b. receive  │   │      │
  │   w_global   │ │   w_global   │ │   w_global   │   │      │
  │   ↓          │ │   ↓          │ │   ↓          │   │      │
  │  c. local_   │ │  c. local_   │ │  c. local_   │   │      │
  │   train(5ep) │ │   train(5ep) │ │   train(5ep) │   │      │
  │   ↓          │ │   ↓          │ │   ↓          │   │      │
  │  d. get_     │ │  d. get_     │ │  d. get_     │   │      │
  │   model_     │ │   model_     │ │   model_     │   │      │
  │   update()   │ │   update()   │ │   update()   │   │      │
  │  (w_k, n_k)  │ │  (w_k, n_k)  │ │  (w_k, n_k)  │   │      │
  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │      │
         │ weights only   │ weights only   │            │      │
         └────────────────┴────────────────┴────────────┘      │
                                                               │
  ┌──────────────────────────────────────────────────────────┐  │
  │  INFERENCE (Flask Dashboard)                              │  │
  │                                                           │  │
  │  load_latest_model()  ←  models/global/global_round_20   │◄─┘
  │  predict_city(model, city)                                │
  │    last 7 sliding windows of 30 days each                 │
  │    → mean flood_risk, mean cyclone_risk, confidence       │
  │  build_map(city_risks)  →  Folium HTML                    │
  │  insert_prediction(...)  →  SQLite                        │
  └──────────────────────────────────────────────────────────┘

  MODEL I/O
  ─────────
  Input  : (batch, 30, 5)  — 30-day window, 5 meteorological features
  Output : (batch, 2)      — [P(flood), P(cyclone)] via sigmoid

  CNN+LSTM ARCHITECTURE
  ─────────────────────
  Input(30, 5)
    → Conv1D(64, k=3, relu) → MaxPool1D(2)
    → Conv1D(32, k=3, relu)
    → LSTM(128, return_sequences=True)
    → LSTM(64)
    → Dense(32, relu) → Dropout(0.3)
    → Dense(2, sigmoid)
```

---

## 3. Why FedAvg Was Chosen

**FedAvg** (McMahan et al., 2017, *"Communication-Efficient Learning of Deep Networks from Decentralized Data"*) was selected for four reasons:

### 3.1 Algorithmic Simplicity
FedAvg reduces to a weighted average of client weight tensors:

```
w_global[l] = Σ_k  (n_k / N) × w_k[l]

where  N = Σ_k n_k   (total training samples across all clients)
       n_k            (training-set size of client k)
       w_k[l]         (weight tensor at layer l of client k's model)
```

The entire server-side aggregation fits in ~25 lines (`server/fedavg.py:48-94`). Every step is auditable without a framework dependency.

### 3.2 Robustness to Non-IID Data
The five coastal cities have substantially different meteorological distributions — Mumbai experiences southwest monsoon while Chennai has northeast monsoon; Visakhapatnam has the highest historical cyclone frequency. FedAvg's sample-size weighting dampens the influence of small or anomalous nodes, providing reasonable convergence even on non-IID data shards.

### 3.3 Low Communication Cost
Each communication round exchanges only model weights (~a few MB for the CNN+LSTM) rather than raw time-series datasets (~1 MB per city CSV × 3 years). For a 5-node system with 5 meteorological features this ratio is negligible, but it demonstrates the principle that scales to thousands of IoT sensors.

### 3.4 Compatibility with the CNN+LSTM Architecture
FedAvg requires no modification to the local training procedure. Each node runs standard Keras `model.fit()`; the server calls `model.get_weights()` / `model.set_weights()`. This means the CNN+LSTM hybrid (`models/hybrid_model.py`) is unchanged between centralized and federated settings, simplifying debugging and baseline comparisons.

---

## 4. Privacy Guarantees

### What Stays at the Node
The raw meteorological CSV (`clients/node_*.csv`) and all intermediate preprocessed tensors (`X_train`, `y_train`) are never serialized, transmitted, or logged outside the `LocalClient` instance. The `get_model_update()` method (`local_client.py:81-88`) returns only:
- `weights`: a list of NumPy arrays (the trained model parameters)
- `n_train`: an integer count of training samples

Neither value contains individual sensor readings.

### What the Server Receives
The `FederationServer.collect_and_aggregate()` method receives a list of `{"weights": [...], "n_samples": int, "client_id": str}` dicts. There is no raw-data field. The server averages the weights and discards the individual client updates.

### Reconstruction Resistance
Under the standard assumption that the CNN+LSTM is not invertible (no closed-form inversion of a non-linear deep network), an adversary who intercepts the weight tensors cannot reconstruct the original time-series. The server itself never issues queries that could constitute a membership-inference or gradient-inversion attack.

### Limitations of the Current Privacy Model
The current system provides **data minimization** but not formal differential privacy. A sophisticated gradient-inversion attack (e.g., Geiping et al., 2020) could potentially recover approximate training sequences from a single-round weight delta. The mitigations (differential privacy noise, secure aggregation) are listed in Section 5.

---

## 5. Limitations and Future Work

### Current Limitations

| Area | Limitation |
|------|-----------|
| **Data** | All five city datasets are synthetically generated (`data/generate_synthetic_data.py`). Real IMD/NASA/NOAA ingestion (`data/fetch_data.py:download_noaa`) is implemented but requires API credentials not committed to the repo. |
| **Privacy** | No differential privacy (DP) noise is injected into client updates. A gradient-inversion adversary with access to both the pre- and post-round global weights could estimate training data. |
| **Secure Aggregation** | The aggregation step (`server/fedavg.py:federated_average`) operates on plaintext weights. There is no cryptographic secure aggregation (e.g., Bonawitz et al., 2017). |
| **FL Framework Integration** | `clients/client.py` and `server/aggregator.py` are stubs. The Flower (`flwr`) library is installed but not wired; the production path would use `flwr.client.NumPyClient` and `flwr.server.strategy.FedAvg`. |
| **Partial Stubs** | `utils/helpers.py` (structured logging, YAML config, seeding) and `data/preprocess.py` (non-IID sharding) are unimplemented placeholders. |
| **F1 at Round 1** | F1 score is 0.000 at round 1 due to class imbalance — the model predicts all-negative before sufficient training; no class weighting or oversampling is applied. |

### Future Work

1. **Differential Privacy** — Add Gaussian noise to weight updates (`σ` tuned by privacy budget `ε`) using TensorFlow Privacy or Opacus before aggregation.
2. **Secure Aggregation** — Implement secret-sharing-based secure sum (Bonawitz et al.) so the server never sees individual client weight tensors in plaintext.
3. **Real Data Ingestion** — Connect `data/fetch_data.py:download_noaa()` to live IMD and NOAA IBTRACS feeds with scheduled refresh.
4. **Flower Integration** — Complete `clients/client.py` and `server/aggregator.py` stubs to enable a production deployment where each city node is an independent process communicating over gRPC.
5. **Personalized Federated Learning** — Replace global FedAvg with FedProx or per-client fine-tuning (FedPer) to handle the severe non-IID distribution across Indian coastal cities.
6. **Real-Time IoT Ingestion** — Replace the static CSV source with a streaming pipeline (Kafka / MQTT) feeding live rainfall, wind, and pressure sensor readings into the local preprocessing stage.
7. **Alert Integration** — Wire the `/predict` endpoint to an SMS/email gateway (e.g., Twilio) to dispatch automated early warnings when `flood_risk > 0.6` or `cyclone_risk > 0.6`.
