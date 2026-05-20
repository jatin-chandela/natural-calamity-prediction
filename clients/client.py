"""
Federated learning client (local node simulator).

Each client represents a regional weather station / IoT cluster that:
    - Loads its private shard of preprocessed data.
    - Receives global weights from the server.
    - Trains the local CNN+LSTM model for the configured number of epochs.
    - Returns updated weights (no raw data leaves the node).

Phase 2+ responsibilities:
    - Implement a `flwr.client.NumPyClient` subclass exposing
      get_parameters / fit / evaluate.
    - Parse a `--cid` CLI flag to choose the data shard.
    - Hook into models.cnn_lstm.build_model() for the local network.

TODO: implement NumPyClient and `flwr.client.start_client(...)`.
TODO: per-client data loader from data.preprocess.
"""
