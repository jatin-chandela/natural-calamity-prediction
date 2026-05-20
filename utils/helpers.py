"""
Shared helper utilities used across server, clients, models, and dashboard.

Planned helpers (Phase 2+):
    - get_logger(name)           : project-wide structured logging.
    - load_config(path)          : YAML/JSON config loader.
    - set_global_seed(seed)      : deterministic NumPy / TF / Python seeding.
    - compute_metrics(y, y_hat)  : accuracy, precision, recall, F1.
    - save_round_metrics(...)    : persist FedAvg round stats to SQLite/Mongo.

TODO: implement helpers as the project moves into Phase 2.
"""
