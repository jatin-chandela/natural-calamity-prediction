"""
Central FedAvg aggregator.

Phase 2+ responsibilities:
    - Spin up a Flower `ServerApp` that orchestrates federated rounds.
    - Configure the FedAvg strategy (client sampling, min available clients,
      evaluation frequency, fit/eval config dictionaries).
    - Persist the global model checkpoint after each round.
    - Stream round-level metrics to the dashboard / SQLite store.

TODO: implement Flower ServerApp + FedAvg strategy.
TODO: load server-side config from utils.helpers.load_config().
TODO: wire metric callbacks to utils logging.
"""
