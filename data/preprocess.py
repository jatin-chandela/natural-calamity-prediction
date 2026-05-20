"""
Data ingestion and preprocessing.

Phase 2+ responsibilities:
    - Load raw CSVs from data/raw/ (IMD, NASA, NOAA, IoT exports).
    - Handle missing values, unit normalization, outlier capping.
    - Engineer features: rolling rainfall, pressure gradients, wind vectors.
    - Generate sliding-window sequences suitable for LSTM input.
    - Split the dataset into N non-IID shards (one per federated client).

TODO: implement load_raw(path) -> pandas.DataFrame
TODO: implement build_windows(df, window_size, horizon) -> (X, y)
TODO: implement split_for_clients(X, y, num_clients) -> list of shards
"""
