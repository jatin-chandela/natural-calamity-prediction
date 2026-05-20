"""
Hybrid CNN + LSTM model for spatio-temporal calamity prediction.

Architecture (planned for Phase 2):
    Input  -> Conv1D / Conv2D blocks  (spatial feature extraction)
           -> LSTM stack              (temporal dependency modelling)
           -> Dense + Softmax / Sigmoid (flood-risk / cyclone-class output)

Compiled with:
    - Optimizer: Adam
    - Loss: binary_crossentropy (flood) or categorical_crossentropy (cyclone)
    - Metrics: accuracy, precision, recall, F1 (via custom callback)

TODO: implement build_model(input_shape, num_classes) -> tf.keras.Model
TODO: expose get_initial_weights() helper used by the Flower server.
"""
