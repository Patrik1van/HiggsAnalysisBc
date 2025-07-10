import tensorflow as tf
import numpy as np

class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, logger, prefix=""):
        super().__init__()
        self.logger = logger
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        train_loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        train_mape = logs.get("mean_absolute_percentage_error", 0.0)
        val_mape = logs.get("val_mean_absolute_percentage_error", 0.0)
        train_mse = logs.get("mean_squared_error", 0.0)
        val_mse = logs.get("val_mean_squared_error", 0.0)

        self.logger.info(
            f"[Epoch {epoch + 1}] "
            f"Train loss: {train_loss:.4f}, MAPE: {train_mape:.2f}, MSE: {train_mse:.2f} | "
            f"Val loss: {val_loss:.4f}, MAPE: {val_mape:.2f}, MSE: {val_mse:.2f}"
        )

@tf.function
def pick_only_data(data, label):
    return data

@tf.function
def pick_only_target(data, label):
    return label

def extract_data(dataset):
    # Extract all elements from the tf.data.Dataset
    return [x for x in dataset.as_numpy_iterator()]

@tf.function
def pick_only_mmc(data, label):
    return data[-1]

def make_filter_slice(lower, upper):
    @tf.function
    def _filter_slice(data, target):
        return tf.logical_and(target >= lower, target < upper)
    return _filter_slice

