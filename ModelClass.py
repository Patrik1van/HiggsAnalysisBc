import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from DatasetClass import Dataset

from tensorflow.keras.layers import Normalization, Input, Dense, BatchNormalization, Dropout, Activation, Layer
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsolutePercentageError
from src.helpers import pick_only_data, EpochLogger
import logging

# Logger zapisuje len do súboru, nie do stdout
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)


class Augmentation(Layer):
    def __init__(self, phi_mask, lorentz_mask, augmentation, **kwargs):
        super(Augmentation, self).__init__(**kwargs)

        self.phi_mask = tf.constant(phi_mask, dtype=tf.bool)
        self.lorentz_mask = tf.constant(lorentz_mask, dtype = tf.bool)
        self.augmentation = augmentation

        if self.augmentation == "lorentz":
            self.lorentz_indices_original = tf.cast(tf.where(self.lorentz_mask), dtype=tf.int32)
            self.n_lorentz_features = tf.shape(self.lorentz_indices_original)[0]
            self.n_vectors_per_batch_item = self.n_lorentz_features // 4
            self.scatter_indices_template = self.lorentz_indices_original# [n_vectors, 4]


    def call(self, inputs, training=None):
        if training is not True: 
            return inputs
        
        if self.augmentation == "phi":
            return self._augment_phi(inputs)
        
        elif self.augmentation == "lorentz":
            return self._augment_lorentz(inputs)
        else: 
            raise ValueError(f"Unknown augmentation technique: {self.augmentation}")
        
    def _augment_phi(self, inputs):
            angle = tf.random.uniform(shape=(tf.shape(inputs)[0], 1), minval=-np.pi, maxval=np.pi)
            data  = tf.where(self.phi_mask, inputs + angle, inputs)
            data = tf.where(self.phi_mask, tf.math.atan2(tf.sin(data), tf.cos(data)), data)

            return data 

    def _augment_lorentz(self, inputs):
        batch_size = tf.shape(inputs)[0]
        #total_features = tf.shape(inputs)[1]

        beta = tf.random.uniform(shape = [batch_size, 1], minval=-0.98, maxval=0.98) # Shape ()
        gamma = 1.0 / tf.sqrt(1.0 - beta**2) 

        lorentz_components = tf.boolean_mask(inputs, self.lorentz_mask, axis=1)
        lv_reshaped = tf.reshape(lorentz_components, [batch_size, self.n_vectors_per_batch_item, 4])
        pt, eta, phi, E = tf.split(lv_reshaped, num_or_size_splits=4, axis=-1)
  
        pz = pt * tf.sinh(eta) 
        E_prime = gamma * (E - beta * pz)
        pz_prime = gamma * (pz - beta * E) 
        
        epsilon = 1e-8

        eta_prime = tf.asinh(pz_prime / (pt + epsilon)) 
        
        update_values = tf.concat([pt, eta_prime, phi, E_prime], axis=-1)
        boosted_lv_flat = tf.reshape(update_values, [-1])
        batch_indices_flat = tf.repeat(tf.range(batch_size, dtype=tf.int32), self.n_lorentz_features)
        lorentz_indices_flat = tf.tile(self.scatter_indices_template, [batch_size, 1])
        scatter_indices = tf.stack([batch_indices_flat, tf.squeeze(lorentz_indices_flat, axis=-1)], axis=-1)

        data = tf.tensor_scatter_nd_update(
            inputs,
            indices=scatter_indices,
            updates=boosted_lv_flat
        )
        return data

    
    def get_config(self):
        config = super().get_config()
        # Store mask as list for serialization, convert back in from_config if needed
        config.update({
            "phi_mask": self.phi_mask.numpy().tolist(),
            "lorentz_mask": self.lorentz_mask.numpy().tolist(),
            "augmentation": self.augmentation,
        })
        return config


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units, activation="relu", dropout_rate=0.2, use_bias=False, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

    def build(self, input_shape):
        # Main path layers
        self.dense1 = Dense(self.units, use_bias=self.use_bias)
        self.bn1 = BatchNormalization()
        self.activation_layer = Activation(self.activation)
        self.dropout = Dropout(self.dropout_rate)
        self.dense2 = Dense(input_shape[-1], use_bias=self.use_bias)
        self.bn2 = BatchNormalization()

    def call(self, inputs):
        # Main path
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.activation_layer(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.bn2(x)
        
        return x + inputs    

class RegressionModel:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.normalizer = None
        self.model = None
        self.history = None
        self.outFolder = "model_checkpoint"
        self.augmentation_type = kwargs.get("augmentation", "phi")
        self.strategy = kwargs.get("strategy", tf.distribute.MirroredStrategy())
        print(f"Number of devices: {self.strategy.num_replicas_in_sync}")

        """
        Model hyperparameters.
        """
        self.model_type = kwargs.get('model_type', "mlp")
        self.activation_function = kwargs.get('activation_function', "relu")
        self.batch_size = kwargs.get('batch_size', 64)
        self.hidden_layer_size = kwargs.get('hidden_layer_size', 100)
        self.n_layers = kwargs.get('n_layers', 4)
        self.initial_learning_rate = kwargs.get('initial_learning_rate', 0.001)
        self.n_epochs = kwargs.get('n_epochs', 10)
        self.n_normalizer_samples = kwargs.get('n_normalizer_samples', 20)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
    
    def save(self, model_name="PatrikNet.keras"):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.") 
        
        current_dir = os.getcwd()
        models_dir = os.path.join(current_dir, "models")

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        model_save_path = os.path.join(models_dir, model_name)
        self.model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    def load(self, model_name="PatrikNet.keras"):
        current_dir = os.getcwd()
        models_dir = os.path.join(current_dir, "models")
        model_load_path = os.path.join(models_dir,  model_name)

        if os.path.exists(model_load_path):
            self.model = load_model(model_load_path)
            print(f"Model loaded from {model_load_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_load_path}")
        
    def prepare_dataset(self):
        """
        Prepare the dataset for training and validation. 
        """
        print("Batching datasets...")
        self.train_batch = self.dataset.train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.val_batch = self.dataset.val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.dev_batch = self.dataset.dev_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def create_normalizer(self):
        """
        Create and adapt the normalization layer.
        Args:
            train_dataset: Training dataset.
        """
        self.normalizer = Normalization()
        self.normalizer.adapt(self.dataset.train_dataset.map(pick_only_data).take(self.n_normalizer_samples))

    def build_model(self):
        """
        Build a deep neural network model with learning rate decay.
        Args:
            n_train: Number of training samples (used for learning rate decay).
        """
        print("Building model...")
        #with self.strategy.scope():
        input_layer = Input(shape=tuple(self.dataset.train_dataset.element_spec[0].shape.as_list()))
        layer = Augmentation(phi_mask=self.dataset.get_phi_mask(), lorentz_mask=self.dataset.get_lorentz_mask(), augmentation = self.augmentation_type)(input_layer)
        layer = self.normalizer(layer)

        for i in range(self.n_layers):
            layer = ResidualBlock(
                units=self.hidden_layer_size // self.n_layers,
                activation=self.activation_function,
                dropout_rate=self.dropout_rate,
                use_bias=False 
            )(layer)

        output_layer = Dense(1, activation=None)(layer)

        # Define learning rate decay schedule
        learning_rate = CosineDecay(
            initial_learning_rate = self.initial_learning_rate,
            decay_steps = self.n_epochs * self.dataset.train_events // self.batch_size,
            alpha = 5e-4
        )
        # Compile the model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(
            optimizer = AdamW(learning_rate=learning_rate, weight_decay=self.weight_decay, clipnorm=1.0),
            loss=MeanSquaredError(),
            metrics=[MeanSquaredError(), MeanAbsolutePercentageError()]
        )
        #print("Model built successfully within strategy scope.")

    def train_model(self):
        """
        Train the model and log results after each epoch using EpochLogger callback.
        """
        logger.info("Training model...")
        if not self.model:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        checkpointFolder = '{}/checkpoints/checkpoints/'.format(self.outFolder)
        os.makedirs(checkpointFolder, exist_ok=True)

        checkpoint = tf.keras.callbacks.BackupAndRestore(backup_dir=checkpointFolder, delete_checkpoint=False, save_freq=10000)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False, verbose=1, mode='min')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='{}/logs'.format(self.outFolder), histogram_freq=10)
        callbacks = [EpochLogger(logger), early_stop, tensorboard, checkpoint]
        
        history = self.model.fit(
            self.train_batch,
            epochs=self.n_epochs,
            validation_data=self.dev_batch,
            steps_per_epoch=self.dataset.train_events // self.batch_size,
            callbacks=callbacks
        )
        self.history = history

    def evaluate(self):
        """
        Evaluate the model on the validation dataset.
        Automatically prepares the dataset if not already prepared.
        """
        print("Evaluating model's performance...")
        if not self.model:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        # Ensure datasets are prepared
        if not hasattr(self, 'val_batch'):
            self.dataset.build_dataset()
        
        # Evaluate the model
        evaluation_results = self.model.evaluate(self.dev_batch)
        print(f"Validation Loss: {evaluation_results}")

    def plot_history(self):
        """
        Plot training history, visualizing the loss curves for both training and validation datasets.
        The plot is saved to a file named 'loss.png'.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss (Mean Squared Error)', fontsize=14)
        plt.title('Training and Validation Loss Curves', fontsize=16)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig("loss.png")


    def plot_output_distributions(self):
        """
        Plot model output distributions for validation and training datasets.
        Args:
            val_dataset: Validation dataset.
            train_dataset: Training dataset.
        """
        y_val = self.model.predict(self.dataset.val_dataset.map(pick_only_data))
        y_train = self.model.predict(self.dataset.train_dataset.map(pick_only_data))

        plt.figure("Model Output Distribution")
        plt.hist(y_val, bins=100, range=(0, 1), histtype='step', label='Validation Output', density=True)
        plt.hist(y_train, bins=100, range=(0, 1), histtype='step', label='Training Output', density=True)
        plt.xlabel("Model Output")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Output Distribution")
        plt.show()








    
    