# Higgs Mass Reconstruction with TensorFlow & Keras

This repository contains the code for a deep learning project aimed at reconstructing the invariant mass of the Higgs boson in the $H\rightarrow\tau\tau$ decay channel. The project uses neural networks built with TensorFlow and Keras to improve upon the predictions of the traditional Missing Mass Calculator (MMC) method.

## Key Features

-   **Data Pipeline:** Efficiently loads and processes high-energy physics data from `.root` files using `uproot` and `awkward`.
-   **Mass Flattening:** Implements a crucial resampling technique (`sample_from_datasets`) to create a uniform training mass distribution, mitigating the heavy bias from the $Z\rightarrow\tau\tau$ background.
-   **Custom Data Augmentation:** Includes an on-the-fly data augmentation layer within the model for both simple azimuthal ($\phi$) rotations and more complex **Lorentz boosts**.
-   **Modern NN Architecture:** Utilizes a flexible regression model built with **residual blocks** for stable and deep network training.
-   **Efficient Workflow:** Includes functionality to pre-process and save `tf.data.Dataset` files to disk, avoiding slow data loading in subsequent runs.

## Core Components

The project is built around two main Python classes: one for handling the data pipeline and one for the regression model.

### 1. Data Processing Pipeline (`DatasetClass.py`)

This module is responsible for all data ingestion and preparation.

-   **`Dataset` Class:** The base class reads specified variables from `.root` files, processes 4-vectors (`TLorentzVector`) into their components (pT, eta, phi, mass), and splits the data into training, validation, and development sets.
-   **`DatasetMass` Class:** This subclass inherits from `Dataset` and adds the core physics-specific functionalities:
    -   **Mass Flattening:** It uses `tf.data.Dataset.sample_from_datasets` to resample events from different mass bins, creating a uniform ("flat") mass distribution for training. This is the most critical step for mitigating background bias.
    -   **Data Augmentation:** It contains the logic for applying both $\phi$ rotation and Lorentz boost augmentations to the training data on-the-fly.

### 2. Regression Model (`RegressionModel.py`)

This module defines the neural network architecture, training loop, and evaluation logic.

-   **`Augmentation` Layer:** A custom Keras layer that can be integrated directly into the model graph. It applies either $\phi$ or Lorentz transformations to the input features during training, improving model generalization.
-   **`ResidualBlock` Layer:** A custom layer implementing a residual connection (`x + inputs`). This allows for the stable training of deeper networks by preventing vanishing gradients.
-   **`RegressionModel` Class:** This class orchestrates the entire modeling process.
    -   It initializes the dataset and a `Normalization` layer.
    -   It builds a flexible neural network using a stack of `ResidualBlock` layers.
    -   It compiles the model with modern training components like the `AdamW` optimizer and a `CosineDecay` learning rate schedule.
    -   It handles the training loop, model saving/loading, and evaluation.


## How to Use

1.  **Prepare the Data:** First, instantiate and run the `DatasetMass` class to process your source `.root` files. Use the `.save_data()` method to create and save the processed `tf.data.Dataset` files to disk.

    ```python
    # Example: prepare_data.py
    from DatasetClass import DatasetMass

    # Configure paths and variables
    dataset = DatasetMass(file_paths="/path/to/your/*.root")
    dataset.build_dataset()
    dataset.save_data(file_name="processed_data")
    ```

2.  **Train the Model:** Create a script to initialize the `RegressionModel`. Load the pre-processed data, build the model, and start the training.

    ```python
    # Example: train.py
    from DatasetClass import DatasetMass
    from RegressionModel import RegressionModel

    # Load the processed dataset
    dataset = DatasetMass()
    dataset.load_data(file_name="processed_data")
    # Apply on-the-fly augmentation if desired
    dataset.augment_data(n_slices =number_of_slices_for_augmentation)
    
    # Build and train the model
    model = RegressionModel(dataset, augmentation="lorentz", n_epochs=50)
    model.prepare_dataset()
    model.create_normalizer()
    model.build_model()
    model.train_model()
    model.save("my_higgs_model.keras")
    ```