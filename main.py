from ModelClass import RegressionModel
from DatasetClass import DatasetMass
import logging
import os
import itertools

# === Nastav logger: iba do súboru, nič do stdout ===
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

def main():
    # === Načítaj dáta ===
    erik_data = "/scratch/ucjf-atlas/htautau/SM_Htautau_R22/V02_skim_mva_01/*/*/*/*/*H125*.root"
    patrik_data = "/scratch/ucjf-atlas/htautau/SM_Htautau_R22/V02_skim_mva_01/*/*/*/*/*Ztt*.root"

    model_name = "PatrikNet_phi_flattened_30s_0_200.keras"
    logger.info(f"Loading data from: {patrik_data}")

    dataset = DatasetMass()
    dataset.load_data(file_name="data_mmc")
    logger.info("Data loaded successfully.")
    dataset.augment_data(n_slices = 30)
    logger.info("Data augmented successfully.")

    # === Grid search ===
    param_grid = {
        #'batch_size': [8192],
        'batch_size': [64],
        'learning_rate': [1e-3],
        #'epochs': [1],
        "epochs": [1],
        'n_layers': [7],
        'hidden_layer_size': [8192],
        'dropout_rate': [0.3],
        'weight_decay': [1e-4],
        "n_normalizer_samples": [1000],
        "activation_function": ["relu"]
    }

    iterable = list(itertools.product(*param_grid.values()))

    for i, params in enumerate(iterable):
        batch_size_val, learning_rate_val, epochs_val, n_layers_val, hidden_size_val, dropout_val, weight_decay_val, n_normalizer_samples_val, activation_function = params

        logger.info(f"\n{'='*80}")
        logger.info(
            f"[{i+1}/{len(iterable)}] Trénujeme s hyperparametrami:\n"
            f"  batch_size={batch_size_val}, learning_rate={learning_rate_val}, epochs={epochs_val},\n"
            f"  n_layers={n_layers_val}, hidden_layer_size={hidden_size_val},\n"
            f"  dropout_rate={dropout_val}, weight_decay={weight_decay_val}, n_normalizer_samples={n_normalizer_samples_val}, activation_function={activation_function}, lorentz\n"
        )

        model = RegressionModel(
            dataset=dataset,
            batch_size=batch_size_val,
            initial_learning_rate=learning_rate_val,
            n_epochs=epochs_val,
            n_layers=n_layers_val,
            hidden_layer_size=hidden_size_val,
            dropout_rate=dropout_val,
            weight_decay=weight_decay_val,
            n_normalizer_samples = n_normalizer_samples_val,
            augmentation_type = "lorentz"
        )

        model.prepare_dataset()
        model.create_normalizer()
        model.build_model()
        model.train_model()
        model.save(model_name = model_name)
        model.load(model_name = model_name)
        model.plot_history()

        #final_loss = model.history.history['val_loss'][-1]
        #logger.info(f"Final training loss: {final_loss:.6f}")
        
    # Finálne echo do job.out
    logger.info(f"Model uložený ako {model_name}")


if __name__ == "__main__":
    main()
