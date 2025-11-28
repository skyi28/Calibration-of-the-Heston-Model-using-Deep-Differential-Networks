"""
Script: Hyperparameter Tuning (Hyperband)
=========================================

Purpose:
    Performs automated hyperparameter optimization (HPO) for the Deep Differential 
    Network (DDN) using the Hyperband algorithm.

    This script searches for the optimal combination of:
    1. Network Architecture (Number of layers, Neurons per layer).
    2. Regularization (Dropout vs Weight Decay interactions).
    3. Activation Functions (Swish, Softplus, etc.).
    4. Optimization Dynamics (Initial Learning Rate, First Decay Epoch for the Cosine Schedule).

    It uses the 'keras-tuner' library to efficiently traverse the search space 
    defined in 'config.py'.

Outputs:
    - 'models/best_hps.json': A JSON file containing the winning configuration. 
      This file is automatically read by 'train_hom.py' to build the final model.
"""

import numpy as np                                      # For numerical operations
import tensorflow as tf                                 # For Deep Learning                     
import keras_tuner as kt                                # For hyperparameter tuning
import json                                             # For hyperparameter saving            
from sklearn.model_selection import train_test_split    # For train/test split
from sklearn.preprocessing import MinMaxScaler          # For scaling
from pathlib import Path                                # For file paths
import sys                                              # For file paths
from typing import Tuple, Callable                      # For type hints

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from model.ddn import DeepDifferentialNetwork            # For the Deep Differential Network
import config                                            # For configuration

import os
# 1. Disable XLA Compilation (Fixes the "Slow Compile" and "Killed" errors)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import numpy as np
import tensorflow as tf

# 2. Enable GPU Memory Growth (Prevents VRAM hoarding)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def load_and_prep_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads and preprocesses the Heston dataset for hyperparameter tuning.

    Returns a tuple of two tuples, each containing the training/validation data 
    and labels, respectively. The input data is scaled to [-1, 1] and 
    the output data is scaled to [0, 1] to improve training stability. The 
    target gradients are also adjusted according to the chain rule to ensure 
    proper backpropagation of errors.
    """
    # Load dataset
    print("Loading data for tuning...")
    data = np.load(config.HESTON_DATASET_PATH)
    X = data['features']
    Y = data['labels']

    # Input Scaling: Map features to [-1, 1].
    # This centering helps the convergence of tanh/softplus activations.
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    X_s = scaler_x.fit_transform(X)
    
    # Output Scaling: Map prices to [0, 1].
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    Y_s = scaler_y.fit_transform(Y)

    # Differential Learning Adjustment (Chain Rule):
    # Since we trained on scaled inputs/outputs, the target gradients must be
    # scaled to match the slope of the neural network's surface.
    # Formula: Target_Grad = Raw_Grad * (Scale_Y / Scale_X)
    s_x = scaler_x.scale_[:5] # Scale factors for the 5 Heston parameters
    s_y = scaler_y.scale_[0]  # Scale factor for the Price
    
    Y_price = Y[:, 0:1] * s_y + scaler_y.min_[0]
    Y_grads = Y[:, 1:] * (s_y / s_x)
    
    # Combine scaled price and adjusted gradients into one label matrix
    Y_final = np.hstack([Y_price, Y_grads])

    # Split into training and validation set
    x_tr, x_val, y_tr, y_val = train_test_split(
        X_s, Y_final, test_size=config.TUNING_VALIDATION_SIZE, random_state=42
    )
    
    return (x_tr, y_tr), (x_val, y_val)

def get_model_builder(steps_per_epoch: int) -> Callable:
    """
    Returns a function that builds a Deep Differential Network model based on hyperparameters.

    The returned function takes in a HyperParameters object and returns a compiled
    Deep Differential Network model.

    The model is built based on the hyperparameters passed in, which are used to
    determine the architecture, regularization, activation functions, and initial
    learning rate of the model.

    The model is then compiled with the AdamW optimizer and the mean squared error
    loss function. The AdamW optimizer is used with a cosine decay schedule to
    enable high-precision weight decay.

    Parameters
    ----------
    steps_per_epoch : int
        The number of steps per epoch in the training process.

    Returns
    -------
    build_model : Callable
        A function that takes in a HyperParameters object and returns a compiled
        Deep Differential Network model.
    """
    def build_model(hp: kt.HyperParameters) -> DeepDifferentialNetwork:
        """
        Builds a Deep Differential Network model based on hyperparameters.

        Parameters
        ----------
        hp : HyperParameters
            The hyperparameters to use for building the model.

        Returns
        -------
        model : DeepDifferentialNetwork
            The compiled Deep Differential Network model.
        """
        cfg = config.TUNING_CONFIG
        
        # Architecture Search
        # Number of hidden layers
        n_layers = hp.Int('num_hidden', 
                          min_value=cfg['num_hidden_layers']['min'], 
                          max_value=cfg['num_hidden_layers']['max'], 
                          step=cfg['num_hidden_layers']['step'])
        # Neurons per hidden layer
        n_neurons = hp.Int('neurons', 
                           min_value=cfg['neurons_per_layer']['min'], 
                           max_value=cfg['neurons_per_layer']['max'], 
                           step=cfg['neurons_per_layer']['step'])
        # Regularization
        dropout = hp.Float('dropout', 
                           min_value=cfg['dropout_rate']['min'], 
                           max_value=cfg['dropout_rate']['max'], 
                           step=cfg['dropout_rate']['step'])
        # Activation functions
        activation = hp.Choice('activation', values=cfg['activation']['values'])
        # Initial learning rate
        initial_lr = hp.Choice('learning_rate', values=cfg['learning_rate']['values'])
        # First decay epochs
        first_decay_epochs = hp.Int('first_decay_epochs', 
                                   min_value=cfg['first_decay_epochs']['min'], 
                                   max_value=cfg['first_decay_epochs']['max'], 
                                   step=cfg['first_decay_epochs']['step'])
        first_decay_steps = first_decay_epochs * steps_per_epoch

        model = DeepDifferentialNetwork(
            num_hidden=n_layers,
            neurons=n_neurons,
            dropout=dropout,
            activation=activation
        )

        # Cosine Decay Schedule
        # We use the passed 'decay_steps' to ensure the tuning schedule matches
        # the duration of the full training run.
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=first_decay_steps,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-6
        )

        # AdamW for high-precision weight decay
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-6
        )

        model.compile(optimizer=optimizer, loss='mse')
        return model

    return build_model

def main() -> None:
    """
    Main entry point for hyperparameter tuning.

    This function performs the following steps:

    1. Loads and prepares data for hyperparameter tuning.
    2. Configures the Hyperband tuner.
    3. Runs the hyperparameter search.
    4. Retrieves and displays the best hyperparameters found.
    5. Saves the best hyperparameters to a configuration file.
    """
    print("Setting seed")
    config.set_reproducibility() 
    
    # 1. Load and Prepare Data
    (x_tr, y_tr), (x_val, y_val) = load_and_prep_data()

    # 2. Configure the Tuner
    cfg = config.TUNING_CONFIG
    steps_per_epoch = len(x_tr) // config.BATCH_SIZE
    print(f"Steps per Epoch for Tuning: {steps_per_epoch}")
    builder = get_model_builder(steps_per_epoch=steps_per_epoch)
    
    # Hyperband is an efficient bandit-based algorithm that kills 
    # underperforming configurations early to save time.
    tuner = kt.Hyperband(
        builder,
        objective='val_loss',
        max_epochs=cfg['max_epochs'],
        factor=cfg['factor'],
        hyperband_iterations=cfg['iterations'],
        directory=cfg['directory'],
        project_name=cfg['project_name'],
        overwrite=cfg.get('overwrite', False) 
    )

    print(f"\n--- Starting Hyperband Search ---")
    tuner.search_space_summary()

    # 3. Run the Search
    # EarlyStopping prevents wasting resources on trials that diverge
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    tuner.search(
        x_tr, y_tr, 
        validation_data=(x_val, y_val), 
        callbacks=[stop_early],
        verbose=1,
        batch_size=config.BATCH_SIZE
    )

    # 4. Retrieve and Display Results
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n--- Best Hyperparameters Found ---")
    print(f"Layers:             {best_hps.get('num_hidden')}")
    print(f"Neurons:            {best_hps.get('neurons')}")
    print(f"Dropout:            {best_hps.get('dropout')}")
    print(f"Activation:         {best_hps.get('activation')}")
    print(f"Initial LR:         {best_hps.get('learning_rate')}")
    print(f"First Decay Epochs: {best_hps.get('first_decay_epochs')}")

    hp_dict = {
        "num_hidden": best_hps.get('num_hidden'),
        "neurons": best_hps.get('neurons'),
        "dropout": best_hps.get('dropout'),
        "activation": best_hps.get('activation'),
        "learning_rate": best_hps.get('learning_rate'),
        "first_decay_epochs": best_hps.get('first_decay_epochs') # Saved for train_hom.py
    }
    
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BEST_HPS_FILE, 'w') as f:
        json.dump(hp_dict, f, indent=4)
    
    print(f"\nConfiguration saved to: {config.BEST_HPS_FILE}")

if __name__ == "__main__":
    main()