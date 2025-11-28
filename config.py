from pathlib import Path
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42                                                   # Seed for reproduceability
def set_reproducibility(seed: int = SEED):
    # 1. Python and Numpy
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. TensorFlow
    tf.random.set_seed(seed)
    
    # 3. Force TensorFlow to use deterministic GPU algorithms
    # tf.config.experimental.enable_op_determinism()
    
    print(f"Randomness locked with seed: {seed}")                                                

# --- File Paths ---
DATA_DIR = Path("data")                                     # Directory in which data is stored
MODEL_DIR = Path("models")                                  # Directory in which models are stored
AAPL_OPTION_FILES = [                                       # Paths to the AAPL option data files
    "data/kaggle/aapl_2016_2020.csv", 
    "data/kaggle/aapl_2021_2023.csv",
]
DESC_ANALYSIS_OUTPUT_DIR = DATA_DIR / "descriptive_analysis"  # Directory for descriptive analysis
HESTON_DATASET_PATH = DATA_DIR / "heston_dataset.npz"       # Path for the generated dataset
BEST_HPS_FILE = MODEL_DIR / "best_hps.json"                 # Paths for the best hyperparameters
SCALER_X_PATH = MODEL_DIR / "scaler_x.save"                 # Path for the feature scaler
SCALER_Y_PATH = MODEL_DIR / "scaler_y.save"                 # Path for the target scaler
WEIGHTS_PATH = MODEL_DIR / "ddn.weights.h5"                 # Path for the model weights
STOCK_CACHE_FILE = DATA_DIR / "AAPL_stock_history.csv"      # Path for the cached stock data

# --- Data Set Generation Configuration ---
NUM_SAMPLES = 200_000                                     # Number of samples to generate
MIN_TAU = 5 / 365                                           # Shortest maturity of options in years
MAX_TAU = 2.5                                               # Longest maturity of options in years
MAX_RISK_FREE_RATE = 0.1                                    # Maximum risk-free rate
HESTON_PARAMS = ['kappa', 'lambda', 'sigma', 'rho', 'v0']   # Names of the Heston parameters
PARAM_RANGES = {
    'kappa': [0.01, 5.0],                                   # Mean reversion speed of the variance process
    'lambda': [0.0, 1.0],                                   # Long-run average variance (often denoted as theta)
    'sigma': [0.1, 1.0],                                    # Volatility of volatility (Paper lower bound is 0.1)
    'rho': [-0.99, 0.0],                                    # Correlation between asset price and variance
    'v0': [0.01, 1.0],                                      # Initial variance at time t=0
    'r': [-0.03, MAX_RISK_FREE_RATE],                       # Risk-free interest rate, allow negative implied interest rates since they appear in the dataset
    'tau': [MIN_TAU, MAX_TAU],                              # Time to maturity (in years)
    'log_moneyness': [-1.0, 1.0]                            # Log-moneyness range (ln(K/S0))
}

# --- Descriptive Analysis Configuration ---
DESC_ANALYSIS_MAX_LENGTH_BEFORE_SAMPLING = 2_000_000        # Maximum number of rows before sub-sampling to save time for plotting
DESC_ANALYSIS_N_SAMPLES = 50_000                            # Number of samples which are used for descriptive analysis if data length exceeds DESC_ANALYSIS_MAX_LENGTH_BEFORE_SAMPLING

# --- Model Training Configuration ---
TRAIN_VALIDATION_SET_SIZE = 0.2                             # Size of the validation set (fraction of total data)
BATCH_SIZE = 256                                            # Batch size for training
EPOCHS = 400                                                # Number of epochs for training
EARLY_STOPPING_PATIENCE = 50                                # Early stopping patience

# --- Backtest Configuration ---
TEST_SET_SIZE = 0.2                                         # Size of the test set (fraction of total data)
STEP_SIZE = 1                                               # Step size for backtesting
BACKTEST_OUTPUT_FILE = DATA_DIR / "backtest_results.csv"    # Output file for backtest results
MIN_OPTION_CONTRACTS = 50                                   # Minimum number of option contracts for a specific day to include in the backtest
MIN_LIQUID_OPTION_CONTRACTS = 20                            # Minimum number of liquid option contracts (afer filtering) for a specific day to include in the backtest
MIN_OPTION_PRICE = 0.50                                     # Minimum option price to include an option in the backtest
BACKTEST_MIN_TAU = MIN_TAU                                  # Shortest allowed maturity to be included in the backtest
BACKTEST_MAX_TAU = MAX_TAU                                  # Longest allowed maturity to be included in the backtest
BACKTEST_MAX_IMPLIED_RATE = MAX_RISK_FREE_RATE              # Maximum implied risk-free to be included in the risk-free rate calculation 
FALLBACK_RISK_FREE_RATE = 0.02                              # Fallback risk-free rate for backtesting
MIN_LOG_MONEYNESS = -0.25                                   # Minimum log moneyness for an option to be included in the backtest. Translates to ~0.78 linear moneyness
MAX_LOG_MONEYNESS = 0.25                                    # Maximum log moneyness for an option to be included in the backtest. Translates to ~1.28 linear moneyness
OPTIMIZATION_STARTING_POINTS = 3                            # Number of starting points for parameter optimization
BACKTEST_PARAMETER_BOUNDS = [                               # Constraints for Heston parameters [Kappa, Lambda, Sigma, Rho, v0]
    (0.01, 5.0),                                            # These match the bounds used in the reference paper.
    (0.0, 1.0),
    (0.1, 1.0),
    (-0.99, 0.0),
    (0.01, 1.0)
]

# --- Hyperparameter Tuning Configuration ---
TUNING_VALIDATION_SIZE = 0.2                                # Size of the validation set (fraction of total data)
TUNING_CONFIG = {
    'num_hidden_layers': {                                  # Search range for number of hidden layers
        'min': 4,
        'max': 8,
        'step': 1
    },
    'neurons_per_layer': {                                  # Search range for number of neurons per hidden layer
        'min': 32,
        'max': 256,
        'step': 32
    },
    'dropout_rate': {                                       # Search range for dropout rate
        'min': 0.0,
        'max': 0.5,
        'step': 0.05
    },
    'learning_rate': {                                      # Search range for learning rate
        'values': [1e-2, 1e-3, 5e-4, 1e-4]
    },
    'activation': {                                         # Search range for activation
        'values': ['swish', 'tanh', 'softplus']
    },
    'first_decay_epochs': {
        'min': 25,
        'max': 200,
        'step': 25
    },
    'max_epochs': 400,                                      # Max epochs per trial
    'factor': 4,                                            # Reduction factor for Hyperband
    'iterations': 1,                                        # How many times to run the full Hyperband process
    'directory': 'tuning_results',                          # Directory in which to save results
    'project_name': 'heston_ddn_v1',                        # Project name
    'overwrite': False                                      # Overwrite existing results, set to True to start a new run
}
