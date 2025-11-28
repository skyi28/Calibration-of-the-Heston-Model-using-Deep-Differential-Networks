"""
Heston Model Calibration Backtesting Module
===========================================

Purpose:
    This script performs a historical backtest of the Deep Differential Network (DDN)
    calibration methodology against the AAPL option chain dataset (2016-2023).

Methodology:
    1.  Data Loading: Aggregates option data from CSV files.
    2.  Daily Processing: Iterates through trading days using a defined step size.
    3.  Implied Rate Calculation**: For each day, calculates the market-implied 
        risk-free rate (net of dividend yield) using Put-Call Parity on liquid 
        At-The-Money (ATM) options. This ensures the model adapts to regime changes.
    4.  Data Filtering: Selects a subset of "High Quality" options (liquid, 
        non-penny, specific moneyness range) to ensure stable calibration.
    5.  Train/Test Split: Splits the daily option chain into:
        -   Calibration Set (80%): Used to find optimal Heston parameters.
        -   Test Set (20%): Used to evaluate out-of-sample pricing accuracy.
    6.  Calibration: Uses the DDN as a surrogate pricing engine within a 
        Scipy L-BFGS-B optimization loop. The DDN provides exact gradients 
        w.r.t parameters, allowing for rapid convergence.
    7.  Metric Calculation: Computes Mean Relative Error (MRE) to align with 
        academic benchmarks.

Usage:
    Run as a standalone script. Ensure 'config.py' and trained model weights are available.
    Output is saved to 'data/backtest_results_final.csv'.
"""

import pandas as pd                                     # For data manipulation            
import numpy as np                                      # For numerical operations
import tensorflow as tf                                 # For the Deep Differential Network
from pathlib import Path                                # For file paths
import sys                                              # For file paths
import joblib                                           # For loading model assets                     
import json                                             # For loading hyperparameters
from scipy.optimize import minimize                     # For L-BFGS-B optimization
from sklearn.model_selection import train_test_split    # For train/test split
from tqdm import tqdm                                   # For progress bars
from typing import Tuple, List                          # For type hints
from sklearn.preprocessing import MinMaxScaler          # For type hints

# Add project root to path to import the Deep Differential Network
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from model.ddn import DeepDifferentialNetwork            # For the Deep Differential Network
import config                                            # For configuration

def load_model_assets() -> Tuple[DeepDifferentialNetwork, MinMaxScaler, MinMaxScaler]:
    """
    Loads model assets from disk and initializes the Deep Differential Network architecture.

    The function returns the initialized Deep Differential Network model, the feature scaler, and the target scaler.

    If a hyperparameter file is found in the model directory, a custom architecture is initialized based on the hyperparameters.
    Otherwise, a default architecture is used, consisting of 6 layers with 150 neurons each, a dropout rate of 0.0, and 'softplus' activation.

    The model weights are loaded from disk after initialization.

    Returns:
        model (DeepDifferentialNetwork): The initialized Deep Differential Network model.
        sx (MinMaxScaler): The feature scaler.
        sy (MinMaxScaler): The target scaler.
    """    
    print(f"Loading model assets from {config.MODEL_DIR}...")
    sx = joblib.load(config.SCALER_X_PATH)
    sy = joblib.load(config.SCALER_Y_PATH)
    
    # Check for hyperparameter file, if it exists, initialize the tuned architecture, else use the default
    if config.BEST_HPS_FILE.exists():
        print(f"Found tuned hyperparameters in {config.BEST_HPS_FILE}. initializing custom architecture.")
        with open(config.BEST_HPS_FILE, 'r') as f:
            hp = json.load(f)
        model = DeepDifferentialNetwork(
            num_hidden=hp['num_hidden'],
            neurons=hp['neurons'],
            dropout=hp['dropout'],
            activation=hp['activation']
        )
    else:
        print("No hyperparameter file found. Initializing default paper-based architecture (6 layers, 150 neurons).")
        model = DeepDifferentialNetwork(num_hidden=6, neurons=150, dropout=0.0, activation='softplus')

    # Run a dummy forward pass to initialize the graph before loading weights
    model(tf.zeros((1, 8))) 
    model.load_weights(config.WEIGHTS_PATH)
    print("Model weights loaded successfully.")
    return model, sx, sy

def get_implied_rate_and_data(daily_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    This function takes a daily DataFrame of option prices and transforms it into a DataFrame
    that can be used for calibrating a Deep Differential Network model.

    The function performs the following operations:
    1. Coerces option prices from strings to numeric values.
    2. Calculates the underlying price, strike, and time to maturity from the option prices.
    3. Calculates the implied rate (r - q) using Put-Call Parity on ATM options.
    4. Transforms the inputs to log-moneyness to match the training distribution.
    5. Applies strict filters to ensure high-quality data for calibration.

    Parameters:
        daily_df (pd.DataFrame): A DataFrame of daily option prices.

    Returns:
        df_calib (pd.DataFrame): A DataFrame of option prices suitable for calibrating a Deep Differential Network model.
        daily_rate (float): The implied rate (r - q) calculated from ATM options.
    """
    # Convert columns to numeric
    cols = ['C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE']
    for c in cols: 
        if c in daily_df.columns:
            daily_df[c] = pd.to_numeric(daily_df[c], errors='coerce')
    
    # Translate to the model inputs
    daily_df['S0'] = daily_df['UNDERLYING_LAST']
    daily_df['K'] = daily_df['STRIKE']
    daily_df['tau'] = daily_df['DTE'] / 365.0
    daily_df['Call_Price'] = (daily_df['C_BID'] + daily_df['C_ASK']) / 2.0
    
    # We calculate the implied rate (r - q) using Put-Call Parity on ATM options.
    # This is critical because using a fixed rate ignores dividend yields and regime shifts.
    if 'P_BID' in daily_df.columns:
        daily_df['Put_Price'] = (daily_df['P_BID'] + daily_df['P_ASK']) / 2.0
        daily_df['abs_mny'] = np.abs(daily_df['S0'] - daily_df['K'])
        # We only use options with a maturity of at least a week (0.02 years)
        atm = daily_df[(daily_df['tau'] >= config.BACKTEST_MIN_TAU) & (daily_df['tau'] <= config.BACKTEST_MAX_TAU)].sort_values('abs_mny')
        
        if len(atm) > 0:
            # Parity formula solved for r: r = -1/T * ln((S - C + P) / K)
            atm['implied_r'] = -1 / atm['tau'] * np.log(
                (atm['S0'] - atm['Call_Price'] + atm['Put_Price']) / atm['K']
            )
            # Filter outliers and take median to get a robust daily rate
            valid = atm[np.abs(atm['implied_r']) <= config.BACKTEST_MAX_IMPLIED_RATE]['implied_r']
            daily_rate = valid.median() if len(valid) > 0 else 0.0
        # Use fallback rate if there are no valid options
        else: 
            daily_rate = config.FALLBACK_RISK_FREE_RATE
    else: 
        daily_rate = config.FALLBACK_RISK_FREE_RATE

    daily_df['r'] = daily_rate
    
    # The network was trained on log-moneyness, not linear moneyness.
    # We must transform the inputs here to match the training distribution.
    daily_df['log_moneyness'] = np.log(daily_df['K'] / daily_df['S0'])

    daily_df['marketPrice'] = daily_df['Call_Price']
    daily_df['norm_price'] = daily_df['marketPrice'] / daily_df['S0']
    
    # Apply strict filters to ensure we calibrate on high-quality data.
    # We avoid penny options (< 0.50) and extreme moneyness to ensure stability.
    df_calib = daily_df[
        (daily_df['log_moneyness'] >= config.MIN_LOG_MONEYNESS) &  # translates to ~0.78 linear moneyness
        (daily_df['log_moneyness'] <= config.MAX_LOG_MONEYNESS) &  # translates to ~1.28 linear moneyness
        (daily_df['marketPrice'] > config.MIN_OPTION_PRICE) &      # avoid penny options, ensure a market price of at least $0.50
        (daily_df['tau'] >= config.BACKTEST_MIN_TAU) &             # avoid options with very short maturities
        (daily_df['tau'] <= config.BACKTEST_MAX_TAU)               # avoid options the neural network has not learned
    ].copy()
    
    return df_calib, daily_rate

# reduce_retracing=True prevents TensorFlow from recompiling the graph for every batch,
# significantly speeding up the loop.
@tf.function(reduce_retracing=True)
def predict_norm_prices(params: tf.Tensor, inputs: tf.Tensor, model: DeepDifferentialNetwork,
                        sx_s: tf.Tensor, sx_m: tf.Tensor, sy_s: tf.Tensor, sy_m: tf.Tensor) -> tf.Tensor:
    """
    Predicts the normalized price of a European Call Option given the model parameters,
    option characteristics, and feature/target scalers.

    Parameters:
        params (tf.Tensor): A tensor containing the Heston model parameters.
        inputs (tf.Tensor): A tensor containing the option characteristics (r, tau, log_moneyness).
        model (DeepDifferentialNetwork): The DeepDifferentialNetwork model.
        sx_s (tf.Tensor): The feature scaler's scale.
        sx_m (tf.Tensor): The feature scaler's mean.
        sy_s (tf.Tensor): The target scaler's scale.
        sy_m (tf.Tensor): The target scaler's mean.

    Returns:
        tf.Tensor: A tensor containing the normalized prices of the options in the batch.
    """
    N = tf.shape(inputs)[0]
    # Tile the parameters (kappa, sigma, etc.) to match the number of options in the batch
    p_tiled = tf.tile(tf.expand_dims(params, 0), [N, 1])
    # Concatenate the parameters with the option characteristics
    full_x = tf.concat([p_tiled, inputs], axis=1)
    
    # Scale inputs to match training distribution
    scaled_x = full_x * sx_s + sx_m
    pred_scaled = model(scaled_x)
    # Inverse transform to get back to normalized price space
    pred_norm = (tf.squeeze(pred_scaled) - sy_m) / sy_s
    return pred_norm

@tf.function(reduce_retracing=True)
def get_weighted_loss_and_grads(params: tf.Tensor, inputs: tf.Tensor, targets: tf.Tensor, model: DeepDifferentialNetwork,
                        sx_s: tf.Tensor, sx_m: tf.Tensor, sy_s: tf.Tensor, sy_m: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes the weighted loss and its gradient with respect to the Heston model parameters.

    Parameters:
        params (tf.Tensor): A tensor containing the Heston model parameters.
        inputs (tf.Tensor): A tensor containing the option characteristics (r, tau, log_moneyness).
        targets (tf.Tensor): A tensor containing the normalized prices of the options in the batch.
        model (DeepDifferentialNetwork): The DeepDifferentialNetwork model.
        sx_s (tf.Tensor): The feature scaler's scale.
        sx_m (tf.Tensor): The feature scaler's mean.
        sy_s (tf.Tensor): The target scaler's scale.
        sy_m (tf.Tensor): The target scaler's mean.

    Returns:
        loss (tf.Tensor): The weighted loss of the batch.
        grads (tf.Tensor): The gradients of the loss with respect to the Heston model parameters.
    """
    with tf.GradientTape() as tape:
        # Ensures that tensor is being traced by this tape.
        tape.watch(params)
        # Predict normalized prices
        preds = predict_norm_prices(params, inputs, model, sx_s, sx_m, sy_s, sy_m)
        # We minimize Squared Relative Error ( (Pred-True)/True )^2
        # This aligns the optimizer with the MRE metric reported in literature.
        rel_error = (preds - targets) / (targets + 1e-6)
        loss = tf.reduce_mean(tf.square(rel_error))
    
    # Compute gradients of the loss w.r.t Heston parameters
    grads = tape.gradient(loss, params)
    return loss, grads

def calibrate(train_df: pd.DataFrame, model: DeepDifferentialNetwork, sx: MinMaxScaler, sy: MinMaxScaler) -> List[float]:
    """
    Calibrates the Heston model parameters using the provided training data and model.

    This function uses Scipy's minimize function with the L-BFGS-B algorithm to minimize the
    weighted loss of the model on the training data. The function value and gradient are
    computed using TensorFlow's automatic differentiation.

    The function is wrapped in a loop to perform multi-start optimization, which helps to avoid
    getting stuck in local minima. The best result from three random initializations is kept.

    Parameters:
        train_df (pd.DataFrame): The training data.
        model (DeepDifferentialNetwork): The model to be calibrated.
        sx (MinMaxScaler): The feature scaler.
        sy (MinMaxScaler): The target scaler.

    Returns:
        best_params (list): The calibrated Heston model parameters.
    """
    # Convert dataframe columns to TensorFlow tensors
    inputs = tf.constant(train_df[['r', 'tau', 'log_moneyness']].values, dtype=tf.float32)
    targets = tf.constant(train_df['norm_price'].values, dtype=tf.float32)
    
    # Pre-cast scaler values to tensors to avoid overhead in the loop
    sx_s = tf.constant(sx.scale_, dtype=tf.float32)
    sx_m = tf.constant(sx.min_, dtype=tf.float32)
    sy_s = tf.constant(sy.scale_[0], dtype=tf.float32)
    sy_m = tf.constant(sy.min_[0], dtype=tf.float32)
    
    # Wrapper function required by Scipy
    def func(p_np: np.array) -> Tuple[float, np.array]:
        """
        Wrapper function for Scipy's minimize function.

        Parameters:
            p_np (numpy array): The Heston model parameters to be optimized.

        Returns:
            loss (float): The weighted loss of the model on the training data.
            grads (numpy array): The gradients of the loss w.r.t the Heston model parameters.
        """
        p_tf = tf.constant(p_np, dtype=tf.float32)
        loss, grads = get_weighted_loss_and_grads(p_tf, inputs, targets, model, sx_s, sx_m, sy_s, sy_m)
        return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
    
    # Initialize best loss and parameters
    best_loss = np.inf
    best_params = None
    
    # Perform Multi-Start Optimization to avoid getting stuck in local minima.
    # We try different random initializations and keep the best result.
    for _ in range(config.OPTIMIZATION_STARTING_POINTS):
        x0 = [
            np.random.uniform(1.0, 4.0), np.random.uniform(0.0, 0.5),
            np.random.uniform(0.2, 0.6), np.random.uniform(-0.8, -0.3),
            np.random.uniform(0.05, 0.3)
        ]
        try:
            # L-BFGS-B is ideal here because we have a smooth function with exact gradients
            res = minimize(func, x0, method='L-BFGS-B', jac=True, bounds=config.BACKTEST_PARAMETER_BOUNDS, tol=1e-6)
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except:
            # In case of failure, skip to the next iteration
            continue
            
    return best_params

def calculate_mre(params: Tuple[float, float, float, float, float], df: pd.DataFrame,
                  model: DeepDifferentialNetwork, sx: MinMaxScaler, sy: MinMaxScaler) -> float:
    """
    Computes the Mean Relative Error (MRE) metric for a given model and dataset.

    Parameters:
        params (list): The Heston model parameters.
        df (pandas DataFrame): The dataset containing the option characteristics and true prices.
        model (DeepDifferentialNetwork): The DeepDifferentialNetwork model.
        sx (MinMaxScaler): The feature scaler.
        sy (MinMaxScaler): The target scaler.

    Returns:
        float: The Mean Relative Error (MRE) metric for the model on the dataset.
    """
    inputs = tf.constant(df[['r', 'tau', 'log_moneyness']].values, dtype=tf.float32)
    
    sx_s = tf.constant(sx.scale_, dtype=tf.float32)
    sx_m = tf.constant(sx.min_, dtype=tf.float32)
    sy_s = tf.constant(sy.scale_[0], dtype=tf.float32)
    sy_m = tf.constant(sy.min_[0], dtype=tf.float32)
    
    pred_norm = predict_norm_prices(tf.constant(params, dtype=tf.float32), inputs, model, sx_s, sx_m, sy_s, sy_m)
    
    S0 = df['S0'].values
    market_prices = df['marketPrice'].values
    model_prices = pred_norm.numpy() * S0
    
    # Filter to ensure metric stability (avoid large % errors on tiny prices) and division by zero
    mask = market_prices > config.MIN_OPTION_PRICE
    if np.sum(mask) > 0:
        # Calculate MRE
        mre = np.mean(np.abs(model_prices[mask] - market_prices[mask]) / market_prices[mask])
    else:
        mre = 0.0
        
    return mre

def main() -> None:
    """
    Main entry point for backtesting the DeepDifferentialNetwork model.

    1. Loads and concatenates CSV data files.
    2. Splits the data into calibration (train) and generalization (test) sets.
    3. Optimizes the Heston model parameters using the calibration set.
    4. Evaluates the optimized model on both the calibration and generalization sets.
    5. Saves the backtest results to a CSV file.

    Parameters:
        None

    Returns:
        None
    """
    print("Setting seed")
    config.set_reproducibility() 
    
    model, sx, sy = load_model_assets()
    
    print("Loading and concatenating CSV data files...")
    full_df = pd.concat([pd.read_csv(f, on_bad_lines='skip', low_memory=False) for f in config.AAPL_OPTION_FILES])
    full_df.columns = full_df.columns.str.strip().str.replace(r'\[|\]', '', regex=True).str.strip()
    full_df['QUOTE_DATE'] = pd.to_datetime(full_df['QUOTE_DATE'])
    unique_dates = sorted(full_df['QUOTE_DATE'].unique())
    
    print(f"Data loaded. Found {len(unique_dates)} unique trading days.")
    print(f"Starting backtest with step size {config.STEP_SIZE} (processing every {config.STEP_SIZE}th day).")
    
    results = []
    
    for date in tqdm(unique_dates[::config.STEP_SIZE]):
        daily_df = full_df[full_df['QUOTE_DATE'] == date].copy()
        
        # Skip days with insufficient data
        if len(daily_df) < config.MIN_OPTION_CONTRACTS:
            print(f'To less options: {len(daily_df)} < {config.MIN_OPTION_CONTRACTS}')
        
        try:
            # Pre-processing: Implied Rates and Filtering
            df_liquid, rate = get_implied_rate_and_data(daily_df)
            if len(df_liquid) < config.MIN_LIQUID_OPTION_CONTRACTS:
                print(f'To less liquid options: {len(df_liquid)} < {config.MIN_LIQUID_OPTION_CONTRACTS}')
                continue
                
            # Split Data: Calibration (Train) vs Generalization (Test)
            train_df, test_df = train_test_split(df_liquid, test_size=config.TEST_SET_SIZE, random_state=42)
            
            # Optimization Step
            params = calibrate(train_df, model, sx, sy)
            if params is None:
                continue 
            
            # Evaluation Step
            train_mre = calculate_mre(params, train_df, model, sx, sy)
            test_mre = calculate_mre(params, test_df, model, sx, sy)
            
            results.append({
                'date': date,
                'in_sample_mre': train_mre,
                'out_sample_mre': test_mre,
                'implied_rate': rate,
                'kappa': params[0], 
                'lambda': params[1], 
                'sigma': params[2], 
                'rho': params[3], 
                'v0': params[4],
                'n_train': len(train_df)
            })
        except Exception:
            continue
    
    # Create results dataframe and save it
    res_df = pd.DataFrame(results)
    res_df.to_csv(config.BACKTEST_OUTPUT_FILE, index=False)
    print(f"Backtest complete. Results saved to {config.BACKTEST_OUTPUT_FILE}")
    
    if not res_df.empty:
        print("\n--- FINAL PERFORMANCE SUMMARY ---")
        print(f"Total Days Processed:     {len(res_df)}")
        print(f"Avg In-Sample MRE:        {res_df['in_sample_mre'].mean():.4f} (Paper/Target: < 0.06)")
        print(f"Avg Out-of-Sample MRE:    {res_df['out_sample_mre'].mean():.4f}")
        print("---------------------------------")

if __name__ == "__main__":
    main()