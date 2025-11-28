import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import joblib
import json
import sys
import os
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import List

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from model.ddn import DeepDifferentialNetwork
import config

# --- SETTINGS ---
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

# --- STYLING ---
# Use a consistent palette based on coolwarm
# Blue (Cool) for Market/In-Sample, Red (Warm) for Model/Out-Sample
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
color_palette = sns.color_palette("coolwarm", 10)
COLORS = color_palette[0], color_palette[-1]

# --- DATA LOADING ---
def get_aapl_volatility():
    """
    Retrieves the historical realized volatility of AAPL from either a cache file or Yahoo Finance.

    Returns a Pandas DataFrame with a single column 'Realized_Vol' containing the daily realized volatility.

    If the cache file exists, it loads the data from the cache. Otherwise, it downloads the data from Yahoo Finance and caches it to the specified file.

    The realized volatility is calculated as the standard deviation of the log returns over a rolling window of 20 trading days, annualized by multiplying by the square root of 252.

    The index of the returned DataFrame is timezone-naive and represents the trading dates.

    :return: pd.DataFrame
    """
    if os.path.exists(config.STOCK_CACHE_FILE):
        print("Loading AAPL stock data from cache...")
        df = pd.read_csv(config.STOCK_CACHE_FILE, index_col=0, parse_dates=True)
    else:
        print("Downloading AAPL stock data from Yahoo Finance...")
        try:
            df = yf.download("AAPL", start="2015-01-01", end="2024-01-01", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(config.STOCK_CACHE_FILE)
        except Exception:
            # Fallback
            dates = pd.date_range(start="2015-01-01", end="2024-01-01")
            df = pd.DataFrame(index=dates, columns=['Close', 'Log_Ret', 'Realized_Vol'])
            return df

    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Realized_Vol'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
    df.index = df.index.tz_localize(None)
    return df[['Realized_Vol']].dropna()

def load_model_and_scalers():
    """
    Loads the trained Deep Differential Network model, feature scaler, and target scaler from disk.

    If a hyperparameter file is found in the model directory, a custom architecture is initialized based on the hyperparameters.
    Otherwise, a default architecture is used, consisting of 6 layers with 150 neurons each, a dropout rate of 0.0, and 'softplus' activation.

    The model weights are loaded from disk after initialization.

    Returns:
        model (DeepDifferentialNetwork): The initialized Deep Differential Network model.
        sx (MinMaxScaler): The feature scaler.
        sy (MinMaxScaler): The target scaler.
    """
    print("Loading DDN Model...")
    sx = joblib.load(config.SCALER_X_PATH)
    sy = joblib.load(config.SCALER_Y_PATH)
    if config.BEST_HPS_FILE.exists():
        with open(config.BEST_HPS_FILE, 'r') as f: hp = json.load(f)
        model = DeepDifferentialNetwork(num_hidden=hp['num_hidden'], neurons=hp['neurons'], activation=hp['activation'])
    else:
        model = DeepDifferentialNetwork(num_hidden=6, neurons=150, activation='softplus')
    model(tf.zeros((1, 8)))
    model.load_weights(config.WEIGHTS_PATH)
    return model, sx, sy

def predict_prices(model, df, params, sx, sy):
    """
    Predicts the prices of a given set of options using the trained Deep Differential Network model.

    Parameters:
        model (DeepDifferentialNetwork): The trained Deep Differential Network model.
        df (pd.DataFrame): A Pandas DataFrame containing the market data for the options to be priced.
        params (list): A list containing the parameters of the Heston model.
        sx (MinMaxScaler): The feature scaler.
        sy (MinMaxScaler): The target scaler.

    Returns:
        np.ndarray: An array containing the predicted prices of the options in the DataFrame.
    """
    N = len(df)
    p_tiled = np.tile(np.array(params), (N, 1))
    m_vars = df[['r', 'tau', 'log_moneyness']].values
    full_x = np.hstack([p_tiled, m_vars])
    scaled_x = full_x * sx.scale_ + sx.min_
    pred_scaled = model.predict(scaled_x, verbose=0)
    pred_norm = (pred_scaled.flatten() - sy.min_[0]) / sy.scale_[0]
    return pred_norm * df['S0'].values

# --- PLOTTING FUNCTIONS ---
def plot_1_stress_test(results_df: pd.DataFrame):
    """
    Generates a plot to visualize the robustness of the calibrated Heston model to 
    changes in market volatility.

    The plot shows the calibration error (MRE) vs. the market volatility (20-day 
    realized volatility) over time. The plot also highlights notable events in the 
    market history, such as the Volmageddon, COVID Crash, and Fed hikes, to provide 
    context for the model's performance.

    Parameters:
        results_df (pd.DataFrame): A Pandas DataFrame containing the calibration results 
                                   (in-sample and out-of-sample MRE) and the corresponding dates.

    Returns:
        None
    """
    print("Generating Plot 1: Stress Test...")
    vol_df = get_aapl_volatility()
    df = results_df.join(vol_df, how='inner')
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # MRE Lines
    ax1.plot(df.index, df['out_sample_mre'], color=COLORS[1], label='Out-of-Sample MRE', linewidth=1.5, alpha=0.9)
    ax1.plot(df.index, df['in_sample_mre'], color=COLORS[0], label='In-Sample MRE', linewidth=1, linestyle='--', alpha=0.6)
    ax1.set_ylabel('Mean Relative Error (MRE)', color='black', fontsize=12)
    ax1.set_ylim(0, 0.20)
    
    # Volatility Area
    ax2 = ax1.twinx()
    ax2.fill_between(df.index, df['Realized_Vol'], color='gray', alpha=0.15, label='20-Day Realized Vol')
    ax2.set_ylabel('Realized Volatility', color='black', fontsize=12)
    ax2.grid(False) # Turn off grid for secondary axis to avoid clutter
    
    # Styling
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Events
    events = [('2018-02-05', 'Volmageddon'), ('2020-03-20', 'COVID Crash'), ('2022-03-16', 'Fed Hikes')]
    for date, label in events:
        dt = pd.to_datetime(date)
        if df.index[0] <= dt <= df.index[-1]:
            ax1.axvline(dt, color='black', linestyle=':', alpha=0.4)
            ax1.text(dt, 0.18, f' {label}', rotation=90, verticalalignment='top', fontsize=9)

    plt.title('Robustness Analysis: Calibration Error vs. Market Volatility')
    
    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_1_stress_test.png")
    plt.close()

def plot_2_rate_sensitivity(results_df: pd.DataFrame):
    """
    Generates a scatter plot showing the relationship between the implied risk-free rate (r - q) and the out-of-sample mean relative error (MRE).

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing the backtest results. Must contain columns 'implied_rate' and 'out_sample_mre'.

    Notes
    -----
    Saves the plot to a file named 'plot_2_rate_sensitivity.png' in the directory specified by `PLOT_DIR`.
    """
    print("Generating Plot 2: Rate Sensitivity...")
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(data=results_df, x='implied_rate', y='out_sample_mre', 
                    color=COLORS[0], alpha=0.5, edgecolor='w', s=40)
    
    sns.regplot(data=results_df, x='implied_rate', y='out_sample_mre', 
                scatter=False, color=COLORS[1], line_kws={'linewidth': 2})
    
    plt.xlabel("Implied Risk-Free Rate (r - q)")
    plt.ylabel("Out-of-Sample MRE")
    plt.title("Model Stability across Interest Rate Regimes")
    plt.ylim(0, 0.20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_2_rate_sensitivity.png")
    plt.close()

def plot_3_price_fit_panel(results_df: pd.DataFrame, model: DeepDifferentialNetwork, sx: MinMaxScaler, sy: MinMaxScaler):
    """
    Generates a 3x2 panel plot showing the calibration fit for six distinct option maturities on a single day.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing the backtest results. Must contain columns 'implied_rate' and 'kappa', 'lambda', 'sigma', 'rho', 'v0'.
    model : DeepDifferentialNetwork
        The calibrated model used to predict option prices.
    sx : MinMaxScaler
        The feature scaler used to normalize inputs to the model.
    sy : MinMaxScaler
        The target scaler used to normalize outputs from the model.

    Notes
    -----
    Saves the plot to a file named 'plot_3_price_fit.png' in the directory specified by `PLOT_DIR`.
    """
    print("Generating Plot 3: Price Fit Panel (3x2)...")
    
    # Select a date with rich data
    target_date = pd.to_datetime("2020-06-15") 
    # Fallback if exact date missing
    closest_idx = results_df.index.get_indexer([target_date], method='nearest')[0]
    closest_date = results_df.index[closest_idx]
    
    # Get params
    row = results_df.loc[closest_date]
    params = [row['kappa'], row['lambda'], row['sigma'], row['rho'], row['v0']]
    rate = row['implied_rate']
    
    # Load Raw Data
    full_df = pd.concat([pd.read_csv(f, on_bad_lines='skip', low_memory=False) for f in config.AAPL_OPTION_FILES])
    full_df.columns = full_df.columns.str.strip().str.replace(r'\[|\]', '', regex=True).str.strip()
    full_df['QUOTE_DATE'] = pd.to_datetime(full_df['QUOTE_DATE'])
    day_df = full_df[full_df['QUOTE_DATE'] == closest_date].copy()
    
    for c in ['C_BID', 'C_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE']:
        day_df[c] = pd.to_numeric(day_df[c], errors='coerce')
        
    day_df['S0'] = day_df['UNDERLYING_LAST']
    day_df['K'] = day_df['STRIKE']
    day_df['tau'] = day_df['DTE'] / 365.0
    day_df['marketPrice'] = (day_df['C_BID'] + day_df['C_ASK']) / 2.0
    day_df['r'] = rate
    day_df['log_moneyness'] = np.log(day_df['K'] / day_df['S0'])
    
    # Filter
    day_df = day_df[(day_df['marketPrice'] > config.MIN_OPTION_PRICE) & (day_df['tau'] > config.BACKTEST_MIN_TAU)].copy()
    day_df['modelPrice'] = predict_prices(model, day_df, params, sx, sy)
    
    # Select 6 distinct maturities
    unique_taus = sorted(day_df['tau'].unique())
    # Pick 6 evenly spaced indices
    indices = np.linspace(0, len(unique_taus)-1, 6).astype(int)
    selected_taus = [unique_taus[i] for i in indices]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    for i, tau in enumerate(selected_taus):
        ax = axes[i]
        subset = day_df[np.isclose(day_df['tau'], tau, atol=0.005)].sort_values('K')
        
        # Market Dots
        ax.scatter(subset['K'], subset['marketPrice'], color=COLORS[0], s=15, label='Market', alpha=0.7)
        # Model Line
        ax.plot(subset['K'], subset['modelPrice'], color=COLORS[1], linewidth=2, label='DDN Model')
        
        ax.set_title(f"Maturity: {tau*365:.0f} Days")
        if i >= 3: ax.set_xlabel("Strike Price ($)")
        if i % 3 == 0: ax.set_ylabel("Option Price ($)")
        
        if i == 0: ax.legend()
        
    plt.suptitle(f"Model Calibration: AAPL {closest_date.date()} (Spot: ${subset['S0'].iloc[0]:.2f})", fontsize=16)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_3_price_fit.png")
    plt.close()

def plot_4_heatmaps(results_df: pd.DataFrame, model: DeepDifferentialNetwork, sx: MinMaxScaler, sy: MinMaxScaler):
    """
    Generates error heatmaps by iterating over the entire backtesting period.
    
    This function processes every date available in the results dataframe to construct
    a comprehensive error surface. It compares Model Prices vs. Market Prices for both
    Call options (direct) and Put options (via Put-Call Parity).

    Args:
        results_df (pd.DataFrame): DataFrame containing calibrated Heston parameters 
                                   and rates, indexed by date.
        model (tf.keras.Model): The trained DDN model.
        sx (scaler): Input scaler (sklearn).
        sy (scaler): Output scaler (sklearn).

    Returns:
        tuple: (scatter_data_market, scatter_data_model) - Lists of prices for Plot 5.
    """
    print("Generating Plot 4: Dual Heatmaps (Calls & Puts) using ALL days...")
    
    # --- 1. Load and Preprocess All Option Data ---
    # We load all files into memory. For very large datasets (>10GB), 
    # consider processing file-by-file instead of concat-ing all at once.
    print("  - Loading option chains...")
    full_df = pd.concat([pd.read_csv(f, on_bad_lines='skip', low_memory=False) for f in config.AAPL_OPTION_FILES])
    
    # Standardize column names
    full_df.columns = full_df.columns.str.strip().str.replace(r'\[|\]', '', regex=True).str.strip()
    full_df['QUOTE_DATE'] = pd.to_datetime(full_df['QUOTE_DATE'])
    
    # Optimization: Filter option data immediately to only include dates present in our results
    # This significantly speeds up the daily slicing loop below.
    valid_dates = set(results_df.index)
    full_df = full_df[full_df['QUOTE_DATE'].isin(valid_dates)].copy()

    # Pre-convert numeric columns to avoid doing it inside the loop
    cols_to_numeric = ['C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE']
    for c in cols_to_numeric:
        if c in full_df.columns:
            full_df[c] = pd.to_numeric(full_df[c], errors='coerce')

    # --- 2. Iterative Inference ---
    errors_call = []
    errors_put = []
    
    # Collectors for Plot 5 (Global Scatter)
    scatter_data_market = []
    scatter_data_model = []
    
    dates_to_process = results_df.index.sort_values()
    total_days = len(dates_to_process)
    
    print(f"  - Starting inference on {total_days} days...")
    
    for i, date in enumerate(dates_to_process):
        # Progress indicator every 50 days
        if i % 50 == 0:
            print(f"    Processing {i}/{total_days} ({date.date()})...")

        # 2a. Prepare Daily Data
        row = results_df.loc[date]
        params = [row['kappa'], row['lambda'], row['sigma'], row['rho'], row['v0']]
        rate = row['implied_rate']
        
        # Fast slice since we pre-filtered full_df
        day_df = full_df[full_df['QUOTE_DATE'] == date].copy()
        
        if day_df.empty:
            continue
            
        day_df['S0'] = day_df['UNDERLYING_LAST']
        day_df['K'] = day_df['STRIKE']
        day_df['tau'] = day_df['DTE'] / 365.0
        day_df['r'] = rate
        day_df['log_moneyness'] = np.log(day_df['K'] / day_df['S0'])
        
        # 2b. Process CALLS
        calls = day_df.copy()
        calls['marketPrice'] = (calls['C_BID'] + calls['C_ASK']) / 2.0
        # Filter for liquidity and avoiding microstructure noise (prices < $1.0)
        calls = calls[(calls['marketPrice'] > 1.0) & (calls['tau'] > 0.02)]
        
        if not calls.empty:
            # Batch inference
            calls['modelPrice'] = predict_prices(model, calls, params, sx, sy)
            calls['abs_pct_err'] = np.abs(calls['modelPrice'] - calls['marketPrice']) / calls['marketPrice']
            
            # Store lightweight results (only necessary columns)
            errors_call.append(calls[['tau', 'K', 'S0', 'abs_pct_err']].copy())
            
            # Collect for scatter plot
            scatter_data_market.extend(calls['marketPrice'].values)
            scatter_data_model.extend(calls['modelPrice'].values)

        # 2c. Process PUTS (via Parity)
        if 'P_BID' in day_df.columns:
            puts = day_df.copy()
            puts['marketPrice'] = (puts['P_BID'] + puts['P_ASK']) / 2.0
            puts = puts[(puts['marketPrice'] > 1.0) & (puts['tau'] > 0.02)]
            
            if not puts.empty:
                # Predict Call Price first using the same model
                call_pred = predict_prices(model, puts, params, sx, sy)
                
                # Apply Put-Call Parity: P = C - S + K*exp(-rT)
                puts['modelPrice'] = call_pred - puts['S0'] + puts['K'] * np.exp(-rate * puts['tau'])
                puts['modelPrice'] = puts['modelPrice'].clip(lower=0.0) # Arbitrage constraint
                
                puts['abs_pct_err'] = np.abs(puts['modelPrice'] - puts['marketPrice']) / puts['marketPrice']
                errors_put.append(puts[['tau', 'K', 'S0', 'abs_pct_err']].copy())

    # --- 3. Data Processing for Heatmap ---
    def process_bins(df_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Process a list of daily results into a single DataFrame for plotting a heatmap.
        
        Parameters
        ----------
        df_list : List[pd.DataFrame]
            A list of daily results containing columns 'tau', 'K', 'S0', and 'abs_pct_err'.
        
        Returns
        -------
        pd.DataFrame
            A single DataFrame with columns 'Mat_Bin', 'Mon_Bin', and 'abs_pct_err' containing the aggregated mean relative error by bin.
        """
        if not df_list: 
            return pd.DataFrame()
        
        # Concatenate all daily results into one large frame
        df = pd.concat(df_list)
        df['Moneyness'] = df['K'] / df['S0']
        
        # Define bins for the Heatmap
        df['Mat_Bin'] = pd.cut(df['tau'], bins=[0, 0.25, 0.5, 1.0, 2.5], 
                               labels=['< 3M', '3M-6M', '6M-1Yr', '> 1Yr'])
        df['Mon_Bin'] = pd.cut(df['Moneyness'], bins=[0, 0.95, 1.05, 3.0], 
                               labels=['< 0.95', '0.95-1.05', '> 1.05'])
        
        # Aggregate Mean Relative Error by bin
        return df.pivot_table(index='Mat_Bin', columns='Mon_Bin', values='abs_pct_err', aggfunc='mean')

    print("  - Aggregating bins...")
    hm_call = process_bins(errors_call)
    hm_put = process_bins(errors_put)

    # --- 4. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap styling
    kwargs = {'annot': True, 'fmt': ".1%", 'cmap': "coolwarm", 'vmin': 0, 'vmax': 0.15, 'cbar': False}
    
    sns.heatmap(hm_call, ax=axes[0], **kwargs)
    axes[0].set_title(f"Call Option MRE (Aggregated {results_df.index[0].year}-{results_df.index[-1].year})")
    axes[0].set_ylabel("Time to Maturity")
    axes[0].set_xlabel("Moneyness (Strike / Spot)")
    
    sns.heatmap(hm_put, ax=axes[1], **kwargs)
    axes[1].set_title(f"Put Option MRE (Aggregated {results_df.index[0].year}-{results_df.index[-1].year})")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Moneyness (Strike / Spot)")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_4_heatmap.png")
    plt.close()
    
    return scatter_data_market, scatter_data_model

def plot_5_regression(results_df: pd.DataFrame, scatter_mkt: List[float], scatter_mod: List[float]):
    """
    Generates a diagnostics panel (Plot 5) containing two subplots:

    1. Generalization Gap Analysis: In-Sample vs Out-Sample MRE scatter plot with a 45 degree line indicating perfect generalization.
    2. Option-Level Pricing Accuracy: Market Price vs Model Price scatter plot with an identity line indicating ideal pricing accuracy.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing the results of the calibration procedure, including in-sample and out-of-sample MRE columns.
    scatter_mkt : List[float]
        List of market prices for option level scatter plot.
    scatter_mod : List[float]
        List of model prices for option level scatter plot.
    """
    print("Generating Plot 5: Diagnostics Panel...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # --- Subplot 1: In-Sample vs Out-Sample ---
    ax = axes[0]
    max_val = 0.15
    sns.scatterplot(x=results_df['in_sample_mre'], y=results_df['out_sample_mre'], 
                    ax=ax, color=COLORS[0], alpha=0.6, edgecolor='w')
    
    # 45 degree line
    ax.plot([0, max_val], [0, max_val], color=COLORS[1], linestyle='--', label='Perfect Generalization')
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("In-Sample MRE")
    ax.set_ylabel("Out-of-Sample MRE")
    ax.set_title("Generalization Gap Analysis")
    ax.legend()
    
    # --- Subplot 2: Option Level Scatter ---
    ax2 = axes[1]
    # Downsample if too many points
    if len(scatter_mkt) > 2000:
        idx = np.random.choice(len(scatter_mkt), 2000, replace=False)
        s_mkt = np.array(scatter_mkt)[idx]
        s_mod = np.array(scatter_mod)[idx]
    else:
        s_mkt, s_mod = scatter_mkt, scatter_mod
        
    ax2.scatter(s_mkt, s_mod, color=COLORS[0], alpha=0.5, s=10)
    
    # Identity Line
    lims = [0, max(max(s_mkt), max(s_mod))]
    ax2.plot(lims, lims, color=COLORS[1], linestyle='--')
    
    ax2.set_xlabel("Market Price ($)")
    ax2.set_ylabel("Model Price ($)")
    ax2.set_title("Option-Level Pricing Accuracy")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_5_regression.png")
    plt.close()

def plot_6_parameters(results_df: pd.DataFrame):
    """
    Plots the evolution of the five Heston model parameters over the backtesting period.

    Parameters:
    results_df (pd.DataFrame): Dataframe containing the daily calibrated Heston parameters.

    Returns:
    None
    """
    print("Generating Plot 6: Parameter Evolution...")
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    
    params = [
        ('kappa', 'Mean Reversion ($\kappa$)', COLORS[0]),
        ('lambda', 'Long-Run Var ($\lambda$)', COLORS[0]),
        ('sigma', 'Vol of Vol ($\sigma$)', COLORS[1]),
        ('rho', 'Correlation ($\\rho$)', COLORS[1]),
        ('v0', 'Initial Var ($v_0$)', COLORS[0])
    ]
    
    for i, (col, label, c) in enumerate(params):
        ax = axes[i]
        ax.plot(results_df.index, results_df[col], color=COLORS[0], linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        
        # Highlight bounds if railed
        if col == 'rho': ax.set_ylim(-1.0, 0.1)
        if col == 'sigma': ax.set_ylim(0, 1.1)
        
    axes[-1].set_xlabel("Date")
    plt.suptitle("Heston Parameter Stability (2016-2023)", y=0.99) # Move title up
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_6_parameters.png")
    plt.close()

def generate_table_1(results_df: pd.DataFrame):
    """
    Generates a table summarizing the performance of the Heston model across different market regimes.

    Parameters:
    results_df (pd.DataFrame): Dataframe containing the daily calibrated Heston parameters and out-of-sample errors.

    Returns:
    None
    """
    print("Generating Table 1...")
    bins = [
        pd.to_datetime("2015-01-01"),
        pd.to_datetime("2018-02-01"), 
        pd.to_datetime("2020-02-01"), 
        pd.to_datetime("2020-06-01"), 
        pd.to_datetime("2022-01-01"), 
        pd.to_datetime("2024-01-01") 
    ]
    labels = ["Pre-Volmageddon", "Trade War", "COVID Crash", "Recovery", "Inflation"]
    
    results_df['Regime'] = pd.cut(results_df.index, bins=bins, labels=labels)
    
    table = results_df.groupby('Regime', observed=True).agg({
        'out_sample_mre': ['mean', 'std'],
        'kappa': 'std',
        'lambda': 'std',
        'sigma': 'std',
        'rho': 'std',
        'v0': 'std'
    })
    
    # Clean columns
    table.columns = ['Avg MRE', 'Std MRE', 'Std(κ)', 'Std(λ)', 'Std(σ)', 'Std(ρ)', 'Std(v0)']
    print("\n--- TABLE 1: REGIME PERFORMANCE ---")
    print(table)
    table.to_csv(PLOT_DIR / "table_1_regimes.csv")

def main():
    """
    Generates all plots for the thesis.

    Loads the backtest results from the specified CSV file and checks for required columns.
    Then, it generates the plots in the following order:
    1. Stress Test
    2. Rate Sensitivity
    3. Price Fit Panel (Requires Model)
    4. Heatmap of Option Prices
    5. Regression Analysis
    6. Parameter Evolution

    Finally, prints a message indicating the location of the generated plots.
    """
    print(f"Loading results from {config.BACKTEST_OUTPUT_FILE}...")
    if not Path(config.BACKTEST_OUTPUT_FILE).exists():
        raise FileNotFoundError("Run backtest.py first!")
        
    df = pd.read_csv(config.BACKTEST_OUTPUT_FILE)
    df.columns = df.columns.str.strip() # Fix whitespace
    
    required = ['date', 'in_sample_mre', 'out_sample_mre', 'kappa', 'lambda', 'sigma', 'rho', 'v0']
    if not all(c in df.columns for c in required):
        raise ValueError(f"Missing cols. Have: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Statics
    plot_1_stress_test(df)
    plot_2_rate_sensitivity(df)
    plot_6_parameters(df)
    generate_table_1(df)
    
    # Dynamics (Need Model)
    model, sx, sy = load_model_and_scalers()
    plot_3_price_fit_panel(df, model, sx, sy)
    
    # Heatmap returns scatter data for Plot 5
    mkt_scatter, mod_scatter = plot_4_heatmaps(df, model, sx, sy)
    plot_5_regression(df, mkt_scatter, mod_scatter)
    
    print("\nAll plots generated in 'plots/'")

if __name__ == "__main__":
    main()