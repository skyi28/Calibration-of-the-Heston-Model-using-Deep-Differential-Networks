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
    print("Loading DDN Model...")
    sx = joblib.load(config.SCALER_X_PATH)
    sy = joblib.load(config.SCALER_Y_PATH)
    if config.BEST_HPS_FILE.exists():
        with open(config.BEST_HPS_FILE, 'r') as f: hp = json.load(f)
        model = DeepDifferentialNetwork(num_hidden=hp['num_hidden'], neurons=hp['neurons'], activation=hp['activation'])
    else:
        model = DeepDifferentialNetwork(num_hidden=6, neurons=150, activation='softplus')
    model(tf.zeros((1, 9)))
    model.load_weights(config.WEIGHTS_PATH)
    return model, sx, sy

def predict_prices(model, df, params, sx, sy):
    """Returns Model Call Prices"""
    N = len(df)
    p_tiled = np.tile(np.array(params), (N, 1))
    m_vars = df[['r', 'q', 'tau', 'log_moneyness']].values
    full_x = np.hstack([p_tiled, m_vars])
    scaled_x = full_x * sx.scale_ + sx.min_
    pred_scaled = model.predict(scaled_x, verbose=0)
    pred_norm = (pred_scaled.flatten() - sy.min_[0]) / sy.scale_[0]
    return pred_norm * df['S0'].values

# ==============================================================================
# PLOTS
# ==============================================================================

def plot_1_stress_test(results_df):
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

def plot_2_rate_sensitivity(results_df):
    print("Generating Plot 2: Rate Sensitivity...")
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(data=results_df, x='avg_risk_free_rate', y='out_sample_mre', 
                    color=COLORS[0], alpha=0.5, edgecolor='w', s=40)
    
    sns.regplot(data=results_df, x='avg_risk_free_rate', y='out_sample_mre', 
                scatter=False, color=COLORS[1], line_kws={'linewidth': 2})
    
    plt.xlabel("Implied Risk-Free Rate (r - q)")
    plt.ylabel("Out-of-Sample MRE")
    plt.title("Model Stability across Interest Rate Regimes")
    plt.ylim(0, 0.20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_2_rate_sensitivity.png")
    plt.close()

def plot_3_price_fit_panel(results_df, model, sx, sy):
    print("Generating Plot 3: Price Fit Panel (3x2)...")
    
    # Select a date with rich data
    target_date = pd.to_datetime("2020-06-15") 
    # Fallback if exact date missing
    closest_idx = results_df.index.get_indexer([target_date], method='nearest')[0]
    closest_date = results_df.index[closest_idx]
    
    # Get params
    row = results_df.loc[closest_date]
    params = [row['kappa'], row['lambda'], row['sigma'], row['rho'], row['v0']]
    dividend_yield = row['implied_dividend_yield']
    
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
    day_df['r'] = row['avg_risk_free_rate']
    day_df['q'] = dividend_yield
    day_df['log_moneyness'] = np.log(day_df['K'] / day_df['S0'])
    
    # Filter
    day_df = day_df[(day_df['marketPrice'] > 0.1) & (day_df['tau'] > 0.02)].copy()
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

def plot_4_heatmaps(results_df, model, sx, sy):
    print("Generating Plot 4: Dual Heatmaps (Calls & Puts)...")
    
    # Sample random days to build a representative error surface
    sample_dates = results_df.sample(15, random_state=42).index
    full_df = pd.concat([pd.read_csv(f, on_bad_lines='skip', low_memory=False) for f in config.AAPL_OPTION_FILES])
    full_df.columns = full_df.columns.str.strip().str.replace(r'\[|\]', '', regex=True).str.strip()
    full_df['QUOTE_DATE'] = pd.to_datetime(full_df['QUOTE_DATE'])
    
    errors_call = []
    errors_put = []
    
    # To generate Plot 5 later, we collect raw data here
    scatter_data_market = []
    scatter_data_model = []
    
    for date in sample_dates:
        row = results_df.loc[date]
        params = [row['kappa'], row['lambda'], row['sigma'], row['rho'], row['v0']]
        avg_rate_r = row['avg_risk_free_rate']
        dividend_yield_q = row['implied_dividend_yield']
        
        day_df = full_df[full_df['QUOTE_DATE'] == date].copy()
        for c in ['C_BID', 'C_ASK', 'P_BID', 'P_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE']:
            if c in day_df.columns: day_df[c] = pd.to_numeric(day_df[c], errors='coerce')
            
        day_df['S0'] = day_df['UNDERLYING_LAST']
        day_df['K'] = day_df['STRIKE']
        day_df['tau'] = day_df['DTE'] / 365.0
        day_df['r'] = avg_rate_r
        day_df['q'] = dividend_yield_q
        day_df['log_moneyness'] = np.log(day_df['K'] / day_df['S0'])
        
        # 1. CALLS
        calls = day_df.copy()
        calls['marketPrice'] = (calls['C_BID'] + calls['C_ASK']) / 2.0
        calls = calls[(calls['marketPrice'] > 1.0) & (calls['tau'] > 0.02)]
        
        if not calls.empty:
            calls['modelPrice'] = predict_prices(model, calls, params, sx, sy)
            calls['abs_pct_err'] = np.abs(calls['modelPrice'] - calls['marketPrice']) / calls['marketPrice']
            errors_call.append(calls[['tau', 'K', 'S0', 'abs_pct_err']].copy())
            
            # Collect for scatter plot
            scatter_data_market.extend(calls['marketPrice'].values)
            scatter_data_model.extend(calls['modelPrice'].values)

        # 2. PUTS (Parity: P = C - S + K*exp(-rT))
        # Note: We use the SAME 'modelPrice' (Call) to derive Put Price
        if 'P_BID' in day_df.columns:
            puts = day_df.copy()
            puts['marketPrice'] = (puts['P_BID'] + puts['P_ASK']) / 2.0
            puts = puts[(puts['marketPrice'] > 1.0) & (puts['tau'] > 0.02)]
            
            if not puts.empty:
                # Predict Call Price first
                call_pred = predict_prices(model, puts, params, sx, sy)
                # Apply Parity
                puts['modelPrice'] = call_pred - puts['S0'] * np.exp(-dividend_yield_q * puts['tau']) + puts['K'] * np.exp(-avg_rate_r * puts['tau'])
                # Put prices can theoretically be negative if Arb exists or parity fails, clip at 0
                puts['modelPrice'] = puts['modelPrice'].clip(lower=0.0)
                
                puts['abs_pct_err'] = np.abs(puts['modelPrice'] - puts['marketPrice']) / puts['marketPrice']
                errors_put.append(puts[['tau', 'K', 'S0', 'abs_pct_err']].copy())

    # --- DATA PROCESSING FOR HEATMAP ---
    def process_bins(df_list):
        if not df_list: return pd.DataFrame()
        df = pd.concat(df_list)
        # Linear Moneyness K/S
        df['Moneyness'] = df['K'] / df['S0']
        
        # Bins as requested
        df['Mat_Bin'] = pd.cut(df['tau'], bins=[0, 0.25, 0.5, 1.0, 2.5], 
                               labels=['< 3M', '3M-6M', '6M-1Yr', '> 1Yr'])
        df['Mon_Bin'] = pd.cut(df['Moneyness'], bins=[0, 0.95, 1.05, 3.0], 
                               labels=['< 0.95', '0.95-1.05', '> 1.05'])
        return df.pivot_table(index='Mat_Bin', columns='Mon_Bin', values='abs_pct_err', aggfunc='mean')

    hm_call = process_bins(errors_call)
    hm_put = process_bins(errors_put)

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Shared settings
    kwargs = {'annot': True, 'fmt': ".1%", 'cmap': "coolwarm", 'vmin': 0, 'vmax': 0.15, 'cbar': False}
    
    sns.heatmap(hm_call, ax=axes[0], **kwargs)
    axes[0].set_title("Call Option MRE")
    axes[0].set_ylabel("Time to Maturity")
    axes[0].set_xlabel("Moneyness (Strike / Spot)")
    
    sns.heatmap(hm_put, ax=axes[1], **kwargs)
    axes[1].set_title("Put Option MRE (via Parity)")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Moneyness (Strike / Spot)")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_4_heatmap.png")
    plt.close()
    
    return scatter_data_market, scatter_data_model

def plot_5_regression(results_df, scatter_mkt, scatter_mod):
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

def plot_6_parameters(results_df):
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

def generate_table_1(results_df):
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
    print(f"Loading results from {config.BACKTEST_OUTPUT_FILE}...")
    if not Path(config.BACKTEST_OUTPUT_FILE).exists():
        raise FileNotFoundError("Run backtest.py first!")
        
    df = pd.read_csv(config.BACKTEST_OUTPUT_FILE)
    df.columns = df.columns.str.strip() # Fix whitespace
    
    required = ['date', 'in_sample_mre', 'out_sample_mre', 'avg_risk_free_rate', 'implied_dividend_yield', 'kappa', 'lambda', 'sigma', 'rho', 'v0']
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