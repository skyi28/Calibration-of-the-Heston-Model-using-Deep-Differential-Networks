"""
Script: Heston Calibration Validation
=====================================

Purpose:
    This script performs a rigorous, methodologically pure validation of the Heston
    parameters calibrated by the Deep Differential Network (DDN).

    Instead of using the DDN as a surrogate pricer for evaluation, this script
    takes the calibrated parameters from the backtest results and prices the
    corresponding option chains using QuantLib's true `AnalyticHestonEngine`.
    This cleanly separates the performance of the calibration method from any
    potential approximation error in the DDN itself.

Methodology:
    1.  Data Loading: Loads the `backtest_results.csv` which contains the
        calibrated Heston parameters for each day, and the raw AAPL option data.
    2.  Regime-Based Sampling: To ensure a representative and computationally
        tractable analysis, the script samples ~100 trading days, balanced
        across key historical market regimes (e.g., Pre-Volmageddon, COVID Crash,
        Inflationary period).
    3.  True Heston Pricing: For each sampled day, it iterates through the
        corresponding option chain and prices each option using the
        `AnalyticHestonEngine` initialized with that day's calibrated parameters.
    4.  MRE Calculation: It computes the Mean Relative Error (MRE) between the
        true Heston model prices and the observed market prices.
    5.  Reporting: The script outputs a clear summary report, showing the
        average MRE for each market regime and an overall average MRE, providing
        a definitive measure of the calibration quality.

Dependencies:
    - QuantLib-Python: For the `AnalyticHestonEngine`.
    - pandas, numpy, tqdm
    - Your project's `config.py` file for file paths.

"""

import pandas as pd
import numpy as np
import QuantLib as ql
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root to path to import config
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
import config

# --- CONFIGURATION ---
config.set_reproducibility()

# Define market regimes for balanced sampling
REGIMES = {
    "Pre-Volmageddon": (pd.to_datetime("2015-01-01"), pd.to_datetime("2018-02-01")),
    "Trade War":       (pd.to_datetime("2018-02-02"), pd.to_datetime("2020-02-01")),
    "COVID Crash":     (pd.to_datetime("2020-02-02"), pd.to_datetime("2020-06-01")),
    "Recovery":        (pd.to_datetime("2020-06-02"), pd.to_datetime("2022-01-01")),
    "Inflation":       (pd.to_datetime("2022-01-02"), pd.to_datetime("2024-01-01")),
}
SAMPLES_PER_REGIME = config.TOTAL_SAMPLES // len(REGIMES)

def calculate_true_heston_mre(day_chain: pd.DataFrame,
                              calibrated_params: dict,
                              risk_free_rate: float,
                              dividend_yield: float) -> float:
    """
    Prices an option chain with QuantLib's AnalyticHestonEngine and calculates MRE.

    Args:
        day_chain: DataFrame of options for a single trading day.
        calibrated_params: Dictionary of Heston parameters {kappa, lambda, ...}.
        risk_free_rate: The risk-free rate for the day.
        dividend_yield: The dividend yield for the day.

    Returns:
        The Mean Relative Error for the given day's option chain.
    """
    if day_chain.empty:
        return np.nan

    # --- QuantLib Setup ---
    trade_date_pd = day_chain['QUOTE_DATE'].iloc[0]
    trade_date_ql = ql.Date(trade_date_pd.day, trade_date_pd.month, trade_date_pd.year)
    ql.Settings.instance().evaluationDate = trade_date_ql
    day_count = ql.Actual365Fixed()

    # Create market data handles
    spot_price = day_chain['S0'].iloc[0]
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(trade_date_ql, risk_free_rate, day_count))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(trade_date_ql, dividend_yield, day_count))

    # Create Heston Process and Engine
    process = ql.HestonProcess(
        rate_h,
        div_h,
        spot_h,
        calibrated_params['v0'],
        calibrated_params['kappa'],
        calibrated_params['lambda'],
        calibrated_params['sigma'],
        calibrated_params['rho']
    )
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process))

    model_prices = []
    market_prices = []

    for _, option_row in day_chain.iterrows():
        market_price = option_row['marketPrice']
        if market_price < config.MIN_OPTION_PRICE:
            continue

        maturity_date = trade_date_ql + ql.Period(int(option_row['DTE']), ql.Days)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, option_row['K'])
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)

        option.setPricingEngine(engine)
        
        try:
            model_prices.append(option.NPV())
            market_prices.append(market_price)
        except Exception:
            # Skip if QuantLib fails for a specific option
            continue

    if not market_prices:
        return np.nan

    # Calculate MRE
    model_prices = np.array(model_prices)
    market_prices = np.array(market_prices)
    mre = np.mean(np.abs(model_prices - market_prices) / market_prices)
    return mre


def main():
    """Main validation workflow."""
    print("--- Starting Heston Calibration Validation ---")

    # 1. Load Backtest Results and Raw Option Data
    print(f"Loading backtest results from {config.BACKTEST_OUTPUT_FILE}...")
    if not config.BACKTEST_OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Backtest results not found at {config.BACKTEST_OUTPUT_FILE}. Please run backtest.py first.")
    
    results_df = pd.read_csv(config.BACKTEST_OUTPUT_FILE, parse_dates=['date'])
    results_df = results_df.set_index('date')

    print("Loading raw AAPL option chain data (this may take a moment)...")
    raw_df = pd.concat([pd.read_csv(f, on_bad_lines='skip', low_memory=False) for f in config.AAPL_OPTION_FILES])
    raw_df.columns = raw_df.columns.str.strip().str.replace(r'\[|\]', '', regex=True).str.strip()
    raw_df['QUOTE_DATE'] = pd.to_datetime(raw_df['QUOTE_DATE'])
    
    # 2. Sample Dates from Regimes
    print(f"Sampling {config.TOTAL_SAMPLES} days balanced across {len(REGIMES)} regimes...")
    
    def assign_regime(date):
        for label, (start, end) in REGIMES.items():
            if start <= date <= end:
                return label
        return "Other"

    results_df['regime'] = results_df.index.to_series().apply(assign_regime)
    
    # Ensure we only sample from defined regimes and handle cases with few samples
    valid_regimes_df = results_df[results_df['regime'] != "Other"]
    sampled_df = valid_regimes_df.groupby('regime').apply(
        lambda x: x.sample(n=min(len(x), SAMPLES_PER_REGIME), random_state=42)
    ).reset_index(level=0, drop=True)

    print(f"Successfully sampled {len(sampled_df)} days for validation.")

    # 3. Process Each Sampled Day
    validation_results = []
    
    progress_bar = tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Validating Days")
    for date, row in progress_bar:
        # Get calibrated params and rates for the day
        params = row[['kappa', 'lambda', 'sigma', 'rho', 'v0']].to_dict()
        r = row['avg_risk_free_rate']
        q = row['implied_dividend_yield']
        regime = row['regime']

        # Get the corresponding option chain
        day_chain = raw_df[raw_df['QUOTE_DATE'] == date].copy()
        
        # Pre-process the day's data
        for col in ['C_BID', 'C_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE']:
            day_chain[col] = pd.to_numeric(day_chain[col], errors='coerce')
            
        day_chain.dropna(subset=['C_BID', 'C_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE'], inplace=True)
        day_chain['marketPrice'] = (day_chain['C_BID'] + day_chain['C_ASK']) / 2.0
        day_chain['S0'] = day_chain['UNDERLYING_LAST']
        day_chain['K'] = day_chain['STRIKE']
        
        day_chain['tau'] = day_chain['DTE'] / 365.0
        day_chain['log_moneyness'] = np.log(day_chain['K'] / day_chain['S0'])

        # Apply the exact same filters as in the backtest script
        filtered_chain = day_chain[
            (day_chain['log_moneyness'] >= config.MIN_LOG_MONEYNESS) &
            (day_chain['log_moneyness'] <= config.MAX_LOG_MONEYNESS) &
            (day_chain['marketPrice'] > config.MIN_OPTION_PRICE) &
            (day_chain['tau'] >= config.BACKTEST_MIN_TAU) &
            (day_chain['tau'] <= config.BACKTEST_MAX_TAU)
        ].copy()
        
        # Calculate MRE using true Heston pricer
        true_mre = calculate_true_heston_mre(filtered_chain, params, r, q)

        validation_results.append({
            'date': date,
            'regime': regime,
            'true_heston_mre': true_mre,
            'surrogate_mre': row['out_sample_mre'] # Keep for comparison
        })
    
    # 4. Report Results
    final_df = pd.DataFrame(validation_results).dropna()
    
    print("\n--- Validation Report ---")
    print("Comparing MRE from true QuantLib Heston pricer vs. the DDN surrogate.\n")

    # Per-regime results
    regime_summary = final_df.groupby('regime')[['true_heston_mre', 'surrogate_mre']].mean()
    print("Average MRE by Market Regime:")
    print("-" * 73)
    print(f"{'Regime':<30} | {'True Heston MRE':<20} | {'DDN Surrogate MRE':<20}")
    print("-" * 73)
    for regime, row in regime_summary.iterrows():
        true_mre_str = f"{row['true_heston_mre']:.2%}"
        surrogate_mre_str = f"{row['surrogate_mre']:.2%}"
        print(f"{regime:<30} | {true_mre_str:<20} | {surrogate_mre_str:<20}")
    print("-" * 73)


    # --- The rest of the script (overall average calculation) can be updated too ---
    # Overall average
    overall_avg_true_mre = final_df['true_heston_mre'].mean()
    overall_avg_surrogate_mre = final_df['surrogate_mre'].mean() # Calculate this as well

    print("\n" + "-" * 43)
    print(f"Overall Average True Heston MRE  : {overall_avg_true_mre:.2%}")
    print(f"Overall Average DDN Surrogate MRE: {overall_avg_surrogate_mre:.2%}")
    print("-" * 43)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()