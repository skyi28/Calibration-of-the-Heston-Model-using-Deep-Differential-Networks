import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import config

# ==============================================================================
#  FILTERING REPORT LOGIC (NEW)
# ==============================================================================

def generate_filtering_report(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Applies sequential filters to the raw option data and generates a 
    'Data Attrition Table' for the research paper.

    Tracks the number of samples removed at each step (Price, Maturity, Moneyness).

    Parameters:
        df (pd.DataFrame): Raw historic dataframe with derived columns (Price, Tau, Moneyness).
        output_dir (Path): Directory to save the CSV report.

    Returns:
        pd.DataFrame: The final cleaned dataframe used for training/analysis.
    """
    report = []
    
    # 1. Initial State
    initial_count = len(df)
    report.append({
        "Step": "1. Raw Data Extraction",
        "Description": "Initial import from CSVs",
        "Remaining Samples": initial_count,
        "Removed": 0,
        "Remaining %": 100.0
    })
    
    # 2. Price Filter
    df_price = df[df['Call_Price'] > config.MIN_OPTION_PRICE].copy()
    
    count_price = len(df_price)
    report.append({
        "Step": "2. Price Filter",
        "Description": f"Call Price > ${config.MIN_OPTION_PRICE:.2f}",
        "Remaining Samples": count_price,
        "Removed": initial_count - count_price,
        "Remaining %": round((count_price / initial_count) * 100, 2)
    })
    
    # 3. Maturity Filter
    # Heston is often unstable for very short maturities (e.g. < 14 days)
    df_tau = df_price[df_price['Tau_Years'] >= config.MIN_TAU].copy()
    df_tau = df_price[df_price['Tau_Years'] <= config.MAX_TAU].copy()
    
    count_tau = len(df_tau)
    report.append({
        "Step": "3. Maturity Filter",
        "Description": f"Time to Maturity > {config.MIN_TAU * 365:.2f} Days",
        "Remaining Samples": count_tau,
        "Removed": count_price - count_tau,
        "Remaining %": round((count_tau / initial_count) * 100, 2)
    })
    
    # 4. Moneyness Filter
    # Focus on the liquid core: Log Moneyness [-0.25, 0.25]
    df_final = df_tau[
        (df_tau['Moneyness_Log'] >= config.MIN_LOG_MONEYNESS) & 
        (df_tau['Moneyness_Log'] <= config.MAX_LOG_MONEYNESS)
    ].copy()
    
    count_final = len(df_final)
    report.append({
        "Step": "4. Moneyness Filter",
        "Description": f"Log Moneyness [{config.MIN_LOG_MONEYNESS}, {config.MAX_LOG_MONEYNESS}]",
        "Remaining Samples": count_final,
        "Removed": count_tau - count_final,
        "Remaining %": round((count_final / initial_count) * 100, 2)
    })
    
    # Save Table
    report_df = pd.DataFrame(report)
    save_path = output_dir / "data_filtering_process.csv"
    report_df.to_csv(save_path, index=False)
    
    print(f"\n[Report] Filtering table saved to: {save_path}")
    print(report_df.to_string(index=False))
    
    return df_final

# ==============================================================================
#  STATISTICAL ANALYSIS LOGIC
# ==============================================================================

def save_statistical_tables(df: pd.DataFrame, output_dir: Path):
    print(f"Generating statistical tables in directory: {output_dir}")
    
    # 1. Main Description
    try:
        summary = df.describe()
        with open(output_dir / "descriptive_summary.txt", "w") as f:
            f.write("Main Descriptive Statistics Summary\n" + "=" * 40 + "\n\n")
            f.write(summary.to_string())
    except Exception as e: print(f"Error saving summary: {e}")

    # 2. Skew/Kurtosis
    try:
        skew = df.skew(numeric_only=True)
        kurt = df.kurtosis(numeric_only=True)
        skew_kurt_df = pd.DataFrame({'skewness': skew, 'kurtosis': kurt})
        with open(output_dir / "skew_kurtosis_summary.txt", "w") as f:
            f.write("Skewness and Kurtosis Summary\n" + "=" * 40 + "\n\n")
            f.write(skew_kurt_df.to_string())
    except Exception as e: print(f"Error saving skew/kurt: {e}")

def generate_statistical_plots(df: pd.DataFrame, output_dir: Path):
    print(f"Generating statistical plots in directory: {output_dir}")
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numerical_cols: return
    
    # Correlation Heatmap
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png')
        plt.close()
    except Exception as e: print(f"Error plot corr: {e}")

    # Histograms and Boxplots
    COLS_PER_PLOT = 6
    col_chunks = [numerical_cols[i:i + COLS_PER_PLOT] for i in range(0, len(numerical_cols), COLS_PER_PLOT)]

    for i, chunk in enumerate(col_chunks):
        num = len(chunk)
        fig, axes = plt.subplots(2, num, figsize=(num * 5, 10))
        if num == 1: axes = np.array(axes).reshape(2, 1)

        for j, col in enumerate(chunk):
            # Hist
            sns.histplot(data=df, x=col, ax=axes[0, j], kde=True, color='#3B4CC0')
            axes[0, j].set_title(f"Histogram of {col}")
            axes[0, j].set_xlabel("")
            # Box
            sns.boxplot(data=df, y=col, ax=axes[1, j], color='#3B4CC0')
            axes[1, j].set_title(f"Box Plot of {col}")

        plt.tight_layout()
        plt.savefig(output_dir / f"combined_statistics_plot_part_{i+1}.png")
        plt.close(fig)

def run_descriptive_analysis(df: pd.DataFrame, sub_dir_name: str):
    output_dir = config.DESC_ANALYSIS_OUTPUT_DIR / sub_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Running Analysis for '{sub_dir_name}' ---")
    save_statistical_tables(df, output_dir)
    generate_statistical_plots(df, output_dir)

# ==============================================================================
#  DATA LOADERS
# ==============================================================================

def load_synthetic_data():
    print("\nLoading Synthetic Data...")
    if not config.HESTON_DATASET_PATH.exists():
        raise FileNotFoundError("Run generate_dataset_hom.py first.")
        
    data = np.load(config.HESTON_DATASET_PATH)
    X = pd.DataFrame(data['features'], columns=['kappa', 'lambda', 'sigma', 'rho', 'v0', 'r', 'tau', 'log_moneyness'])
    Y = pd.DataFrame(data['labels'], columns=['price', 'd_kappa', 'd_lambda', 'd_sigma', 'd_rho', 'd_v0'])
    return pd.concat([X, Y], axis=1)

def load_raw_historic_data():
    """Loads raw data without dropping rows yet."""
    print("\nLoading Raw Historic Data...")
    dfs = []
    
    for f in config.AAPL_OPTION_FILES:
        try:
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
            dfs.append(df)
        except Exception as e: print(f"Skipping {f}: {e}")
            
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Cleaning columns
    full_df.columns = full_df.columns.str.strip().str.replace(r'\[|\]', '', regex=True).str.strip()
    cols_to_use = ['C_BID', 'C_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE', 'C_IV', 'P_IV']
    full_df = full_df[cols_to_use]
    
    # Conversions
    for c in cols_to_use:
        full_df[c] = pd.to_numeric(full_df[c], errors='coerce')
            
    # Calculate Derived Columns needed for filtering
    full_df['Call_Price'] = (full_df['C_BID'] + full_df['C_ASK']) / 2.0
    full_df['Moneyness_Log'] = np.log(full_df['STRIKE'] / full_df['UNDERLYING_LAST'])
    full_df['Tau_Years'] = full_df['DTE'] / 365.0
    
    # Initial basic clean (drop rows where calculation failed, e.g. div by zero)
    full_df = full_df.dropna(subset=['Call_Price', 'Moneyness_Log', 'Tau_Years'])
    
    return full_df

# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    config.set_reproducibility()
    
    # 1. Historic Data Analysis (With Filtering Report)
    try:
        # Load Raw
        df_hist_raw = load_raw_historic_data()
        
        # Apply Filters & Generate Table 1
        output_dir = config.DESC_ANALYSIS_OUTPUT_DIR / "historic_aapl_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_hist_clean = generate_filtering_report(df_hist_raw, output_dir)
        
        # Subsample for plotting if huge
        if len(df_hist_clean) > config.DESC_ANALYSIS_MAX_LENGTH_BEFORE_SAMPLING:
            print(f"Subsampling historic data ({len(df_hist_clean)} -> {config.DESC_ANALYSIS_N_SAMPLES}) for plots.")
            df_plot = df_hist_clean.sample(config.DESC_ANALYSIS_N_SAMPLES, random_state=config.SEED)
        else:
            df_plot = df_hist_clean
            
        run_descriptive_analysis(df_plot, "historic_aapl_data")
        
    except Exception as e:
        print(f"Historic Analysis Failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Synthetic Data Analysis
    try:
        df_syn = load_synthetic_data()
        if len(df_syn) > config.DESC_ANALYSIS_MAX_LENGTH_BEFORE_SAMPLING:
            print(f"Subsampling synthetic data ({len(df_syn)} -> {config.DESC_ANALYSIS_N_SAMPLES}) for plots.")
            df_syn = df_syn.sample(config.DESC_ANALYSIS_N_SAMPLES, random_state=config.SEED)
        run_descriptive_analysis(df_syn, "synthetic_data")
    except Exception as e:
        print(f"Synthetic Analysis Failed: {e}")

if __name__ == "__main__":
    main()