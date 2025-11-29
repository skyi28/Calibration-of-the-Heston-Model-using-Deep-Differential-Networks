"""
Descriptive Analysis Module for Option Pricing Data.

Purpose:
    This script performs comprehensive descriptive analysis on two distinct datasets:
    1. Synthetic Data: Generated via the Heston Model (features and labels).
    2. Historic Data: Real-world market data (AAPL options).

Key Functionalities:
    - Data Loading: Ingests .npz files for synthetic data and raw .csv files for historic data.
    - Preprocessing: standardize column names and calculates financial metrics (Moneyness, Tau).
    - Data Attrition Reporting: Applies research-specific filters (Price, Maturity, Moneyness) 
      to historic data and logs exactly how many samples are lost at each step.
    - Statistical Analysis: Generates txt summaries (skew, kurtosis) and visualizations 
      (correlation heatmaps, histograms, boxplots) for the final datasets.

Usage:
    Run as a standalone script. It relies on a local 'config.py' for constants and file paths.
"""

import pandas as pd             # Data manipulation
import numpy as np              # Numerical operations
import matplotlib.pyplot as plt # Plotting
import seaborn as sns           # Plotting
import sys                      # System path manipulation
from pathlib import Path        # Path manipulations

# Dynamically add the parent directory to sys.path to ensure we can import the 'config' module
# regardless of where this script is executed from.
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import config

def generate_filtering_report(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Generates a comprehensive report on the data filtering process applied to the historic dataset.

    Parameters:
        df (pd.DataFrame): The historic dataset to be filtered.
        output_dir (Path): The directory in which the filtering report will be saved.

    Returns:
        pd.DataFrame: The filtered dataset after all filters have been applied.

    The report includes the following columns:
        Step (int): The step number in the filtering process.
        Description (str): A human-readable description of the filter applied.
        Remaining Samples (int): The number of samples remaining after the filter has been applied.
        Removed (int): The number of samples removed by the filter.
        Remaining % (float): The percentage of samples remaining after the filter has been applied.

    The report is saved to a CSV file in the specified output directory.
    """
    report = []
    
    # Snapshot of data size before any filtering
    initial_count = len(df)
    report.append({
        "Step": "1. Raw Data Extraction",
        "Description": "Initial import from CSVs",
        "Remaining Samples": initial_count,
        "Removed": 0,
        "Remaining %": 100.0
    })
    
    # Filter 1: Remove options below a minimum price threshold (removes penny options/noise)
    df_price = df[df['Call_Price'] > config.MIN_OPTION_PRICE].copy()
    
    count_price = len(df_price)
    report.append({
        "Step": "2. Price Filter",
        "Description": f"Call Price > ${config.MIN_OPTION_PRICE:.2f}",
        "Remaining Samples": count_price,
        "Removed": initial_count - count_price,
        "Remaining %": round((count_price / initial_count) * 100, 2)
    })
    
    # Filter 2: Remove options outside the specific time-to-maturity window
    # Short maturities are often unstable in Heston calibration.
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
    
    # Filter 3: Remove deep OTM/ITM options by filtering on Log Moneyness
    # Focuses the dataset on the liquid core around the strike price.
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
    
    # Export the attrition table to CSV for use in the research paper methodology section
    report_df = pd.DataFrame(report)
    save_path = output_dir / "data_filtering_process.csv"
    report_df.to_csv(save_path, index=False)
    
    print(f"\n[Report] Filtering table saved to: {save_path}")
    print(report_df.to_string(index=False))
    
    return df_final

def save_statistical_tables(df: pd.DataFrame, output_dir: Path):
    """
    Saves statistical tables of the input DataFrame to the specified directory.

    Two files are generated:
    1. descriptive_summary.txt: Standard pandas description (mean, std, min, max, quartiles)
    2. skew_kurtosis_summary.txt: Higher-order moments (Skewness and Kurtosis) to detect distribution asymmetry
    """
    print(f"Generating statistical tables in directory: {output_dir}")
    
    # Generate standard pandas description (mean, std, min, max, quartiles)
    try:
        summary = df.describe()
        with open(output_dir / "descriptive_summary.txt", "w") as f:
            f.write("Main Descriptive Statistics Summary\n" + "=" * 40 + "\n\n")
            f.write(summary.to_string())
    except Exception as e: print(f"Error saving summary: {e}")

    # Generate higher-order moments (Skewness and Kurtosis) to detect distribution asymmetry
    try:
        skew = df.skew(numeric_only=True)
        kurt = df.kurtosis(numeric_only=True)
        skew_kurt_df = pd.DataFrame({'skewness': skew, 'kurtosis': kurt})
        with open(output_dir / "skew_kurtosis_summary.txt", "w") as f:
            f.write("Skewness and Kurtosis Summary\n" + "=" * 40 + "\n\n")
            f.write(skew_kurt_df.to_string())
    except Exception as e: print(f"Error saving skew/kurt: {e}")

def generate_statistical_plots(df: pd.DataFrame, output_dir: Path):
    """
    Generates statistical plots for the given DataFrame and saves them to the given output directory.

    The generated plots include a correlation heatmap and a set of combined plots for each numerical column,
    which include a histogram with a kernel density estimate and a box plot for outlier detection.

    The combined plots are split into batches of COLS_PER_PLOT columns per image file, to prevent overcrowding.
    """
    print(f"Generating statistical plots in directory: {output_dir}")
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numerical_cols: return
    
    # Create Correlation Heatmap
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png')
        plt.close()
    except Exception as e: print(f"Error plot corr: {e}")

    # Create Distribution Plots (Histogram + Boxplot)
    # To prevent overcrowding, we batch columns into groups of 6 per image file.
    COLS_PER_PLOT = 6
    col_chunks = [numerical_cols[i:i + COLS_PER_PLOT] for i in range(0, len(numerical_cols), COLS_PER_PLOT)]

    for i, chunk in enumerate(col_chunks):
        num = len(chunk)
        # Create a subplot grid: Top row for Histograms, Bottom row for Boxplots
        fig, axes = plt.subplots(2, num, figsize=(num * 5, 10))
        if num == 1: axes = np.array(axes).reshape(2, 1)

        for j, col in enumerate(chunk):
            # Top row: Histogram with KDE
            sns.histplot(data=df, x=col, ax=axes[0, j], kde=True, color='#3B4CC0')
            axes[0, j].set_title(f"Histogram of {col}")
            axes[0, j].set_xlabel("")
            
            # Bottom row: Boxplot for outlier detection
            sns.boxplot(data=df, y=col, ax=axes[1, j], color='#3B4CC0')
            axes[1, j].set_title(f"Box Plot of {col}")

        plt.tight_layout()
        plt.savefig(output_dir / f"combined_statistics_plot_part_{i+1}.png")
        plt.close(fig)

def run_descriptive_analysis(df: pd.DataFrame, sub_dir_name: str):
    """
    Wrapper function to execute table saving and plotting for a specific dataset.
    Creates the output directory if it doesn't exist.
    """
    output_dir = config.DESC_ANALYSIS_OUTPUT_DIR / sub_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Running Analysis for '{sub_dir_name}' ---")
    save_statistical_tables(df, output_dir)
    generate_statistical_plots(df, output_dir)

def load_synthetic_data():
    """
    Loads the synthetic dataset generated by generate_dataset.py.
    
    Returns a pandas DataFrame containing both the feature inputs (X) and the labels (Y).
    X columns: kappa, lambda, sigma, rho, v0, r, q, tau, log_moneyness
    Y columns: price, d_kappa, d_lambda, d_sigma, d_rho, d_v0
    
    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    print("\nLoading Synthetic Data...")
    if not config.HESTON_DATASET_PATH.exists():
        raise FileNotFoundError("Run generate_dataset.py first.")
        
    data = np.load(config.HESTON_DATASET_PATH)
    
    # Map numpy arrays to DataFrame columns. 
    # 'q' represents Dividend Yield, 'tau' is time to maturity.
    X = pd.DataFrame(data['features'], columns=['kappa', 'lambda', 'sigma', 'rho', 'v0', 'r', 'q', 'tau', 'log_moneyness'])
    
    # Labels include option price and the Greeks (derivatives)
    Y = pd.DataFrame(data['labels'], columns=['price', 'd_kappa', 'd_lambda', 'd_sigma', 'd_rho', 'd_v0'])
    return pd.concat([X, Y], axis=1)

def load_raw_historic_data() -> pd.DataFrame:
    """
    Loads and processes raw historic data from multiple CSV files.

    The function loads, cleans and processes the data from the specified files. 
    It aggregates the data into a single DataFrame, cleans the column names, 
    and defines the essential columns. DIV_YIELD is sometimes missing or named 
    differently in raw data. The function also standardizes the 'DIV_YIELD' column 
    to 'q' to match Heston model notation. 

    The function calculates derived features, such as Mid-Price, Log Moneyness and Tau 
    (Time to Maturity). It removes rows where feature calculation failed (e.g. 
    division by zero).

    Returns:
        full_df (pd.DataFrame): The processed DataFrame containing the raw data and 
        derived features.
    """
    print("\nLoading Raw Historic Data...")
    dfs = []
    
    # Aggregate multiple daily/monthly CSV files into one list
    for f in config.AAPL_OPTION_FILES:
        try:
            df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
            dfs.append(df)
        except Exception as e: print(f"Skipping {f}: {e}")
            
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Clean dirty column names (remove brackets and whitespace)
    full_df.columns = full_df.columns.str.strip().str.replace(r'\[|\]', '', regex=True).str.strip()
    
    # Define essential columns. DIV_YIELD is sometimes missing or named differently in raw data.
    cols_to_use = ['C_BID', 'C_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE', 'C_IV', 'P_IV', 'DIV_YIELD']
    
    existing_cols = [c for c in cols_to_use if c in full_df.columns]
    full_df = full_df[existing_cols]
    
    # Ensure numeric types for all columns
    for c in existing_cols:
        full_df[c] = pd.to_numeric(full_df[c], errors='coerce')
    
    # Standardize 'DIV_YIELD' to 'q' to match Heston model notation
    if 'DIV_YIELD' in full_df.columns:
        full_df.rename(columns={'DIV_YIELD': 'q'}, inplace=True)
            
    # Calculate derived features
    # 1. Mid-Price: Average of Bid and Ask
    full_df['Call_Price'] = (full_df['C_BID'] + full_df['C_ASK']) / 2.0
    # 2. Log Moneyness: Log(Strike / Underlying)
    full_df['Moneyness_Log'] = np.log(full_df['STRIKE'] / full_df['UNDERLYING_LAST'])
    # 3. Tau (Time to Maturity): Days to Expiration / 365
    full_df['Tau_Years'] = full_df['DTE'] / 365.0
    
    # Remove rows where feature calculation failed (e.g. division by zero)
    full_df = full_df.dropna(subset=['Call_Price', 'Moneyness_Log', 'Tau_Years'])
    
    return full_df

def main():
    # Set seed for reproducible sampling
    """
    Main function to execute the data pipeline.

    The function consists of two parts: Historic Data Pipeline and Synthetic Data Pipeline.
    The Historic Data Pipeline loads raw CSVs, calculates features, applies strict filters, 
    generates an attrition report, and runs plotting and stats generation.
    The Synthetic Data Pipeline loads pre-generated .npz data, downsamples if necessary, 
    and runs plotting and stats generation.

    The function sets the seed for reproducible sampling and handles exceptions for each part.
    """
    config.set_reproducibility()
    
    # --- Part 1: Historic Data Pipeline ---
    try:
        # Load raw CSVs and calculate features
        df_hist_raw = load_raw_historic_data()
        
        # Define output directory
        output_dir = config.DESC_ANALYSIS_OUTPUT_DIR / "historic_aapl_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply strict filters and generate the attrition report
        df_hist_clean = generate_filtering_report(df_hist_raw, output_dir)
        
        # If dataset is too large, downsample for visualization speed
        if len(df_hist_clean) > config.DESC_ANALYSIS_MAX_LENGTH_BEFORE_SAMPLING:
            print(f"Subsampling historic data ({len(df_hist_clean)} -> {config.DESC_ANALYSIS_N_SAMPLES}) for plots.")
            df_plot = df_hist_clean.sample(config.DESC_ANALYSIS_N_SAMPLES, random_state=config.SEED)
        else:
            df_plot = df_hist_clean
            
        # Run plotting and stats generation
        run_descriptive_analysis(df_plot, "historic_aapl_data")
        
    except Exception as e:
        print(f"Historic Analysis Failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Part 2: Synthetic Data Pipeline ---
    try:
        # Load pre-generated .npz data
        df_syn = load_synthetic_data()
        
        # Downsample if necessary
        if len(df_syn) > config.DESC_ANALYSIS_MAX_LENGTH_BEFORE_SAMPLING:
            print(f"Subsampling synthetic data ({len(df_syn)} -> {config.DESC_ANALYSIS_N_SAMPLES}) for plots.")
            df_syn = df_syn.sample(config.DESC_ANALYSIS_N_SAMPLES, random_state=config.SEED)
        
        # Run plotting and stats generation
        run_descriptive_analysis(df_syn, "synthetic_data")
    except Exception as e:
        print(f"Synthetic Analysis Failed: {e}")

if __name__ == "__main__":
    main()