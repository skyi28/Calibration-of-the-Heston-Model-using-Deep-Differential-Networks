"""
Script: Heston Model Dataset Generator
======================================

Purpose:
    Generates a large-scale synthetic dataset of European Call Option prices and their 
    parameter sensitivities (gradients) under the Heston Stochastic Volatility Model.

    This data is typically used to train Neural Networks for "Neural Pricing" or 
    calibration acceleration, allowing a model to learn the mapping from 
    model parameters -> price/greeks without running the expensive analytic engine 
    during live inference.

Methodology:
    1.  Sampling: Uses Latin Hypercube Sampling (LHS) via scipy.stats.qmc to ensure 
        efficient, space-filling coverage of the high-dimensional parameter space 
        (Kappa, Theta, Sigma, Rho, v0, etc.), which is superior to random Monte Carlo sampling.
    2.  Pricing: Utilizes QuantLib's `AnalyticHestonEngine` (based on Fourier 
        Transforms/Integration) for ground-truth pricing.
    3.  Differentiation: Computes sensitivities (gradients) of the price with respect 
        to Heston parameters using Central Finite Differences.
    4.  Parallelization: Distributes the compute load across all available CPU cores 
        using `joblib`.

Dependencies:
    - QuantLib: For financial models and pricing engines.
    - config.py: Must contain `HESTON_PARAMS` (list), `PARAM_RANGES` (dict), and `NUM_SAMPLES` (int).
"""

import QuantLib as ql                       # For option pricing
import numpy as np                          # For numerical operations
from scipy.stats import qmc                 # For Latin Hypercube Sampling
from tqdm import tqdm                       # For progress bars
from pathlib import Path                    # For file paths
import sys                                  # For file paths
from joblib import Parallel, delayed        # For parallel execution
from typing import Tuple                    # For type hinting

# Add project root to path to import the config file
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
import config                               # Configuration file

def calculate_price_and_grads(params: dict, epsilon: float = 1e-4) -> Tuple[dict, float, list]|None:
    """
    Computes the price and its parameter sensitivities (gradients) under the Heston Stochastic Volatility Model.

    Parameters:
        params (dict): A dictionary containing the Heston model parameters.
        epsilon (float): The perturbation size for Central Finite Differences.

    Returns:
        A tuple containing the input parameters, option price, and the parameter sensitivities (gradients).
        None if the option price is negative, zero, not finite, or an error occurs.
    """
    try:
        # --- 1. Market & Instrument Setup ---
        S0 = 1.0  # Normalized Spot Price (Standard practice in ML for finance to ensure scale invariance)
        
        # Calculate Strike (K) from Log-Moneyness. 
        # Formula: K = S0 * e^{log_moneyness}. 
        # If log_moneyness < 0, option is OTM (for calls); if > 0, ITM.
        K = S0 * np.exp(params['log_moneyness']) 
        
        # QuantLib Date setup, any arbitrary date can be used
        today = ql.Date(1, 1, 2024)
        ql.Settings.instance().evaluationDate = today
        # Day count convention
        # Act/365 is standard for crypto/fx, Act/360 or 30/360 often used elsewhere
        day_count = ql.Actual365Fixed() 
        
        # Calculate maturity date of the option based on 'tau' (Time to maturity in years)
        maturity = today + ql.Period(int(params['tau'] * 365), ql.Days)
        
        # Define the Instrument (European Call Option)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)
        
        # --- 2. Heston Process Configuration ---
        # Market Handles
        spot_h = ql.QuoteHandle(ql.SimpleQuote(S0))
        # Risk-free rate curve (Flat forward assumption)
        rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, params['r'], day_count))
        # Dividend yield (Assumed 0 for this dataset)
        div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, params['q'], day_count))
        
        # Initialize Stochastic Process
        # Note: QuantLib Heston Signature is (RiskFree, Div, Spot, v0, kappa, theta, sigma, rho).
        # 'lambda' in params maps to 'theta' (Long-run variance) in QuantLib context.
        process = ql.HestonProcess(
            rate_h,             # Risk-free rate
            div_h,              # Dividend yield
            spot_h,             # Underyling price
            params['v0'],       # Initial Variance
            params['kappa'],    # Mean Reversion Speed
            params['lambda'],   # Long-run Variance (Theta)
            params['sigma'],    # Volatility of Volatility
            params['rho']       # Correlation between Asset and Variance Brownian motions
        )
        
        # Set Engine: Analytic engine uses semi-closed form integration (efficient)
        engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
        option.setPricingEngine(engine)
        
        # --- 3. Base Pricing ---
        price = option.NPV()
        
        # Validation check: Filter out numerical instabilities (NaN or negative prices)
        if not (np.isfinite(price) and price >= 0): return None

        # --- 4. Gradient Calculation (Sensitivities) ---
        # We compute dPrice/dParam using Central Finite Differences: (f(x+h) - f(x-h)) / 2h
        grads = []
        for p_name in config.HESTON_PARAMS:
            # Create perturbed parameter sets
            p_up = params.copy(); p_up[p_name] += epsilon
            p_dn = params.copy(); p_dn[p_name] -= epsilon
            
            # Re-instantiate processes using perturbed values
            proc_up = ql.HestonProcess(rate_h, div_h, spot_h, 
                p_up['v0'], p_up['kappa'], p_up['lambda'], p_up['sigma'], p_up['rho'])
            proc_dn = ql.HestonProcess(rate_h, div_h, spot_h, 
                p_dn['v0'], p_dn['kappa'], p_dn['lambda'], p_dn['sigma'], p_dn['rho'])
            
            # Price Up-move
            option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc_up)))
            pup = option.NPV()
            
            # Price Down-move
            option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc_dn)))
            pdn = option.NPV()
            
            if not (np.isfinite(pup) and np.isfinite(pdn)): return None
            
            # Append gradient
            grads.append((pup - pdn) / (2 * epsilon))
            
        return params, price, grads

    except Exception:
        # Return None to signify failure
        return None

def main(n_samples: int =200_000) -> None:
    """
    Generates a large-scale synthetic dataset of European Call Option prices and their 
    parameter sensitivities (gradients) under the Heston Stochastic Volatility Model.

    This data is typically used to train Neural networks for "Neural Pricing" or 
    calibration acceleration, allowing a model to learn the mapping from 
    model parameters -> price/greeks without running the expensive analytic engine 
    during live inference.

    Parameters:
        n_samples (int): The number of samples to generate. Default is 200_000.

    Returns:
        None
    """
    print("Setting seed")
    config.set_reproducibility() 
    
    print(f"Generating {n_samples} samples...")
    # --- 1. Latin Hypercube Sampling (LHS) ---
    # LHS generates a quasi-random distribution that covers the parameter hypercube 
    # more evenly than standard random sampling, reducing data requirements for training.
    sampler = qmc.LatinHypercube(d=len(config.PARAM_RANGES))
    
    # Scale sample from [0, 1] unit hypercube to actual parameter domains defined in config
    sample_scaled = qmc.scale(sampler.random(n_samples), 
                              [v[0] for v in config.PARAM_RANGES.values()], # Lower bounds
                              [v[1] for v in config.PARAM_RANGES.values()]) # Upper bounds
    
    # Convert numpy array to list of dictionaries for readability in the processing loop
    cols = list(config.PARAM_RANGES.keys())
    samples = [dict(zip(cols, row)) for row in sample_scaled]
    
    # --- 2. Parallel Execution ---
    # Use joblib to utilize all CPU cores (-1). 
    # tqdm provides a progress bar for the long-running process.
    results = Parallel(n_jobs=-1)(delayed(calculate_price_and_grads)(s) for s in tqdm(samples))
    
    # --- 3. Post-Processing & Filtering ---
    feats, labs = [], []
    for res in results:
        if res:
            p, price, grads = res
            # Construct Feature Vector (Inputs)
            # [Kappa, Theta(lambda), VolOfVol, Rho, v0, RiskFreeRate, DividenYield, TimeToMaturity, Moneyness]
            f_vec = [p['kappa'], p['lambda'], p['sigma'], p['rho'], p['v0'], 
                     p['r'], p['q'], p['tau'], p['log_moneyness']] 
            
            # Construct Label Vector (Outputs)
            # [Option Price, Gradient_1, Gradient_2, ...]
            l_vec = [price] + grads
            
            feats.append(f_vec)
            labs.append(l_vec)
            
    print(f"Valid samples generated: {len(feats)} / {n_samples}")
    
    # --- 4. Serialization ---
    # Save as compressed numpy arrays for efficient loading during model training
    np.savez_compressed(config.HESTON_DATASET_PATH, 
                        features=np.array(feats, dtype=np.float32), 
                        labels=np.array(labs, dtype=np.float32))
    print(f"Dataset saved to: {config.HESTON_DATASET_PATH}")

if __name__ == "__main__":
    # Run the main function if the script is run directly (not imported)
    main(n_samples=config.NUM_SAMPLES)