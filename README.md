# Deep Differential Networks for Heston Model Calibration

## 1. Overview

This project implements a state-of-the-art **Deep Differential Network (DDN)** to solve the calibration problem for the Heston Stochastic Volatility Model. 

By combining deep learning with financial mathematics, this framework acts as a surrogate modeling engine that is orders of magnitude faster than traditional numerical integration methods (e.g., QuantLib) while maintaining research-grade precision.

**Key Difference:** Unlike standard neural networks that only learn to predict option *prices*, this DDN is trained using **Sobolev Training**. It learns to predict both the **Price** and the **Gradients (Greeks)** with respect to the model parameters. These exact gradients are then used to drive a deterministic optimization loop (L-BFGS-B), resulting in extremely fast and robust calibration to market data.

### Basis & Attribution
This project is based on the methodology proposed in:
> *Zhang, C., Amici, G., & Morandotti, M. (2024). "Calibrating the Heston model with deep differential networks."*

**Extensions:** This repository further develops the ideas in the paper by:
1.  Implementing a robust **L-BFGS-B calibration engine** (replacing the paper's Adam-based optimization).
2.  Introducing **Log-Moneyness** input transformations for higher gradient stability.
3.  Implementing **Implied Risk-Free Rate** calculation via Put-Call Parity to handle real-world regime shifts (dividends/interest rates).
4.  Providing a 7-year longitudinal backtest on AAPL option data (2016–2023).
5.  Detailed statistical analysis of the data attrition and filtering process.

---

## 2. The Problem vs. The Solution

### The Problem
Calibrating the Heston model requires finding five parameters ($\kappa, \lambda, \sigma, \rho, v_0$) that minimize the error between model prices and market prices. 
*   **Computational Cost:** The analytical Heston formula involves complex integration, making it slow to evaluate thousands of times inside an optimizer.
*   **Non-Convexity:** The error surface is bumpy. Gradient-free optimizers (Nelder-Mead) are slow, while gradient-based optimizers require expensive finite-difference approximations.

### The Solution
1.  **The Surrogate:** We train a Neural Network to memorize the Heston formula. It predicts prices in microseconds.
2.  **Differential Learning:** The network is trained to minimize the error of the **Output** AND the **Derivative of the Output**. This forces the network to learn the exact shape of the pricing surface.
3.  **Gradient-Based Calibration:** During calibration, we extract the exact gradients from the network (via Automatic Differentiation) and feed them to Scipy's L-BFGS-B optimizer. This allows the solver to jump directly to the global minimum.

---

## 3. Project Structure

```text
├── config.py                   # Global configuration (Paths, Hyperparameter ranges, Seeds)
├── data/                       # Stores datasets and results
│   ├── heston_dataset.npz      # Synthetic training data
│   ├── AAPL_stock_history.csv  # AAPL historic stock data (yfinance)
│   ├── descriptive_analysis/   # Output tables and plots from data analysis
│   └── kaggle                  # Subfolder for option data
│       └── aapl_year_year.csv  # Historical option data
├── model/
│   ├── ddn.py                  # Custom Keras Model (Deep Differential Network class)
│   ├── ddn.weights.h5          # Trained model weights
│   └── scaler_*.save           # Scikit-learn scalers for inputs/outputs
├── scripts/
│   ├── generate_dataset.py     # Generates synthetic Heston data (Price + Greeks)
│   ├── descriptive_analysis.py # Generates statistical reports and data filtering tables
│   ├── tune.py                 # Hyperparameter Tuning (Hyperband)
│   ├── train.py                # Trains the DDN (AdamW + Cosine Decay)
│   ├── backtest.py             # Runs historical calibration (The "Engine")
│   └── generate_plots.py       # Visualizes final backtest results
└── plots/                      # Output directory for final thesis graphs
```

---

## 4. Usage Guide (Execution Order)

To replicate the results, execute the scripts in the following order:

### Step 1: Generate the "Textbook"
Create the synthetic dataset using QuantLib. This generates random Heston parameters via Latin Hypercube Sampling and calculates the corresponding Prices and Greeks.
```bash
python scripts/generate_dataset.py
```
*   *Output:* `data/heston_dataset.npz` (~200k samples)

### Step 2: Analyze Data & Filtering
Perform a descriptive statistical analysis on both the synthetic dataset and the historical AAPL data. This script generates histograms, correlation heatmaps, and the crucial **Data Attrition Table** showing exactly how many options remain after applying liquidity and moneyness filters.
```bash
python scripts/descriptive_analysis.py
```
*   *Output:* `data/descriptive_analysis/` (Tables and PNGs)

### Step 3: Optimize Architecture (Optional)
Use the Hyperband algorithm to find the optimal number of layers, neurons, and learning rate schedules.
```bash
python scripts/tune.py
```
*   *Output:* `models/best_hps.json`

### Step 4: Train the "Brain"
Train the Deep Differential Network. This uses **AdamW** optimizer and a **Cosine Decay with Restarts** scheduler to achieve high precision ($MSE \approx 10^{-6}$).
```bash
python scripts/train.py
```
*   *Output:* `models/ddn.weights.h5` and scalers.

### Step 5: Run the Backtest (The "Detective")
Calibrate the model to historical AAPL option data (2016–2023). This script:
1.  Calculates the daily implied risk-free rate using Puts/Calls.
2.  Filters for liquid options (matches Step 2 logic).
3.  Calibrates on 80% of data (In-Sample).
4.  Tests accuracy on the hidden 20% (Out-of-Sample).
```bash
python scripts/backtest.py
```
*   *Output:* `data/backtest_results.csv`

### Step 6: Visualize Results
Generate the thesis-ready plots and tables (Error Heatmaps, Time Series Analysis, Regime Table).
```bash
python scripts/generate_plots.py
```
*   *Output:* Images in `plots/` folder.

---

## 5. Technical Implementation Details

### Homogeneity & Log-Moneyness
To ensure the model generalizes across all stock prices ($S_0$), we exploit the homogeneity of the Heston model.
*   **Input:** We do not feed $S_0$ or $K$ directly. We feed **Log-Moneyness**: $\ln(K / S_0)$.
*   **Output:** The network predicts the **Normalized Price**: $Price / S_0$.
This allows a network trained on a stock price of $1.00 to price options on AAPL at $150.00 perfectly.

### High-Precision Training
Standard regression training is insufficient for calibration. We use:
*   **Loss Function:** Weighted sum of Price MSE and Gradient MSE.
*   **Scaling:** Inputs scaled to $[-1, 1]$ to optimize Tanh/Softplus activation zones.
*   **Optimizer:** AdamW (Decoupled Weight Decay) to prevent overfitting without the gradient noise introduced by Dropout.

### Calibration Logic
For every trading day:
1.  **Filtering:** Options < $0.50 are removed to prevent "penny option" noise. Maturities > 2 years are removed to avoid extrapolation.
2.  **Optimization:** Scipy's `minimize(method='L-BFGS-B')` is called.
3.  **Objective:** Minimize the **Squared Relative Error** (MRE).
4.  **Result:** The system typically converges in < 50 iterations (< 0.5 seconds per day).

---

## 6. Performance

The resulting model demonstrates exceptional accuracy and robustness:
*   **In-Sample MRE:** ~3.7% (Beating the paper's reported ~6.0%).
*   **Out-of-Sample MRE:** ~4.1% (Demonstrating strong generalization).
*   **Stability:** Successfully handles the 2020 COVID volatility spike and the 2022 interest rate regime shift without requiring retraining.