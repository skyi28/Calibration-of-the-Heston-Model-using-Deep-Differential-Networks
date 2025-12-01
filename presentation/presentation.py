import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import joblib
import tensorflow as tf
from pathlib import Path
import json
import os
import yfinance as yf

# Add project root to path to import the Deep Differential Network
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from model.ddn import DeepDifferentialNetwork            # For the Deep Differential Network
import config

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heston Calibration via DDN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (DARK MODE + RED ACCENTS) ---
st.markdown("""
<style>
    /* 1. Global Background & Text */
    .stApp {
        background-color: #111111;
        color: #FFFFFF;
    }
    
    p, h1, h2, h3, h4, h5, h6, li, span, label, .stSelectbox, .stDateInput, .stMarkdown {
        color: #FFFFFF !important;
    }

    /* 2. Highlight Text (Red) */
    .highlight {
        color: #FF4B4B !important;
        font-weight: bold;
    }

    /* 3. Custom Red Boxes */
    .red-box {
        background-color: #2b0505;
        border-left: 5px solid #FF4B4B;
        padding: 1.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .red-box h4 { color: #FF4B4B !important; margin-top: 0; }
    .red-box ul { margin-bottom: 0; }

    /* 4. Streamlit UI Overrides */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1E1E1E; color: white; border-radius: 4px 4px 0px 0px; }
    .stTabs [aria-selected="true"] { background-color: #FF4B4B !important; color: white !important; }
    
    /* Input widgets background */
    .stSelectbox > div > div { background-color: #1E1E1E; color: white; }
    .stDateInput > div > div { background-color: #1E1E1E; color: white; }
    
    footer {visibility: hidden;}
    .katex { color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADERS (CACHED) ---

@st.cache_resource
def load_model_assets():
    """Loads the trained DDN model and Scalers from disk."""
    try:
        MODEL_DIR = config.MODEL_DIR
        # Load Scalers
        sx = joblib.load(MODEL_DIR / "scaler_x.save")
        sy = joblib.load(MODEL_DIR / "scaler_y.save")
        
        # Initialize Model (Default Architecture used in backtest)
        with open(config.BEST_HPS_FILE, 'r') as f:
            hp = json.load(f)
        
        model = DeepDifferentialNetwork(
            num_hidden=hp['num_hidden'],
            neurons=hp['neurons'],
            dropout=hp['dropout'],
            activation=hp['activation']
        )
        
        # Run dummy forward pass to build graph
        model(tf.zeros((1, 9)))
        
        # Load Weights
        weights_path = MODEL_DIR / "ddn.weights.h5"
        model.load_weights(weights_path)
        return model, sx, sy
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None

@st.cache_data
def load_backtest_results():
    """Loads the CSV containing calibrated parameters for every day."""
    try:
        df = pd.read_csv(config.BACKTEST_OUTPUT_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').sort_index()
    except Exception:
        import traceback
        traceback.print_exc()
        return pd.DataFrame() 

@st.cache_data
def load_raw_option_data(target_date):
    """Loads raw option data for a specific date from the CSVs."""
    # This function remains unchanged, as it's used for the interactive pricing fit.
    print(f"Loading raw option data for {target_date}...")
    try:
        files = config.AAPL_OPTION_FILES
        dfs = []
        for f in files:
            print(f"Checking file: {f}")
            if Path(f).exists():
                print(f'Loading file: {f}')
                df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
                print(f'Loaded {len(df)} rows.')
                df.columns = df.columns.str.strip().str.replace(r'\[|\]', '', regex=True).str.strip()
                df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
                daily = df[df['QUOTE_DATE'] == target_date].copy()
                if not daily.empty:
                    dfs.append(daily)
                    break
            else:
                print(f"File {f} does not exist.")
        
        if not dfs: return pd.DataFrame()
        return pd.concat(dfs)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def get_aapl_volatility():
    """
    Retrieves the historical realized volatility of AAPL, caching the results.
    """
    # Ensure the cache directory exists
    cache_dir = Path(config.STOCK_CACHE_FILE).parent
    os.makedirs(cache_dir, exist_ok=True)
    
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
        except Exception as e:
            st.error(f"Failed to download stock data: {e}")
            return pd.DataFrame() # Return empty if download fails

    # Calculate Realized Volatility
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Realized_Vol'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
    df.index = df.index.tz_localize(None) # Make timezone-naive for joining
    return df[['Realized_Vol']].dropna()

def get_heatmap_data():
    """Returns aggregated error data matching the paper's findings."""
    maturities = ['< 3M', '3M-6M', '6M-1Yr', '> 1Yr']
    moneyness = ['< 0.95', '0.95-1.05', '> 1.05']
    
    call_data = [[4.4, 5.0, 12.7], [5.3, 4.5, 6.0], [4.2, 3.2, 5.0], [2.3, 2.9, 10.4]]
    put_data = [[28.1, 8.2, 1.0], [27.4, 4.5, 1.3], [24.5, 3.4, 1.6], [24.4, 4.4, 3.0]]
    return maturities, moneyness, call_data, put_data

def predict_real_prices(model, df, params, sx, sy):
    """Inference Logic."""
    N = len(df)
    p_tiled = np.tile(np.array(params), (N, 1))
    inputs = df[['r','q', 'tau', 'log_moneyness']].values
    full_x = np.hstack([p_tiled, inputs])
    scaled_x = full_x * sx.scale_ + sx.min_
    pred_scaled = model.predict(scaled_x, verbose=0)
    pred_norm = (pred_scaled.flatten() - sy.min_[0]) / sy.scale_[0]
    return pred_norm * df['S0'].values

# --- MAIN APP LOGIC ---

# Load and merge the two datasets
backtest_df = load_backtest_results()
vol_df = get_aapl_volatility()
df_ts = backtest_df.join(vol_df, how='left') # Join volatility onto the backtest results

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: white;'>Calibration of the <span class='highlight'>Heston Model</span><br>using Neural Networks</h1>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["1️⃣ The Methodology", "2️⃣ Results & Analysis", "3️⃣ Conclusion & Implications"])


# ==============================================================================
# TAB 1: THE METHODOLOGY
# ==============================================================================
with tab1:
    # --- PART 1: THE FOUNDATION ---
    with st.expander("🔍 The Foundation: Heston Stochastic Volatility", expanded=False):
        st.markdown("### 🔍 The Foundation: Heston Stochastic Volatility")
        
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.markdown("""
            <div class="red-box">
                <h4>Concept</h4>
                <ul>
                    <li><b>The Problem:</b> Black-Scholes assumes <span class="highlight">Constant Volatility</span> (fails to explain the Smile).</li>
                    <li><b>The Heston Solution:</b> Volatility ($v_t$) is treated as a <span class="highlight">random process</span>.</li>
                    <li><b>Key Feature:</b> Volatility is correlated with the Asset Price ($S_t$), capturing real market dynamics.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("**Heston Dynamics:**")
            st.latex(r'''
            \begin{aligned}
            dS_t &= r S_t dt + \sqrt{v_t} S_t dW_t^S \\
            dv_t &= \kappa(\lambda - v_t) dt + \sigma \sqrt{v_t} dW_t^v
            \end{aligned}
            ''')

    # --- PART 2: THE CHALLENGE ---
    with st.expander("⚠️ The Challenge: Traditional Calibration", expanded=False):
        st.markdown("### ⚠️ The Challenge: Traditional Calibration")
        
        st.markdown("""
        <p>To use the model, we must find the parameters <span class="highlight">Θ = {κ, λ, σ, ρ, v₀}</span> that minimize the error between Market Prices and Model Prices:</p>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        \Theta^* = \arg\min_{\Theta} \sum_{i=1}^{N} \left( C_{mkt}^i - C_{mod}^i(\Theta) \right)^2
        ''')
        
        col_prob, col_why = st.columns([1, 1], gap="medium")
        
        with col_prob:
            st.markdown("""
            <div class="red-box">
                <h4>Traditional Methods</h4>
                <p><b>1. Local Optimizers</b> (e.g., Levenberg-Marquardt)</p>
                <p><b>2. Global Optimizers</b> (e.g., Genetic Algorithms)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_why:
            st.markdown("""
            <div class="red-box">
                <h4>Why it is Slow <span style='font-size:20px'>🐢</span></h4>
                <p>1. <b>Gradient Calculation:</b> Calculating derivatives (∇Θ) requires solving the pricing engine 10x per step (Finite Differences).</p>
                <p>2. <b>Integration Cost:</b> Every pricing call requires solving a complex integral.</p>
            </div>
            """, unsafe_allow_html=True)

    # --- PART 3: THE SOLUTION ---
    with st.expander("🚀 The Solution: Deep Differential Networks", expanded=False):
        st.markdown("### 🚀 The Solution: Deep Differential Networks (DDN)")
        
        col_sol_text, col_sol_vis = st.columns([2, 1], gap="large")
        
        with col_sol_text:
            st.markdown("""
            <div class="red-box">
                <h4>The DDN Surrogate <span style='font-size:20px'>🚀</span></h4>
                <p>We replace the slow integral solver with a <b>Neural Network</b> trained using <span class="highlight">Sobolev Training</span>.</p>
                <ul>
                    <li><b>Instant Pricing:</b> Matrix multiplication replaces integration.</li>
                    <li><b>Instant Gradients:</b> We use Backpropagation to get Exact Gradients (Neural Greeks) instantly.</li>
                    <li><b>Result:</b> The optimizer runs millions of times faster.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_sol_vis:
            st.markdown("""
            <div style="background-color: #111; border: 1px solid #FF4B4B; padding: 15px; border-radius: 10px; text-align: center;">
                <span style="font-size: 40px;">🧠</span><br>
                <span class="highlight">DDN Model</span><br>
                ⬇️<br>
                Predicted Price<br>
                <span class="highlight">+</span><br>
                Predicted Gradients<br>
                (∇Θ)
            </div>
            """, unsafe_allow_html=True)
            
    # --- PART 4: RESEARCH GAP & MOTIVATION ---
    with st.expander("🔬 Research Gap & Motivation", expanded=False):
        st.markdown("""
        **The Gap:** Prior DDN research (Zhang et al., 2025) relied on static datasets, leaving real-world longitudinal performance untested.
        
        **Our Approach:** We conducted a rigorous **7-year backtest** (AAPL 2016-2023) to answer:
        """)
        st.markdown("""
        <div class="red-box">
            <h4>Key Research Questions</h4>
            <ul>
                <li><b>Longitudinal Robustness:</b> Does calibration survive <span class="highlight">diverse regimes</span> (Crashes, Low Vol, Hikes)?</li>
                <li><b>Generalization:</b> How does the model perform <span class="highlight">out-of-sample</span>?</li>
                <li><b>Error Dynamics:</b> How does <span class="highlight">market stress</span> impact the models performance?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: RESULTS
# ==============================================================================
with tab2:
    if df_ts.empty:
        st.warning("⚠️ Backtest results not found. Ensure `data/backtest_results.csv` exists and is correctly formatted.")
    else:
        with st.expander("📊 Backtest Results (2016-2023)", expanded=False):
            st.markdown("### Backtest Results (2016-2023)")
            
            fig_ts = go.Figure()

            # Plot 1: Realized Volatility as a background area chart
            fig_ts.add_trace(go.Scatter(
                x=df_ts.index, y=df_ts['Realized_Vol'],
                fill='tozeroy', mode='none', name='20-Day Realized Vol',
                fillcolor='rgba(255, 255, 255, 0.1)'
            ))
            
            # Plot 2: In-Sample MRE
            fig_ts.add_trace(go.Scatter(
                x=df_ts.index, y=df_ts['in_sample_mre'],
                mode='lines', name='In-Sample Error (MRE)',
                line=dict(color='grey', width=1.5)
            ))

            # Plot 3: Out-of-Sample MRE (highlighted)
            fig_ts.add_trace(go.Scatter(
                x=df_ts.index, y=df_ts['out_sample_mre'],
                mode='lines', name='Out-of-Sample Error (MRE)',
                line=dict(color='#FF4B4B', width=2)
            ))
            
            events = [
                {"date": "2018-02-05", "text": "Volmageddon"},
                {"date": "2020-01-13", "text": "COVID Crash"},
                {"date": "2022-03-16", "text": "Fed Hikes Start"}
            ]

            for event in events:
                # 1. Add the vertical line using add_shape
                fig_ts.add_shape(
                    type="line",
                    x0=pd.to_datetime(event["date"]), x1=pd.to_datetime(event["date"]),
                    y0=0, y1=1, yref="paper", # yref='paper' makes it span the full plot height
                    line=dict(color="gray", width=1, dash="dash")
                )
                # 2. Add the annotation text separately
                fig_ts.add_annotation(
                    x=pd.to_datetime(event["date"]),
                    y=0.95, yref="paper", # Position text near the top
                    text=event["text"],
                    showarrow=False,
                    yshift=10,
                    font=dict(color="white", size=12),
                    bgcolor="rgba(0,0,0,0.5)"
                )

            fig_ts.update_layout(
                template="plotly_dark", 
                title="Daily Calibration Error vs. Market Volatility (2016-2023)", 
                height=400, 
                plot_bgcolor="#111111", 
                paper_bgcolor="#111111",
                yaxis_title="MRE / Volatility",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            
            st.markdown(
                f"""
                #### Key Findings:
                *   **No Overfitting:** Out-of-Sample error (Red) tracks In-Sample error (Grey) almost perfectly.
                *   **Correlation:** Calibration error spikes align with stress events (e.g., <span class='highlight'>COVID-19</span>).
                *   **Mean Reversion:** Error is <span class='highlight'>bounded</span> (<15% max) and reverts quickly after crashes, proving robustness.
                """, unsafe_allow_html=True
            )
            
        # -------------------------------------------------------------------------
        # SECTION B: HEATMAPS & PARITY ANALYSIS
        # -------------------------------------------------------------------------
        with st.expander("🗺️ Error Analysis by Moneyness & Maturity", expanded=False):
            st.markdown("### Error Analysis by Moneyness & Maturity")
            
            rows, cols, calls, puts = get_heatmap_data()
            
            c1, c2 = st.columns(2)
            
            with c1:
                fig_call = px.imshow(calls, x=cols, y=rows, text_auto=True, color_continuous_scale="Reds", title="Call Option MRE (%)")
                fig_call.update_layout(template="plotly_dark", plot_bgcolor="#111111", paper_bgcolor="#111111")
                st.plotly_chart(fig_call, use_container_width=True)
                st.caption("Call errors highest for **OTM (>1.05)**. Low absolute prices = high % error.")

            with c2:
                fig_put = px.imshow(puts, x=cols, y=rows, text_auto=True, color_continuous_scale="Reds", title="Put Option MRE (%)")
                fig_put.update_layout(template="plotly_dark", plot_bgcolor="#111111", paper_bgcolor="#111111")
                st.plotly_chart(fig_put, use_container_width=True)
                st.caption("Put errors highest for **OTM (<0.95)** due to Parity propagation.")

            st.markdown("""
            <div class="red-box">
                <h4>Why the asymmetry? (Put-Call Parity)</h4>
                <ul>
                    <li>Puts are priced via Parity: <img src="https://latex.codecogs.com/svg.latex?\color{white}P_{model}=C_{model}^{DDN}-S_0e^{-qt}+Ke^{-rt}" style="vertical-align:middle;height:1.5em;" alt="Put-Call Parity with Dividends" /></li>
                    <li>An <b>OTM Put</b> corresponds to an <span class='highlight'>Expensive ITM Call</span>.</li>
                    <li>A small error on the expensive Call becomes a <b>massive relative error</b> on the cheap Put.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        # ==============================================================================
        # SECTION C: MODEL VALIDATION
        # ==============================================================================

        with st.expander("🔬 Model Validation: DDN vs. Analytical Pricer", expanded=False):
            st.markdown("""
            **The Test:** We fed DDN-calibrated parameters into an independent **Analytical Engine** (QuantLib) to measure pure approximation error.
            """)

            # --- 2. DATA EXTRACTION (TABLE 12 & 13) ---
            validation_data = {
                "Market Regime": ["Pre-Volmageddon", "Trade War", "COVID Crash", "Recovery", "Inflation", "Overall Average"],
                "Period": ["Jan 2016 - Jan 2018", "Feb 2018 - Jan 2020", "Feb 2020 - May 2020", "Jun 2020 - Dec 2021", "Jan 2022 - Dec 2023", "Jan 2016 - Dec 2023"],
                "DDN MRE (%)": [4.59, 4.47, 5.60, 4.63, 3.75, 4.48],
                "True Heston MRE (%)": [5.04, 5.06, 5.59, 5.01, 4.05, 4.87],
                "Difference (p.p.)": [0.45, 0.59, -0.01, 0.38, 0.30, 0.39]
            }
            
            df_validation = pd.DataFrame(validation_data).set_index("Market Regime")

            # --- 3. VISUAL STYLING & DISPLAY ---
            # Specific formatting for numeric columns only
            styled_df = df_validation.style.format({
                "DDN MRE (%)": "{:.2f}",
                "True Heston MRE (%)": "{:.2f}",
                "Difference (p.p.)": "{:.2f}"
            })

            st.dataframe(styled_df, use_container_width=True)

            st.caption("""
            **Result:** The average difference is just <span class='highlight'>0.39 p.p.</span> The DDN is a high-fidelity emulator that introduces negligible error.
            """, unsafe_allow_html=True)
            
        # -------------------------------------------------------------------------
        # SECTION D: REAL DATA SURFACE FIT
        # -------------------------------------------------------------------------
        with st.expander("📈 Pricing Fit (Real Data)", expanded=False):
            st.markdown("### Pricing Fit (Real Data)")
            
            res_df = df_ts # Reuse the already loaded dataframe
            model, sx, sy = load_model_assets()
            
            if res_df.empty or model is None:
                st.warning("⚠️ Real data or Model weights not found. Ensure 'data/backtest_results.csv' and 'models/' exist.")
            else:
                col_sel, col_plot = st.columns([1, 3])
                
                with col_sel:
                    st.markdown("#### Configuration")
                    min_date = res_df.index.min().date()
                    max_date = res_df.index.max().date()
                    default_date = pd.to_datetime("2020-06-15").date()
                    if default_date < min_date or default_date > max_date: default_date = min_date
                        
                    selected_date_input = st.date_input("Select Trading Day", value=default_date, min_value=min_date, max_value=max_date)
                    selected_date = pd.to_datetime(selected_date_input)
                    
                    if selected_date not in res_df.index:
                        idx = res_df.index.get_indexer([selected_date], method='nearest')[0]
                        target_date = res_df.index[idx]
                        st.info(f"Date missing. Snapped to: {target_date.date()}")
                    else:
                        target_date = selected_date

                    day_df = load_raw_option_data(target_date)
                    
                    if day_df.empty:
                        st.error("No raw CSV data for this date.")
                    else:
                        row = res_df.loc[target_date]
                        params = [row['kappa'], row['lambda'], row['sigma'], row['rho'], row['v0']]
                        
                        cols_to_fix = ['C_BID', 'C_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE']
                        for c in cols_to_fix: day_df[c] = pd.to_numeric(day_df[c], errors='coerce')
                        
                        day_df['S0'] = day_df['UNDERLYING_LAST']
                        day_df['K'] = day_df['STRIKE']
                        day_df['tau'] = day_df['DTE'] / 365.0
                        day_df['marketPrice'] = (day_df['C_BID'] + day_df['C_ASK']) / 2.0
                        day_df['r'] = row['avg_risk_free_rate']
                        day_df['q'] = row['implied_dividend_yield']
                        day_df['log_moneyness'] = np.log(day_df['K'] / day_df['S0'])
                        
                        day_df = day_df[(day_df['marketPrice'] > 0.50) & (day_df['tau'] > 0.02)].copy()
                        
                        available_taus = sorted(day_df['tau'].unique())
                        if not available_taus:
                            st.error("No liquid options found.")
                        else:
                            tau_map = {t: f"{int(t*365)} Days (T={t:.2f})" for t in available_taus}
                            default_idx = len(available_taus)//2
                            selected_tau_val = st.selectbox("Select Maturity", options=available_taus, format_func=lambda x: tau_map[x], index=default_idx)
                            
                            subset = day_df[np.isclose(day_df['tau'], selected_tau_val, atol=0.001)].sort_values('K')
                            
                            if not subset.empty:
                                with st.spinner("Calculating DDN Prices..."):
                                    model_prices = predict_real_prices(model, subset, params, sx, sy)
                                
                                with col_plot:
                                    fig_fit = go.Figure()
                                    fig_fit.add_trace(go.Scatter(
                                        x=subset['K'], y=subset['marketPrice'],
                                        mode='markers', name='Market Price',
                                        marker=dict(color='white', size=8, line=dict(width=1, color='gray'))
                                    ))
                                    fig_fit.add_trace(go.Scatter(
                                        x=subset['K'], y=model_prices,
                                        mode='lines', name='DDN Model',
                                        line=dict(color='#FF4B4B', width=3)
                                    ))
                                    
                                    spot_price = subset['S0'].iloc[0]
                                    fig_fit.add_vline(x=spot_price, line_dash="dash", line_color="gray", annotation_text="Spot")
                                    
                                    fig_fit.update_layout(
                                        template="plotly_dark",
                                        title=f"<b>Calibration Fit</b> | {target_date.date()} | Spot: ${spot_price:.2f}",
                                        xaxis_title="Strike Price ($)",
                                        yaxis_title="Option Price ($)",
                                        plot_bgcolor="#111111",
                                        paper_bgcolor="#111111",
                                        legend=dict(y=0.9, x=0.8),
                                        height=500
                                    )
                                    st.plotly_chart(fig_fit, use_container_width=True)
                                    
                                    mae = np.mean(np.abs(model_prices - subset['marketPrice']))
                                    st.caption(f"Mean Absolute Error (MAE): **${mae:.2f}**")

# ==============================================================================
# TAB 3: CONCLUSION & IMPLICATIONS
# ==============================================================================
with tab3:
    synthesis_data = [
        {
            "id": "1",
            "question": "Robustness across market regimes?",
            "finding_header": "Longitudinal Stability",
            "finding_body": """
            **Regime Resilience.** Calibration error remained <span class='highlight'>bounded</span> during the COVID-19 exogenous shock (Avg MRE: 5.59%) and exhibited rapid mean reversion post-crisis.
            """,
            "implication_header": "Operational Viability",
            "implication_body": """
            **Stable Deployment.** The framework demonstrates consistent performance across diverse market conditions, supporting its suitability for continuous historical analysis.
            """
        },
        {
            "id": "2",
            "question": "Generalization vs. Overfitting?",
            "finding_header": "Generalization Capability",
            "finding_body": """
            **Minimal Variance.** The negligible divergence between In-Sample (4.29%) and Out-of-Sample (4.49%) MRE indicates **minimal overfitting** to the training data.
            """,
            "implication_header": "Surface Approximation",
            "implication_body": """
            **Latent Feature Learning.** The results suggest the model effectively approximates the underlying volatility surface topology rather than memorizing specific contract prices.
            """
        },
        {
            "id": "3",
            "question": "Error Dynamics?",
            "finding_header": "Error Determinants",
            "finding_body": """
            **Volatility Correlation.** Calibration residuals exhibit a positive correlation with realized market volatility, notably during stress events.
            """,
            "implication_header": "Performance Monitoring",
            "implication_body": """
            **Systematic Behavior.** The deterministic nature of the error degradation under stress facilitates the establishment of quantifiable confidence intervals.
            """
        }
    ]

    # --- Loop through the data to render each section dynamically ---
    for item in synthesis_data:
        with st.expander(f"🔍 RQ {item['id']}: {item['question']}", expanded=False):
            st.markdown(
                f"#### <span class='highlight'>RQ {item['id']}: {item['question']}</span>", 
                unsafe_allow_html=True
            )

            col_finding, col_implication = st.columns(2, gap="medium")

            with col_finding:
                st.markdown(
                    f"##### <span class='highlight'>{item['finding_header']}</span>", 
                    unsafe_allow_html=True
                )
                st.markdown(item['finding_body'], unsafe_allow_html=True)

            with col_implication:
                st.markdown(
                    f"##### <span class='highlight'>{item['implication_header']}</span>", 
                    unsafe_allow_html=True
                )
                st.markdown(item['implication_body'], unsafe_allow_html=True)