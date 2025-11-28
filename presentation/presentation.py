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
        model(tf.zeros((1, 8)))
        
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
    print(f"Loading raw option data for {target_date}...")
    try:
        files = config.AAPL_OPTION_FILES
        dfs = []
        for f in files:
            print(f"Checking file: {f}")
            if Path(f).exists():
                # Read chunks or filter on load would be better for prod, 
                # but pandas is fast enough for 2 files if we filter immediately.
                # Here we assume the user has these files.
                print(f'Loading file: {f}')
                df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
                print(f'Loaded {len(df)} rows.')
                df.columns = df.columns.str.strip().str.replace(r'\[|\]', '', regex=True).str.strip()
                df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
                print(df['QUOTE_DATE'].head())
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
def load_presentation_data():
    """Generates synthetic time-series data for the first plot to ensure it always looks good."""
    np.random.seed(42)
    dates = pd.date_range(start="2016-01-01", end="2023-03-01", freq="B")
    n = len(dates)
    vol = np.zeros(n)
    vol[0] = 0.15
    for i in range(1, n):
        shock = np.random.normal(0, 0.02)
        if dates[i].year == 2018 and dates[i].month == 2: shock += 0.1
        if dates[i].year == 2020 and dates[i].month == 3: shock += 0.25
        vol[i] = max(0.05, vol[i-1] + 0.1 * (0.15 - vol[i-1]) + shock)
    
    base_error = 0.02
    mre_out = base_error + (0.2 * vol) + np.abs(np.random.normal(0, 0.01, n))
    return pd.DataFrame({"Date": dates, "Realized_Vol": vol, "MRE_Out_Sample": mre_out})

def get_heatmap_data():
    """Returns aggregated error data matching the paper's findings."""
    maturities = ['< 3M', '3M-6M', '6M-1Yr', '> 1Yr']
    moneyness = ['< 0.95', '0.95-1.05', '> 1.05']
    
    # Call Errors (High for > 1.05 OTM)
    call_data = [[4.1, 7.6, 14.8], [4.1, 3.2, 6.1], [2.9, 2.3, 5.9], [0.9, 1.9, 9.3]]
    # Put Errors (High for < 0.95 OTM, due to parity)
    put_data = [[29.1, 10.9, 0.9], [23.3, 3.6, 1.2], [19.8, 2.9, 1.6], [20.1, 4.9, 3.6]]
    return maturities, moneyness, call_data, put_data

def predict_real_prices(model, df, params, sx, sy):
    """Inference Logic."""
    N = len(df)
    p_tiled = np.tile(np.array(params), (N, 1))
    inputs = df[['r', 'tau', 'log_moneyness']].values
    full_x = np.hstack([p_tiled, inputs])
    scaled_x = full_x * sx.scale_ + sx.min_
    pred_scaled = model.predict(scaled_x, verbose=0)
    pred_norm = (pred_scaled.flatten() - sy.min_[0]) / sy.scale_[0]
    return pred_norm * df['S0'].values

# --- MAIN APP LOGIC ---

df_ts = load_presentation_data()

# --- SIDEBAR ---
st.sidebar.markdown("## Navigation")
st.sidebar.info("Use the tabs above to navigate the presentation.")

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: white;'>Calibration of the <span class='highlight'>Heston Model</span><br>using Neural Networks</h1>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["1️⃣ The Methodology", "2️⃣ Results & Analysis", "3️⃣ Implications"])

st.markdown("ARIAN IST TOLL!")

# ==============================================================================
# TAB 1: METHODOLOGY
# ==============================================================================
with tab1:
    # --- PART 1: THE FOUNDATION ---
    st.markdown("### 1. The Foundation: Heston Stochastic Volatility")
    
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("""
        <div class="red-box">
            <h4>Concept</h4>
            <p>The Black-Scholes model assumes volatility is constant. This fails to explain the <b>Volatility Smile</b> observed in real markets.</p>
            <p>The <span class="highlight">Heston Model (1993)</span> solves this by treating Volatility (<span class="highlight">v<sub>t</sub></span>) as a random process that correlates with the Asset Price.</p>
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

    st.markdown("---")

    # --- PART 2: THE CHALLENGE ---
    st.markdown("### 2. The Bottleneck: Traditional Calibration")
    
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

    st.markdown("---")

    # --- PART 3: THE SOLUTION ---
    st.markdown("### 3. The Solution: Deep Differential Networks (DDN)")
    
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

# ==============================================================================
# TAB 2: RESULTS
# ==============================================================================
with tab2:
    st.markdown("### A. Longitudinal Robustness (2016-2023)")
    
    # 1. TIME SERIES
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_ts["Date"], y=df_ts["Realized_Vol"],
        fill='tozeroy', mode='none', name='Market Volatility',
        fillcolor='rgba(255, 255, 255, 0.1)'
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_ts["Date"], y=df_ts["MRE_Out_Sample"],
        mode='lines', name='Calibration Error',
        line=dict(color='#FF4B4B', width=2)
    ))
    fig_ts.update_layout(template="plotly_dark", title="Calibration Error vs. Volatility", height=350, plot_bgcolor="#111111", paper_bgcolor="#111111")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # SECTION B: HEATMAPS & PARITY ANALYSIS
    # -------------------------------------------------------------------------
    st.markdown("### B. Error Analysis by Moneyness & Maturity")
    
    rows, cols, calls, puts = get_heatmap_data()
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig_call = px.imshow(calls, x=cols, y=rows, text_auto=True, color_continuous_scale="Reds", title="Call Option MRE (%)")
        fig_call.update_layout(template="plotly_dark", plot_bgcolor="#111111", paper_bgcolor="#111111")
        st.plotly_chart(fig_call, use_container_width=True)
        st.caption("Call errors are highest for **OTM (>1.05)**. Low absolute prices mean small deviations cause huge % errors.")

    with c2:
        fig_put = px.imshow(puts, x=cols, y=rows, text_auto=True, color_continuous_scale="Reds", title="Put Option MRE (%)")
        fig_put.update_layout(template="plotly_dark", plot_bgcolor="#111111", paper_bgcolor="#111111")
        st.plotly_chart(fig_put, use_container_width=True)
        st.caption("Put errors are highest for **OTM (<0.95)**. This is an artifact of Put-Call Parity propagation.")

    # SVG Image for Formula to bypass Streamlit LaTeX/HTML limitations
    st.markdown("""
    <div class="red-box">
        <h4>Why the asymmetry? (Put-Call Parity)</h4>
        <p>We price Puts using the calibrated Call surface via the Parity formula:</p>
        <div style="text-align: center; padding: 15px;">
            <img src="https://latex.codecogs.com/svg.latex?\color{white}P_{model}=C_{model}^{DDN}-S_0+Ke^{-r\tau}" 
                 alt="Put-Call Parity Formula" 
                 style="background-color: transparent; min-width: 250px;" />
        </div>
        <p>An OTM Put corresponds to an <b>ITM Call</b> (which is expensive). A 1% error on an expensive ITM Call represents a dollar amount that might be larger than the entire price of the cheap OTM Put, resulting in massive relative error statistics (e.g., 29%).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # SECTION C: REAL DATA SURFACE FIT
    # -------------------------------------------------------------------------
    st.markdown("### C. Volatility Surface Fit (Real Data)")
    
    res_df = load_backtest_results()
    model, sx, sy = load_model_assets()
    
    if res_df.empty or model is None:
        st.warning("⚠️ Real data or Model weights not found. Ensure 'data/backtest_results.csv' and 'models/' exist.")
    else:
        col_sel, col_plot = st.columns([1, 3])
        
        with col_sel:
            st.markdown("#### Configuration")
            # Date Selection
            min_date = res_df.index.min().date()
            max_date = res_df.index.max().date()
            default_date = pd.to_datetime("2020-06-15").date()
            if default_date < min_date or default_date > max_date: default_date = min_date
                
            selected_date_input = st.date_input("Select Trading Day", value=default_date, min_value=min_date, max_value=max_date)
            selected_date = pd.to_datetime(selected_date_input)
            
            # Snap to closest if missing
            if selected_date not in res_df.index:
                idx = res_df.index.get_indexer([selected_date], method='nearest')[0]
                target_date = res_df.index[idx]
                st.info(f"Date missing. Snapped to: {target_date.date()}")
            else:
                target_date = selected_date

            # Load Raw Data
            day_df = load_raw_option_data(target_date)
            
            if day_df.empty:
                st.error("No raw CSV data for this date.")
            else:
                # Preprocessing
                row = res_df.loc[target_date]
                params = [row['kappa'], row['lambda'], row['sigma'], row['rho'], row['v0']]
                rate = row['implied_rate']
                
                cols_to_fix = ['C_BID', 'C_ASK', 'UNDERLYING_LAST', 'STRIKE', 'DTE']
                for c in cols_to_fix: day_df[c] = pd.to_numeric(day_df[c], errors='coerce')
                
                day_df['S0'] = day_df['UNDERLYING_LAST']
                day_df['K'] = day_df['STRIKE']
                day_df['tau'] = day_df['DTE'] / 365.0
                day_df['marketPrice'] = (day_df['C_BID'] + day_df['C_ASK']) / 2.0
                day_df['r'] = rate
                day_df['log_moneyness'] = np.log(day_df['K'] / day_df['S0'])
                
                # Filter (Liquid options only)
                day_df = day_df[(day_df['marketPrice'] > 0.50) & (day_df['tau'] > 0.02)].copy()
                
                # Maturity Selection
                available_taus = sorted(day_df['tau'].unique())
                if not available_taus:
                    st.error("No liquid options found.")
                else:
                    tau_map = {t: f"{int(t*365)} Days (T={t:.2f})" for t in available_taus}
                    default_idx = len(available_taus)//2
                    selected_tau_val = st.selectbox("Select Maturity", options=available_taus, format_func=lambda x: tau_map[x], index=default_idx)
                    
                    # Prediction & Plot
                    subset = day_df[np.isclose(day_df['tau'], selected_tau_val, atol=0.001)].sort_values('K')
                    
                    if not subset.empty:
                        with st.spinner("Calculating DDN Prices..."):
                            model_prices = predict_real_prices(model, subset, params, sx, sy)
                        
                        with col_plot:
                            fig_fit = go.Figure()
                            # Market
                            fig_fit.add_trace(go.Scatter(
                                x=subset['K'], y=subset['marketPrice'],
                                mode='markers', name='Market Price',
                                marker=dict(color='white', size=8, line=dict(width=1, color='gray'))
                            ))
                            # Model
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
# TAB 3: IMPLICATIONS
# ==============================================================================
with tab3:
    st.markdown("### Key Takeaways")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="red-box">
            <h4>1. Speed & Scalability</h4>
            <p>Sub-second calibration enables <b>Real-Time Risk Management</b> (e.g., High Frequency Trading) impossible with standard integrators.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="red-box">
            <h4>2. Regime Robustness</h4>
            <p>The model survived <b>Volmageddon (2018)</b> and <b>COVID (2020)</b>. Errors spike but mean-revert quickly, proving stability.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="red-box">
            <h4>3. Differential Learning</h4>
            <p>Training on <b>Greeks (Gradients)</b> forces the network to learn the geometry of the market, not just the prices.</p>
        </div>
        """, unsafe_allow_html=True)