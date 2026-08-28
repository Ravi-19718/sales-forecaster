import streamlit as st
import requests
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sales Forecaster",
    page_icon="📈",
    layout="wide"
)

# ── Custom styling ────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #4f8ef7; }
    .metric-label { font-size: 13px; color: #8b9ab5; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────
st.title("📈 Sales Forecaster")
st.markdown("**LSTM Deep Learning** · Grocery Sales Prediction · Store Item Demand Dataset")
st.divider()

# ── Load historical data ──────────────────────────────
@st.cache_resource
def load_data():
    with open('models/daily_sales.pkl', 'rb') as f:
        return pickle.load(f)

daily_sales = load_data()

# ── Sidebar controls ──────────────────────────────────
st.sidebar.header("Forecast Settings")
days_ahead = st.sidebar.slider("Days to Forecast", min_value=7, max_value=90, value=30, step=7)
history_days = st.sidebar.slider("Historical Days to Show", min_value=30, max_value=365, value=90)

st.sidebar.divider()
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("LSTM MAE",      "17,714")
st.sidebar.metric("Baseline MAE",  "40,403")
st.sidebar.metric("Improvement",   "56.2%", delta="vs 7-day moving avg")

# ── Metrics row ───────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", "LSTM (2 layers)")
with col2:
    st.metric("Training Data", "1,564 days")
with col3:
    st.metric("MAE Improvement", "56.2%")
with col4:
    st.metric("Sequence Length", "30 days")

st.divider()

# ── Get forecast from API ─────────────────────────────
col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.subheader("Forecast Chart")
    with st.spinner("Fetching forecast..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/forecast",
                json={"days_ahead": days_ahead}
            )
            forecast_data = response.json()["predictions"]
            forecast_df = pd.DataFrame(forecast_data)
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])

            # Historical slice
            hist = daily_sales.tail(history_days).copy()

            # Build Plotly chart
            fig = go.Figure()

            # Historical line
            fig.add_trace(go.Scatter(
                x=hist['date'], y=hist['sales'],
                name='Historical Sales',
                line=dict(color='#4f8ef7', width=2),
                mode='lines'
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['predicted_sales'],
                name='LSTM Forecast',
                line=dict(color='#f97b4f', width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=5)
            ))

            # Shaded forecast region
            fig.add_vrect(
                x0=forecast_df['date'].min(),
                x1=forecast_df['date'].max(),
                fillcolor='rgba(249,123,79,0.08)',
                line_width=0
            )

            fig.update_layout(
                paper_bgcolor='#0f1117',
                plot_bgcolor='#0f1117',
                font=dict(color='#c9d1e0'),
                legend=dict(bgcolor='#1e2130', bordercolor='#2d3250', borderwidth=1),
                xaxis=dict(gridcolor='#1e2130', showgrid=True),
                yaxis=dict(gridcolor='#1e2130', showgrid=True, title='Units Sold'),
                hovermode='x unified',
                margin=dict(l=20, r=20, t=30, b=20),
                height=420
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not connect to API: {e}")
            st.info("Make sure the FastAPI server is running: python3 -m uvicorn app.main:app --reload")

with col_table:
    st.subheader("Forecast Values")
    if 'forecast_df' in dir():
        display_df = forecast_df.copy()
        display_df['date'] = display_df['date'].dt.strftime('%b %d, %Y')
        display_df['predicted_sales'] = display_df['predicted_sales'].apply(lambda x: f"{x:,.0f}")
        display_df.columns = ['Date', 'Predicted Sales']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:13px;'>"
    "Built with PyTorch · FastAPI · Streamlit · Plotly · Store Sales Dataset (Kaggle)"
    "</div>",
    unsafe_allow_html=True
)