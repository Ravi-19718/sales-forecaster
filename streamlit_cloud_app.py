import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sales Forecaster",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
</style>
""", unsafe_allow_html=True)

# ── LSTM Model definition ─────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Load model & data ─────────────────────────────────
@st.cache_resource
def load_model():
    model = LSTMForecaster()
    model.load_state_dict(torch.load('models/lstm_model.pth', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_data():
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/daily_sales.pkl', 'rb') as f:
        daily_sales = pickle.load(f)
    return scaler, daily_sales

model                = load_model()
scaler, daily_sales  = load_data()

# ── Header ────────────────────────────────────────────
st.title("📈 Sales Forecaster")
st.markdown("**LSTM Deep Learning** · Grocery Sales Prediction · Store Item Demand Dataset")
st.divider()

# ── Sidebar ───────────────────────────────────────────
st.sidebar.header("Forecast Settings")
days_ahead    = st.sidebar.slider("Days to Forecast",        min_value=7,  max_value=90,  value=30, step=7)
history_days  = st.sidebar.slider("Historical Days to Show", min_value=30, max_value=365, value=90)

st.sidebar.divider()
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("LSTM MAE",     "17,714")
st.sidebar.metric("Baseline MAE", "40,403")
st.sidebar.metric("Improvement",  "56.2%", delta="vs 7-day moving avg")

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

# ── Generate forecast ─────────────────────────────────
def generate_forecast(days):
    last_30    = daily_sales['sales'].values[-30:]
    sequence   = scaler.transform(last_30.reshape(-1, 1))
    current_seq = sequence.copy()
    predictions = []

    for _ in range(days):
        x = torch.FloatTensor(current_seq).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(x).numpy()
        pred_value = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(round(float(pred_value), 2))
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = pred_scaled[0][0]

    last_date  = daily_sales['date'].max()
    dates      = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
    return pd.DataFrame({'date': dates, 'predicted_sales': predictions})

# ── Chart ─────────────────────────────────────────────
col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.subheader("Forecast Chart")
    with st.spinner("Generating forecast..."):
        forecast_df = generate_forecast(days_ahead)
        hist        = daily_sales.tail(history_days).copy()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist['date'], y=hist['sales'],
            name='Historical Sales',
            line=dict(color='#4f8ef7', width=2),
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df['date'], y=forecast_df['predicted_sales'],
            name='LSTM Forecast',
            line=dict(color='#f97b4f', width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=5)
        ))

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

with col_table:
    st.subheader("Forecast Values")
    display_df = forecast_df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%b %d, %Y')
    display_df['predicted_sales'] = display_df['predicted_sales'].apply(lambda x: f"{x:,.0f}")
    display_df.columns = ['Date', 'Predicted Sales']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:13px;'>"
    "Built with PyTorch · Streamlit · Plotly · Store Sales Dataset (Kaggle)"
    "</div>",
    unsafe_allow_html=True
)