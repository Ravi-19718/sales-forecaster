# 📈 Sales Forecaster — LSTM Deep Learning Time-Series Forecast

A deep learning time-series forecasting system built with **PyTorch LSTM** that predicts retail grocery sales 7–90 days into the future. Achieves **56.2% lower MAE** than a 7-day moving average baseline. Deployed live on Streamlit Cloud with an interactive dashboard.

---

## 🚀 Live Demo

👉 **[Launch App on Streamlit Cloud](https://share.streamlit.io/Ravi-19718/sales-forecaster)**

---

## 🎯 Project Overview

| Metric | Value |
|---|---|
| Dataset | Kaggle Store Item Demand Forecasting |
| Target | Daily GROCERY I family sales |
| Training period | Jan 2013 – Aug 2017 (1,684 days) |
| Test set | Last 90 days |
| Baseline MAE | 40,403 (7-day moving average) |
| LSTM MAE | **17,713** |
| Improvement | **56.2% reduction in MAE** |

---

## 🧠 Model Architecture

```
Input: 30-day rolling window of normalized sales
  ↓
LSTM Layer 1 (hidden_size=64, dropout=0.2)
  ↓
LSTM Layer 2 (hidden_size=64, dropout=0.2)
  ↓
Fully Connected Layer → 1 value (next day prediction)
  ↓
Iterative rolling forecast for N days ahead
```

**Key design choices:**
- **Sequence length:** 30 days (captures monthly seasonality)
- **MinMaxScaler:** Normalizes sales before feeding to LSTM
- **Rolling window inference:** Each predicted day feeds back into the sequence for multi-step forecasting
- **Epochs:** 50 | **Batch size:** 32 | **LR:** 0.001 (Adam optimizer)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 2.x, LSTM |
| Data Processing | Pandas, NumPy, scikit-learn |
| API | FastAPI (local deployment) |
| Dashboard | Streamlit |
| Visualization | Plotly (interactive charts) |
| Dataset | Kaggle Store Item Demand |

---

## 📊 Dashboard Features

- **Interactive forecast chart** — blue historical line + orange dashed forecast
- **Adjustable horizon** — slide from 7 to 90 days ahead
- **Adjustable history window** — show last 30 to 365 days of context
- **Live metrics** — model info, training data range, MAE improvement, sequence length
- **Forecast table** — date + predicted sales for each day

---

## 🗂️ Project Structure

```
sales-forecaster/
├── train.py                  # LSTM training script
├── streamlit_cloud_app.py    # Streamlit Cloud standalone app
├── streamlit_app.py          # Local app (calls FastAPI)
├── app/
│   └── main.py               # FastAPI REST API
├── models/
│   ├── lstm_model.pth        # Trained LSTM weights
│   ├── scaler.pkl            # MinMaxScaler
│   └── daily_sales.pkl       # Aggregated daily sales data
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/Ravi-19718/sales-forecaster.git
cd sales-forecaster
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install "numpy<2"   # required for PyTorch 2.x compatibility
```

### 3. Run the Streamlit app (standalone — no API needed)
```bash
streamlit run streamlit_cloud_app.py
```

### 4. (Optional) Run with FastAPI backend
```bash
# Terminal 1 — start API
uvicorn app.main:app --reload

# Terminal 2 — start UI
streamlit run streamlit_app.py
```

### 5. (Optional) Retrain the model
Download the dataset from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) and place `train.csv` in `data/`, then:
```bash
python train.py
```

---

## 📈 Results

```
Baseline (7-day MA) MAE : 40,403
LSTM MAE               : 17,713
Improvement            : 56.2% ✅
```

The LSTM significantly outperforms the moving average by learning non-linear temporal patterns in daily grocery sales volume.

---

## 🤝 Connect

Built as part of an ML/Data Science portfolio.  
[LinkedIn](https://linkedin.com/in/josephravi) | [GitHub](https://github.com/Ravi-19718)
