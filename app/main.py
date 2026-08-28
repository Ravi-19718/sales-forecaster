from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd

# ── LSTM Model definition (must match train.py) ───────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Load model & scaler ───────────────────────────────
model = LSTMForecaster()
model.load_state_dict(torch.load('models/lstm_model.pth', map_location='cpu'))
model.eval()

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/daily_sales.pkl', 'rb') as f:
    daily_sales = pickle.load(f)

app = FastAPI(title="Sales Forecaster API", version="1.0")

class ForecastRequest(BaseModel):
    days_ahead: int = 7  # how many days to forecast

@app.get("/")
def root():
    return {"message": "Sales Forecaster API is running", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    # Use the last 30 days of known data as the seed sequence
    last_30 = daily_sales['sales'].values[-30:]
    sequence = scaler.transform(last_30.reshape(-1, 1))

    predictions = []
    current_seq = sequence.copy()

    for _ in range(request.days_ahead):
        x = torch.FloatTensor(current_seq).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(x).numpy()
        pred_value = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(round(float(pred_value), 2))
        # Roll the window forward
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = pred_scaled[0][0]

    # Build date labels starting from the day after last known date
    last_date = daily_sales['date'].max()
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                          periods=request.days_ahead, freq='D')
    dates_str = [str(d.date()) for d in dates]

    return {
        "forecast_days": request.days_ahead,
        "predictions": [{"date": d, "predicted_sales": p}
                        for d, p in zip(dates_str, predictions)]
    }