import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────
SEQUENCE_LENGTH = 30   # use last 30 days to predict next day
EPOCHS         = 50
BATCH_SIZE     = 32
HIDDEN_SIZE    = 64
NUM_LAYERS     = 2
LEARNING_RATE  = 0.001

# ── Load & prepare data ───────────────────────────────
print("Loading data...")
train_df = pd.read_csv('data/train.csv', parse_dates=['date'])

# Focus on one family (GROCERY I) across all stores — clean and representative
df = train_df[train_df['family'] == 'GROCERY I'].copy()
daily_sales = df.groupby('date')['sales'].sum().reset_index()
daily_sales = daily_sales.sort_values('date').reset_index(drop=True)

print(f"Date range: {daily_sales['date'].min()} to {daily_sales['date'].max()}")
print(f"Total days: {len(daily_sales)}")

# ── Scale data ────────────────────────────────────────
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(daily_sales[['sales']])

# ── Create sequences ──────────────────────────────────
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(sales_scaled, SEQUENCE_LENGTH)

# Train/test split — last 90 days as test
split = len(X) - 90
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test  = torch.FloatTensor(y_test)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# ── LSTM Model ────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Baseline model (moving average) ──────────────────
print("\n--- Baseline: 7-day Moving Average ---")
y_test_actual = scaler.inverse_transform(y_test.numpy())
baseline_preds = []
for i in range(len(X_test)):
    window = scaler.inverse_transform(X_test[i].numpy())[-7:]
    baseline_preds.append(window.mean())
baseline_preds = np.array(baseline_preds).reshape(-1, 1)
baseline_mae  = mean_absolute_error(y_test_actual, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test_actual, baseline_preds))
print(f"Baseline MAE:  {baseline_mae:,.2f}")
print(f"Baseline RMSE: {baseline_rmse:,.2f}")

# ── Train LSTM ────────────────────────────────────────
print("\n--- Training LSTM ---")
model     = LSTMForecaster()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(len(X_train))
    epoch_loss  = 0
    batches     = 0
    for i in range(0, len(X_train), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        X_batch, y_batch = X_train[indices], y_train[indices]
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batches    += 1
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {epoch_loss/batches:.6f}")

# ── Evaluate LSTM ─────────────────────────────────────
print("\n--- LSTM Results ---")
model.eval()
with torch.no_grad():
    lstm_preds_scaled = model(X_test).numpy()
lstm_preds = scaler.inverse_transform(lstm_preds_scaled)
lstm_mae   = mean_absolute_error(y_test_actual, lstm_preds)
lstm_rmse  = np.sqrt(mean_squared_error(y_test_actual, lstm_preds))
print(f"LSTM MAE:  {lstm_mae:,.2f}")
print(f"LSTM RMSE: {lstm_rmse:,.2f}")
improvement = ((baseline_mae - lstm_mae) / baseline_mae) * 100
print(f"Improvement over baseline: {improvement:.1f}%")

# ── Save model & scaler ───────────────────────────────
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/lstm_model.pth')
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/daily_sales.pkl', 'wb') as f:
    pickle.dump(daily_sales, f)

print("\n✅ Model saved to models/lstm_model.pth")
print(f"\nSummary:")
print(f"  Baseline MAE:  {baseline_mae:,.2f}")
print(f"  LSTM MAE:      {lstm_mae:,.2f}")
print(f"  Improvement:   {improvement:.1f}%")