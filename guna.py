import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load stock data from Excel file
file_path = "stock_prices.xlsx"  # Ensure the file is in the working directory
df = pd.read_excel(file_path, engine="openpyxl")

# Convert date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Extract closing price
data = df[["Close"]].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Prepare training data
X_train, y_train = [], []
for i in range(60, len(data_scaled)):
    X_train.append(data_scaled[i-60:i, 0])
    y_train.append(data_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, batch_size=1, epochs=5)

# Predict future stock price
future_data = data_scaled[-60:].reshape(1, 60, 1)
predicted_price = model.predict(future_data)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"Predicted stock price: ${predicted_price[0][0]:.2f}")

# Plot actual vs predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Actual Stock Price", color="blue")
plt.axhline(y=predicted_price[0][0], color="red", linestyle="--", label="Predicted Price")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("AI-driven Stock Price Prediction using Time Series Analysis")
plt.legend()
plt.grid()
plt.show()
