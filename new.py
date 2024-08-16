import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
file_path = 'C:\\Users\\Sai Manasa\\Downloads\\cleaned_data.csv'  # Update with your actual file path
data = pd.read_csv(file_path)

# Check the data structure
print(data.head())

# Select the relevant features
data = data[['Year', 'Month', 'General index']]

# Convert Year and Month into a datetime format
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

# Sort by date
data = data.sort_values(by='Date').reset_index(drop=True)

# Drop Year and Month as they are now redundant
data = data.drop(columns=['Year', 'Month'])

# Set Date as index
data.set_index('Date', inplace=True)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Preparing the dataset for the LSTM model
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Create the dataset with a time step of 12 (i.e., 1 year)
time_step = 12
X, Y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features] for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]
# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
# Predicting the Test set results
Y_pred = model.predict(X_test)

# Inverse transform the scaled data
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_pred_inv = scaler.inverse_transform(Y_pred)

# Calculate precision, recall, F1 score, and accuracy
def calculate_metrics(Y_test_inv, Y_pred_inv):
    # Convert to binary predictions for metrics calculation
    threshold = 0.5
    Y_test_bin = np.where(Y_test_inv > threshold, 1, 0)
    Y_pred_bin = np.where(Y_pred_inv > threshold, 1, 0)

    accuracy = accuracy_score(Y_test_bin, Y_pred_bin)
    precision = precision_score(Y_test_bin, Y_pred_bin)
    recall = recall_score(Y_test_bin, Y_pred_bin)
    f1 = f1_score(Y_test_bin, Y_pred_bin)

    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = calculate_metrics(Y_test_inv, Y_pred_inv)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
# Predicting for the next 5 years (60 months)
future_predictions = []

current_input = X_test[-1].reshape(1, time_step, 1)
for _ in range(60):
    future_pred = model.predict(current_input)
    future_predictions.append(future_pred[0, 0])
    current_input = np.append(current_input[:, 1:, :], future_pred.reshape(1, 1, 1), axis=1)

# Inverse transform the future predictions
future_predictions_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Print future predictions
print(future_predictions_inv)
import matplotlib.pyplot as plt

# Plotting the predicted vs actual values for the test set
plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(Y_test):], Y_test_inv, color='blue', label='Actual General Index')
plt.plot(data.index[-len(Y_test):], Y_pred_inv, color='red', label='Predicted General Index')
plt.title('Test Set: Actual vs Predicted General Index')
plt.xlabel('Date')
plt.ylabel('General Index')
plt.legend()
plt.show()
# Creating a date range for the next 5 years (60 months)
last_date = data.index[-1]
future_dates = pd.date_range(last_date, periods=61, freq='M')[1:]

# Plotting the future predictions
plt.figure(figsize=(14, 7))
plt.plot(future_dates, future_predictions_inv, color='green', label='Predicted General Index for Next 5 Years')
plt.title('Predicted General Index for the Next 5 Years')
plt.xlabel('Date')
plt.ylabel('General Index')
plt.legend()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
file_path = 'C:\\Users\\Sai Manasa\\Downloads\\cleaned_data.csv'  # Update with your actual file path
data = pd.read_csv(file_path)

# Select the relevant features
data = data[['Year', 'Month', 'General index']]

# Convert Year and Month into a datetime format
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

# Sort by date
data = data.sort_values(by='Date').reset_index(drop=True)

# Drop Year and Month as they are now redundant
data = data.drop(columns=['Year', 'Month'])

# Set Date as index
data.set_index('Date', inplace=True)

# Plot the General Index to understand its trend
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['General index'], label='General Index')
plt.title('General Index Over Time')
plt.xlabel('Date')
plt.ylabel('General Index')
plt.legend()
plt.grid(True)
plt.show()
# Define the SARIMA model
# You might need to adjust (p, d, q) and (P, D, Q, s) based on your data
sarima_model = SARIMAX(data['General index'], 
                       order=(1, 1, 1),  # ARIMA parameters (p, d, q)
                       seasonal_order=(1, 1, 1, 12),  # Seasonal parameters (P, D, Q, s)
                       enforce_stationarity=False,
                       enforce_invertibility=False)

# Fit the model
sarima_results = sarima_model.fit(disp=False)

# Summary of the model
print(sarima_results.summary())
# Forecasting the next 60 months (5 years)
forecast_steps = 60
forecast = sarima_results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Inverse transform the forecast if scaled (not needed here as it's not scaled)
# future_predictions_inv = scaler.inverse_transform(np.array(forecast_mean).reshape(-1, 1))

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['General index'], label='Actual General Index')
plt.plot(forecast_index, forecast_mean, color='green', marker='o', linestyle='--', label='Forecasted General Index')
plt.fill_between(forecast_index, 
                 forecast_conf_int.iloc[:, 0], 
                 forecast_conf_int.iloc[:, 1], 
                 color='green', alpha=0.3)
plt.title('SARIMA Forecast vs Actual General Index')
plt.xlabel('Date')
plt.ylabel('General Index')
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE and MAE for the forecasted values (compared to actual future values)
# Note: You will need actual future values to calculate these metrics
# For demonstration, using dummy future values
dummy_future_actual = np.random.rand(forecast_steps)  # Replace with actual future values
rmse = np.sqrt(mean_squared_error(dummy_future_actual, forecast_mean))
mae = mean_absolute_error(dummy_future_actual, forecast_mean)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
file_path = 'C:\\Users\\Sai Manasa\\Downloads\\cleaned_data.csv'  # Update with your actual file path
data = pd.read_csv(file_path)

# Select the relevant features
data = data[['Year', 'Month', 'General index']]

# Convert Year and Month into a datetime format
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

# Sort by date
data = data.sort_values(by='Date').reset_index(drop=True)

# Drop Year and Month as they are now redundant
data = data.drop(columns=['Year', 'Month'])

# Set Date as index
data.set_index('Date', inplace=True)

# Plot the General Index to understand its trend
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['General index'], label='General Index')
plt.title('General Index Over Time')
plt.xlabel('Date')
plt.ylabel('General Index')
plt.legend()
plt.grid(True)
plt.show()
from sklearn.model_selection import train_test_split

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Creating the dataset
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 12
X, Y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features] for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
# Predicting the next 60 months (5 years) using LSTM
forecast_steps = 60
future_predictions_lstm = []

current_input_lstm = X_test[-1].reshape(1, time_step, 1)
for _ in range(forecast_steps):
    future_pred_lstm = model.predict(current_input_lstm)
    future_predictions_lstm.append(future_pred_lstm[0, 0])
    current_input_lstm = np.append(current_input_lstm[:, 1:, :], future_pred_lstm.reshape(1, 1, 1), axis=1)

future_dates_lstm = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
future_predictions_lstm_inv = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1))
# Define the SARIMA model
sarima_model = SARIMAX(data['General index'], 
                       order=(1, 1, 1),  # ARIMA parameters (p, d, q)
                       seasonal_order=(1, 1, 1, 12),  # Seasonal parameters (P, D, Q, s)
                       enforce_stationarity=False,
                       enforce_invertibility=False)

# Fit the model
sarima_results = sarima_model.fit(disp=False)
# Forecasting the next 60 months (5 years) using SARIMA
forecast_sarima = sarima_results.get_forecast(steps=forecast_steps)
forecast_index_sarima = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
forecast_mean_sarima = forecast_sarima.predicted_mean
forecast_conf_int_sarima = forecast_sarima.conf_int()
# Dummy future actual values for comparison (Replace with actual future values if available)
dummy_future_actual = np.random.rand(forecast_steps)  # Replace with actual future values

# Evaluate SARIMA
rmse_sarima = np.sqrt(mean_squared_error(dummy_future_actual, forecast_mean_sarima))
mae_sarima = mean_absolute_error(dummy_future_actual, forecast_mean_sarima)

# Evaluate LSTM
rmse_lstm = np.sqrt(mean_squared_error(dummy_future_actual, future_predictions_lstm_inv))
mae_lstm = mean_absolute_error(dummy_future_actual, future_predictions_lstm_inv)

print(f"SARIMA - RMSE: {rmse_sarima}, MAE: {mae_sarima}")
print(f"LSTM - RMSE: {rmse_lstm}, MAE: {mae_lstm}")
# Plot SARIMA and LSTM forecasts
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['General index'], label='Actual General Index')
plt.plot(forecast_index_sarima, forecast_mean_sarima, color='blue', linestyle='--', label='SARIMA Forecast')
plt.plot(future_dates_lstm, future_predictions_lstm_inv, color='green', marker='o', linestyle='--', label='LSTM Forecast')
plt.fill_between(forecast_index_sarima, 
                 forecast_conf_int_sarima.iloc[:, 0], 
                 forecast_conf_int_sarima.iloc[:, 1], 
                 color='blue', alpha=0.3)
plt.title('SARIMA vs LSTM Forecasts')
plt.xlabel('Date')
plt.ylabel('General Index')
plt.legend()
plt.grid(True)
plt.show()







