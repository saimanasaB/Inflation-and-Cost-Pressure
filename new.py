import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit App Title
st.title('General Index Forecasting using LSTM and SARIMA')

# Load the dataset
file_path = st.text_input('Enter the file path for the dataset:', 'C:\Users\Sai Manasa\Downloads\cleaned_data.csv')
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
st.subheader('General Index Over Time')
base_chart = alt.Chart(data.reset_index()).mark_line().encode(
    x='Date:T',
    y='General index:Q'
).properties(
    width=700,
    height=400
).interactive()
st.altair_chart(base_chart)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Creating the dataset for LSTM
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

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
st.subheader('Training LSTM Model...')
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

st.subheader('Model Evaluation Metrics')
st.write(f"SARIMA - RMSE: {rmse_sarima}, MAE: {mae_sarima}")
st.write(f"LSTM - RMSE: {rmse_lstm}, MAE: {mae_lstm}")

# Prepare data for plotting SARIMA and LSTM forecasts
forecast_data_sarima = pd.DataFrame({
    'Date': forecast_index_sarima,
    'Forecasted General Index (SARIMA)': forecast_mean_sarima
})

forecast_data_lstm = pd.DataFrame({
    'Date': future_dates_lstm,
    'Forecasted General Index (LSTM)': future_predictions_lstm_inv.flatten()
})

# Merge the datasets
forecast_data = pd.merge_asof(forecast_data_sarima, forecast_data_lstm, on='Date')

# Plotting the results using Altair
st.subheader('SARIMA vs LSTM Forecasts')
actual_chart = alt.Chart(data.reset_index()).mark_line(color='red').encode(
    x='Date:T',
    y='General index:Q',
    tooltip=['Date:T', 'General index:Q']
).properties(
    width=700,
    height=400
)

sarima_chart = alt.Chart(forecast_data).mark_line(color='blue').encode(
    x='Date:T',
    y='Forecasted General Index (SARIMA):Q',
    tooltip=['Date:T', 'Forecasted General Index (SARIMA):Q']
)

lstm_chart = alt.Chart(forecast_data).mark_line(color='green').encode(
    x='Date:T',
    y='Forecasted General Index (LSTM):Q',
    tooltip=['Date:T', 'Forecasted General Index (LSTM):Q']
)

final_chart = actual_chart + sarima_chart + lstm_chart
st.altair_chart(final_chart)
