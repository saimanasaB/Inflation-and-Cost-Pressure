import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit App Title
st.title('General Index Forecasting using LSTM and SARIMA')

# Load the dataset
file_path = st.text_input('cleaned_data.csv')
data = pd.read_csv('cleaned_data.csv')

# Display the DataFrame
st.write("Data Preview:")
st.dataframe(data)

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
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
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

future_dates_lstm = pd.date_range(start='2025-01-01', end='2030-12-01', freq='M')
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
forecast_index_sarima = pd.date_range(start='2025-01-01', end='2030-12-01', freq='M')
forecast_mean_sarima = forecast_sarima.predicted_mean
forecast_conf_int_sarima = forecast_sarima.conf_int()

# Handle length mismatch if needed
if len(forecast_mean_sarima) != len(forecast_index_sarima):
    st.error(f"Length mismatch: SARIMA forecast data length ({len(forecast_mean_sarima)}) does not match forecast index length ({len(forecast_index_sarima)})")
    st.stop()

if len(future_predictions_lstm_inv) != len(future_dates_lstm):
    st.error(f"Length mismatch: LSTM forecast data length ({len(future_predictions_lstm_inv)}) does not match forecast index length ({len(future_dates_lstm)})")
    st.stop()

# Prepare data for SARIMA plot
forecast_data_sarima = pd.DataFrame({
    'Date': forecast_index_sarima,
    'Forecasted General Index (SARIMA)': forecast_mean_sarima
})

# Separate Plotting for SARIMA
st.subheader('SARIMA Forecast')
sarima_chart = alt.Chart(forecast_data_sarima).mark_line(color='blue').encode(
    x='Date:T',
    y='Forecasted General Index (SARIMA):Q',
    tooltip=['Date:T', 'Forecasted General Index (SARIMA):Q']
).properties(
    width=700,
    height=400
)
st.altair_chart(sarima_chart)

# Prepare data for LSTM plot
forecast_data_lstm = pd.DataFrame({
    'Date': future_dates_lstm,
    'Forecasted General Index (LSTM)': future_predictions_lstm_inv.flatten()
})

# Separate Plotting for LSTM
st.subheader('LSTM Forecast')
lstm_chart = alt.Chart(forecast_data_lstm).mark_line(color='green').encode(
    x='Date:T',
    y='Forecasted General Index (LSTM):Q',
    tooltip=['Date:T', 'Forecasted General Index (LSTM):Q']
).properties(
    width=700,
    height=400
)
st.altair_chart(lstm_chart)

# Comparison Plot
st.subheader('Comparison of Forecasts')
comparison_data = pd.DataFrame({
    'Date': forecast_index_sarima,
    'SARIMA Forecast': forecast_mean_sarima,
    'LSTM Forecast': np.concatenate([
        np.full(len(forecast_index_sarima) - len(future_predictions_lstm_inv), np.nan),
        future_predictions_lstm_inv.flatten()
    ])
})

comparison_chart = alt.Chart(comparison_data).mark_line().encode(
    x='Date:T',
    y=alt.Y('value:Q', title='Forecasted General Index'),
    color='variable:N',
    tooltip=['Date:T', 'value:Q', 'variable:N']
).transform_fold(
    ['SARIMA Forecast', 'LSTM Forecast'],
    as_=['variable', 'value']
).properties(
    width=700,
    height=400
).interactive()
st.altair_chart(comparison_chart)

# Ensure the plots and metrics are displayed properly
st.subheader('Forecast Data')
st.write("Forecasted General Index using SARIMA:")
st.dataframe(forecast_data_sarima)

st.write("Forecasted General Index using LSTM:")
st.dataframe(forecast_data_lstm)
