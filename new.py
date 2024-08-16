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
file_path = st.text_input('Enter file path of cleaned data (e.g., cleaned_data.csv)', 'cleaned_data.csv')
data = pd.read_csv(file_path)

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

# Convert predictions to binary (using a threshold)
threshold = 0.5
lstm_binary_preds = (future_predictions_lstm_inv.flatten() >= threshold).astype(int)
sarima_binary_preds = (forecast_mean_sarima >= threshold).astype(int)
dummy_binary_actual = (dummy_future_actual >= threshold).astype(int)

# Evaluate SARIMA
precision_sarima = precision_score(dummy_binary_actual, sarima_binary_preds)
recall_sarima = recall_score(dummy_binary_actual, sarima_binary_preds)
f1_sarima = f1_score(dummy_binary_actual, sarima_binary_preds)
accuracy_sarima = accuracy_score(dummy_binary_actual, sarima_binary_preds)
mse_sarima = mean_squared_error(dummy_future_actual, forecast_mean_sarima)
rmse_sarima = np.sqrt(mse_sarima)

# Evaluate LSTM
precision_lstm = precision_score(dummy_binary_actual, lstm_binary_preds)
recall_lstm = recall_score(dummy_binary_actual, lstm_binary_preds)
f1_lstm = f1_score(dummy_binary_actual, lstm_binary_preds)
accuracy_lstm = accuracy_score(dummy_binary_actual, lstm_binary_preds)
mse_lstm = mean_squared_error(dummy_future_actual, future_predictions_lstm_inv.flatten())
rmse_lstm = np.sqrt(mse_lstm)

st.subheader('Model Evaluation Metrics')
st.write(f"SARIMA - Precision: {precision_sarima}, Recall: {recall_sarima}, F1 Score: {f1_sarima}, Accuracy: {accuracy_sarima}, MSE: {mse_sarima}, RMSE: {rmse_sarima}")
st.write(f"LSTM - Precision: {precision_lstm}, Recall: {recall_lstm}, F1 Score: {f1_lstm}, Accuracy: {accuracy_lstm}, MSE: {mse_lstm}, RMSE: {rmse_lstm}")

# Prepare data for plotting SARIMA and LSTM forecasts
forecast_data_sarima = pd.DataFrame({
    'Date': forecast_index_sarima,
    'Year': forecast_index_sarima.year,
    'Forecasted General Index (SARIMA)': forecast_mean_sarima
})

forecast_data_lstm = pd.DataFrame({
    'Date': future_dates_lstm,
    'Year': future_dates_lstm.year,
    'Forecasted General Index (LSTM)': future_predictions_lstm_inv.flatten()
})

# Separate Plotting for SARIMA
st.subheader('SARIMA Forecast')
sarima_chart = alt.Chart(forecast_data_sarima).mark_line(color='blue').encode(
    x=alt.X('Year:O', title='Year'),
    y='Forecasted General Index (SARIMA):Q',
    tooltip=['Year:O', 'Forecasted General Index (SARIMA):Q']
).properties(
    width=700,
    height=400
)
st.altair_chart(sarima_chart)

# Separate Plotting for LSTM
st.subheader('LSTM Forecast')
lstm_chart = alt.Chart(forecast_data_lstm).mark_line(color='green').encode(
    x=alt.X('Year:O', title='Year'),
    y='Forecasted General Index (LSTM):Q',
    tooltip=['Year:O', 'Forecasted General Index (LSTM):Q']
).properties(
    width=700,
    height=400
)
st.altair_chart(lstm_chart)

# Comparison of forecasts
comparison_data = pd.concat([
    forecast_data_sarima[['Year', 'Forecasted General Index (SARIMA)']].rename(columns={'Forecasted General Index (SARIMA)': 'Forecast', 'Year': 'Year'}).assign(Model='SARIMA'),
    forecast_data_lstm[['Year', 'Forecasted General Index (LSTM)']].rename(columns={'Forecasted General Index (LSTM)': 'Forecast', 'Year': 'Year'}).assign(Model='LSTM')
])

comparison_chart = alt.Chart(comparison_data).mark_line().encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Forecast:Q', title='Forecasted General Index'),
    color='Model:N',
    tooltip=['Year:O', 'Model:N', 'Forecast:Q']
).properties(
    width=700,
    height=400
)
st.altair_chart(comparison_chart)

# Ensure the plots and metrics are displayed properly
st.subheader('Forecast Data')
st.write("Forecasted General Index using SARIMA:")
st.dataframe(forecast_data_sarima)

st.write("Forecasted General Index using LSTM:")
st.dataframe(forecast_data_lstm)
