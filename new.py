import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Streamlit app setup
st.title("General Index Forecasting App")
st.write("This app forecasts the General Index using ARIMA and LSTM models up to the year 2034.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Convert 'Year' and 'Month' columns to a datetime object and set as index
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
    df.set_index('Date', inplace=True)

    # Keep only the 'General index' column
    df = df[['General index']]

    # Display the first few rows
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Calculate the number of months to forecast till the end of 2034
    last_date = df.index[-1]
    forecast_years = 2034 - last_date.year + (12 - last_date.month + 1) / 12  # Ensure full years and months
    months_to_forecast = int(forecast_years * 12)

    # ARIMA Model
    st.subheader("ARIMA Model Forecast")
    model_arima = ARIMA(df['General index'], order=(5, 1, 0))  # You can adjust (p, d, q) as needed
    arima_result = model_arima.fit()
    forecast_arima = arima_result.get_forecast(steps=months_to_forecast)
    forecast_index = pd.date_range(start=df.index[-1], periods=months_to_forecast + 1, freq='M')[1:]
    forecast_arima_df = pd.DataFrame(forecast_arima.predicted_mean, index=forecast_index, columns=['ARIMA Forecast'])

    # Plot ARIMA forecast
    st.write("### ARIMA Forecast Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['General index'], label='Observed')
    ax.plot(forecast_arima_df, label='ARIMA Forecast', color='red')
    ax.legend()
    ax.set_title('ARIMA Forecast of General Index')
    st.pyplot(fig)

    # LSTM Model
    st.subheader("LSTM Model Forecast")

    # Scale the data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    # Prepare the data for LSTM
    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step):
            a = data[i:(i + time_step), 0]
            X.append(a)
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 12  # Number of months in the time step
    X, Y = create_dataset(df_scaled, time_step)

    # Reshape for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build the LSTM model
    model_lstm = tf.keras.Sequential()
    model_lstm.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model_lstm.add(tf.keras.layers.LSTM(50, return_sequences=False))
    model_lstm.add(tf.keras.layers.Dense(25))
    model_lstm.add(tf.keras.layers.Dense(1))

    # Compile the model
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model_lstm.fit(X, Y, batch_size=1, epochs=10)

    # Forecasting with LSTM
    x_input = df_scaled[-time_step:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    i = 0
    while i < months_to_forecast:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, time_step, 1))
            yhat = model_lstm.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, time_step, 1))
            yhat = model_lstm.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    # Inverse transform to get actual values
    forecast_lstm = scaler.inverse_transform(lst_output)
    forecast_lstm_df = pd.DataFrame(forecast_lstm, index=forecast_index, columns=['LSTM Forecast'])

    # Plot LSTM forecast
    st.write("### LSTM Forecast Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['General index'], label='Observed')
    ax.plot(forecast_lstm_df, label='LSTM Forecast', color='orange')
    ax.legend()
    ax.set_title('LSTM Forecast of General Index')
    st.pyplot(fig)

    # Combine both forecasts
    combined_forecast_df = pd.concat([forecast_arima_df, forecast_lstm_df], axis=1)

    # Plot to compare
    st.write("### Comparison of ARIMA and LSTM Forecasts")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['General index'], label='Observed')
    ax.plot(combined_forecast_df['ARIMA Forecast'], label='ARIMA Forecast', color='red')
    ax.plot(combined_forecast_df['LSTM Forecast'], label='LSTM Forecast', color='orange')
    ax.legend()
    ax.set_title('Comparison of ARIMA and LSTM Forecasts')
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")

