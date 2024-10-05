import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd

# Constants for date range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title of the app
st.title("Stock Prediction App")

# Stock options
stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

# Number of years to predict
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Fetching stock data
data = yf.download(selected_stocks, START, TODAY)

# Display raw data
st.subheader("Raw Data")
st.write(data)

# Preparing data for Prophet
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data = data[['Date', 'Close']]
data.columns = ['ds', 'y']  # Renaming columns to fit Prophet's requirements

# Initialize Prophet model
model = Prophet()
model.fit(data)

# Future predictions
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Display forecast data
st.subheader('Forecast Data')
st.write(forecast)

# Plotting the forecast
st.subheader('Forecast Plot')
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)

# Display forecast components
st.subheader('Forecast Components')
fig2 = model.plot_components(forecast)
st.write(fig2)

# Display time series data
st.subheader('Time Series Data')
st.write(data)

# Displaying forecast data in a table
st.subheader("Detailed Forecast Data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Backtesting (Simple Implementation)
if st.button("Run Backtesting"):
    # Calculate MAE and RMSE for the last 'n' days
    last_n_days = 30  # Example
    actual = data['y'].values[-last_n_days:]
    predicted = forecast['yhat'].values[-last_n_days:]
    mae = abs(actual - predicted).mean()
    rmse = ((actual - predicted) ** 2).mean() ** 0.5
    st.success(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Download option for forecast data
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
st.download_button(
    label="Download Forecast Data as CSV",
    data=csv,
    file_name='forecast_data.csv',
    mime='text/csv',
)
