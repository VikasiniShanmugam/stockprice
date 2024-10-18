import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Title of the app
st.title("Stock Price Prediction App")

# Sidebar inputs for user
ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
period = st.sidebar.selectbox('Select time period for prediction', ['1y', '2y', '5y'])

# Download stock data from Yahoo Finance
@st.cache_data
def load_data(ticker, period):
    stock_data = yf.download(ticker, period=period)
    return stock_data

data = load_data(ticker, period)

# Display raw stock data in the app
st.subheader('Raw Stock Data')
st.write(data.tail())

# Prepare the data for machine learning (we'll use just the Date and Close price for simplicity)
data['Date'] = data.index
data = data[['Date', 'Close']]

# Feature engineering: Convert date to numerical values for regression
data['Date'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)

# Split data into features (X) and target (y)
X = np.array(data['Date']).reshape(-1, 1)
y = data['Close'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict stock prices
y_pred = model.predict(X_test)

# Show the actual vs predicted results
st.subheader("Actual vs Predicted Stock Prices")
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
plt.xlabel('Date (ordinal)')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)

# Predict the future stock price (we'll simply extend the date range)
days_in_future = 30
future_dates = np.array([data['Date'].max() + i for i in range(1, days_in_future+1)]).reshape(-1, 1)
future_pred = model.predict(future_dates)

# Visualizing the prediction for the next 30 days
st.subheader(f"Next {days_in_future} Days Stock Price Prediction")
plt.figure(figsize=(10,5))
plt.plot(future_dates, future_pred, color='green', label='Predicted Future Prices')
plt.xlabel('Date (ordinal)')
plt.ylabel('Predicted Stock Price')
plt.legend()
st.pyplot(plt)
