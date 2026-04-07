import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

import nltk

# Download the sentiment model if not already available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')





# Set up the app title and description
st.title('Stock Price Prediction using LSTM')
st.write("This app uses an LSTM model to predict stock prices based on historical data.")

# Allow users to enter a stock ticker
stock_ticker = st.text_input("Enter Stock Ticker", "AAPL")

# Fetch data using yfinance
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2020-01-01")
    return data

df = load_data(stock_ticker)

# Show the raw data and plot closing price
st.subheader(f"Data for {stock_ticker}")
st.write(df.tail())

st.subheader(f"Closing Price of {stock_ticker}")
plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.title(f'{stock_ticker} Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
st.pyplot(plt)

# Prepare the data for LSTM
# Prepare the data for LSTM with checks
if 'Close' not in df.columns:
    st.error("The 'Close' column is missing in the downloaded data.")
    st.stop()

data = df[['Close']].dropna()

if data.empty:
    st.error("The 'Close' column has no valid data to train the model.")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# Split data into training and testing
training_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len - 60:]

# Create X_train, y_train
X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Prepare X_test
X_test = []
y_test = scaled_data[training_data_len:]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict the stock prices
y_predicted = model.predict(X_test)
y_predicted = scaler.inverse_transform(y_predicted)

# Inverse scale the y_test for comparison
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
st.subheader(f"Predicted vs Original Prices for {stock_ticker}")
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot(plt)


# ---- Sentiment Analysis Section ----
st.subheader("📈 News Sentiment Analysis")

# Step 1: Sample Headlines (Replace with real API later)
news_headlines = [
    f"{stock_ticker} stock hits new highs as investors remain optimistic.",
    f"{stock_ticker} faces setbacks due to regulatory issues.",
    f"Experts suggest {stock_ticker} is a strong buy this quarter.",
    f"{stock_ticker} continues to struggle with market volatility.",
    f"Positive trends observed in {stock_ticker}'s recent earnings report."
]

# Step 2: Analyze Sentiment
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = []

for headline in news_headlines:
    score = analyzer.polarity_scores(headline)
    sentiment_scores.append(score['compound'])  # compound = overall score

# Step 3: Display Headline Sentiment
for i, headline in enumerate(news_headlines):
    st.write(f"📰 {headline}")
    st.write(f"Sentiment Score: `{sentiment_scores[i]:.2f}`")
    st.markdown("---")

# Step 4: Show Average Sentiment
average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

if average_sentiment >= 0.05:
    sentiment_summary = "Positive 😊"
elif average_sentiment <= -0.05:
    sentiment_summary = "Negative 😟"
else:
    sentiment_summary = "Neutral 😐"

st.markdown(f"### 🧠 Overall Sentiment: **{sentiment_summary}**")
st.markdown(f"Average Sentiment Score: `{average_sentiment:.2f}`")

# Step 5: Show a simple sentiment chart
st.subheader("🗂 Sentiment Scores Chart")
fig, ax = plt.subplots()
ax.bar(range(len(sentiment_scores)), sentiment_scores, color='skyblue')
ax.set_xticks(range(len(sentiment_scores)))
ax.set_xticklabels([f"News {i+1}" for i in range(len(sentiment_scores))], rotation=45)
ax.axhline(0, color='gray', linestyle='--')
st.pyplot(fig)


import nltk

# Download the sentiment model if not already available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')


