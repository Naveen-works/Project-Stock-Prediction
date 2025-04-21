import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(page_title="📈 Stock Trend Predictor", layout="wide")


st.markdown("""
    <style>
        .block-container {
            padding: 2rem 2rem 2rem 2rem;
            max-width: 900px;
            margin: auto;
        }
        h1, h2, h3 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='color: #00FFAB;'>🔮 Futuristic Stock Trend Predictor</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #00FFAB;'>", unsafe_allow_html=True)


user_input = st.text_input('🚀 Enter Stock Ticker', 'AAPL')


start = '2010-01-01'
end = '2019-12-31'


df = yf.download(user_input, start=start, end=end)


st.subheader('📊 Stock Data Summary (2010–2019)')
st.dataframe(df.describe(), use_container_width=True)


st.subheader('📉 Closing Price vs Time')
fig = plt.figure(figsize=(7, 3), dpi=100)
plt.style.use("dark_background")
plt.plot(df['Close'], color='#00FFAB')
plt.title('Closing Price Over Time', fontsize=12)
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)
st.pyplot(fig, use_container_width=False)


st.subheader('📈 Closing Price with 100-Day Moving Average')
ma100 = df['Close'].rolling(100).mean()
fig1 = plt.figure(figsize=(7, 3), dpi=100)
plt.plot(df['Close'], label='Close', color='#00C4FF', alpha=0.7)
plt.plot(ma100, label='100 MA', color='#FF00A8')
plt.title('100-Day MA vs Closing Price', fontsize=12)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
st.pyplot(fig1, use_container_width=False)


st.subheader('📊 100-Day & 200-Day Moving Averages')
ma200 = df['Close'].rolling(200).mean()
fig2 = plt.figure(figsize=(7, 3), dpi=100)
plt.plot(df['Close'], label='Close', color='#00C4FF', alpha=0.7)
plt.plot(ma100, label='100 MA', color='#FF00A8')
plt.plot(ma200, label='200 MA', color='#FFD700')
plt.title('100MA & 200MA Trend', fontsize=12)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
st.pyplot(fig2, use_container_width=False)


close_prices = df['Close']
split_index = int(len(close_prices) * 0.70)
data_training = pd.DataFrame(close_prices[:split_index])
data_testing = pd.DataFrame(close_prices[split_index:])


scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)


x_train, y_train = [], []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)


model = load_model('keras_model.h5')


past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)


y_predicted = model.predict(x_test)
scaling_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scaling_factor
y_test = y_test * scaling_factor


st.subheader('🤖 Predicted vs Actual Stock Prices')
fig3 = plt.figure(figsize=(7, 3), dpi=100)
plt.plot(y_test, label='Actual Price', color='#00FFAB')
plt.plot(y_predicted, label='Predicted Price', color='#FF005C')
plt.title('Model Prediction vs Actual', fontsize=12)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True, alpha=0.3)
st.pyplot(fig3, use_container_width=False)


st.markdown("<hr style='border: 1px solid #00FFAB;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>⚡ Powered by LSTM | Built with ❤ using Streamlit</p>", unsafe_allow_html=True)
