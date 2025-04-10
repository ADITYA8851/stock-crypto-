import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model_utils import load_model, preprocess_data, predict_next_move

st.set_page_config(page_title="Stock & Crypto Predictor", layout="wide")
st.title("📈 Stock & Crypto Price Movement Predictor")

option = st.radio("Choose input method:", ["Upload CSV", "Enter Symbol (AAPL, BTC-USD, etc.)"])

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload historical OHLC data", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    symbol = st.text_input("Enter stock/crypto symbol (e.g., AAPL, BTC-USD)", value="BTC-USD")
    period = st.selectbox("Select historical period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    if symbol:
        df = yf.download(symbol, period=period)
        df.reset_index(inplace=True)

if 'df' in locals() and not df.empty:
    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'])])
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Prediction
    try:
        model = load_model()
        X, y, processed_df = preprocess_data(df.copy())
        prediction = predict_next_move(model, X)
        movement = "🔼 UP" if prediction[-1] == 1 else "🔽 DOWN"
        st.success(f"Predicted Next Move: {movement}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    st.subheader("Close Price Bar Chart")
    st.bar_chart(df.set_index("Date")["Close"])