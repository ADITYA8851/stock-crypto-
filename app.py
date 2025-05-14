import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fast Finance Predictor", layout="wide")
st.title("🚀 FINANCE PREDICTION APP")

# Selectable options
stock_options = ["AAPL", "GOOGL", "BTC-USD", "^BSESN", "ADANIENT.NS", "^NSEI"]
stocks = st.multiselect("Select stocks to analyze", stock_options, default=stock_options[:2])
period = st.selectbox("Select historical period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
max_rows = st.slider("Maximum rows to analyze", 100, 2000, 300)

# Cache stock data
@st.cache_data(ttl=3600)
def load_stock_data(symbol, period):
    return yf.download(symbol, period=period)

for symbol in stocks:
    st.header(f"📊 Analysis for {symbol}")
    with st.spinner(f"Fetching and processing data for {symbol}..."):
        df = load_stock_data(symbol, period)
        df.reset_index(inplace=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]

        # Rename standard columns
        standard_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in standard_cols:
            for df_col in df.columns:
                if col.lower() in df_col.lower() and col not in df.columns:
                    df.rename(columns={df_col: col}, inplace=True)

        if df.empty:
            st.warning(f"No data found for {symbol}")
            continue

        required_columns = {"Open", "Close", "Volume", "High", "Low"}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            st.warning(f"{symbol} is missing required columns: {missing_cols}. Skipping...")
            continue

        df = df.dropna(subset=list(required_columns))
        df = df.tail(max_rows)

        # Feature Engineering
        df['Price_Change'] = df['Close'] - df['Open']
        df['Percent_Change'] = (df['Price_Change'] / df['Open'].replace(0, np.nan)) * 100
        df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Low'].replace(0, np.nan)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)

        # Cleaned Features
        features = ['Percent_Change', 'MA5', 'MA10', 'Volume_Change', 'High_Low_Spread']
        X = df[features].replace([np.inf, -np.inf], np.nan).dropna()
        y = df['Target'].loc[X.index]

        if X.shape[0] < 10:
            st.warning(f"Not enough clean data to analyze {symbol}.")
            continue

        # Fast XGBoost model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=4, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        next_prediction = model.predict(X.tail(1))
        movement = "🔼 UP" if next_prediction[0] == 1 else "🔽 DOWN"

    # Candlestick Chart
    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    fig.update_layout(title=f"Candlestick Chart - {symbol}", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Histogram and Line Chart
    st.subheader("Close Price Distribution & Trend")
    col1, col2 = st.columns(2)

    with col1:
        fig_hist, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df['Close'], bins=30, kde=True, ax=ax)
        ax.set_title("Histogram of Close Prices")
        st.pyplot(fig_hist)

    with col2:
        fig_line, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(df['Date'], df['Close'], color='blue')
        ax2.set_title("Close Price Trend")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        plt.xticks(rotation=45)
        st.pyplot(fig_line)

    # Prediction result
    st.subheader("Prediction Results")
    st.success(f"Predicted Next Move for {symbol}: {movement}")
    st.info(f"Model Accuracy: {accuracy * 100:.2f}%")
