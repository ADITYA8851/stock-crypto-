import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Finance Prediction App", layout="wide")
st.title("📈 Finance Prediction App")

# Select stocks
stock_options = ["AAPL", "GOOGL", "BTC-USD"]
stocks = st.multiselect("Select stocks to analyze", stock_options, default=stock_options[:2])
period = st.selectbox("Select historical period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

for symbol in stocks:
    st.header(f"Analysis for {symbol}")
    df = yf.download(symbol, period=period)
    df.reset_index(inplace=True)

    # Robust flattening of MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]

    # Attempt to rename columns to standard names
    standard_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in standard_cols:
        for df_col in df.columns:
            if col.lower() in df_col.lower() and col not in df.columns:
                df.rename(columns={df_col: col}, inplace=True)

    if df.empty:
        st.warning(f"No data found for {symbol}")
        continue

    # Safe column check
    required_columns = {"Open", "Close", "Volume", "High", "Low"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        st.warning(f"{symbol} is missing required columns: {missing_cols}. Skipping...")
        continue
    df = df.dropna(subset=list(required_columns))

    # Feature Engineering
    df['Price_Change'] = df['Close'] - df['Open']
    df['Percent_Change'] = (df['Price_Change'] / df['Open'].replace(0, np.nan)) * 100
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    features = ['Percent_Change', 'MA5', 'MA10', 'Volume_Change']
    X = df[features]
    y = df['Target']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # Predict next move
    next_prediction = model.predict(X.tail(1))
    movement = "🔼 UP" if next_prediction[0] == 1 else "🔽 DOWN"

    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'])])
    fig.update_layout(title=f"Candlestick Chart - {symbol}", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Close Price Histogram")
    fig_hist, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(df['Close'], bins=30, kde=True, ax=ax)
    ax.set_title("Close Price Distribution")
    st.pyplot(fig_hist)

    st.subheader("Prediction Results")
    st.success(f"Predicted Next Move for {symbol}: {movement}")
    st.info(f"Model Accuracy: {accuracy * 100:.2f}%")
