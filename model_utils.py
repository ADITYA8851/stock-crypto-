import joblib
import pandas as pd

def load_model():
    return joblib.load("model.pkl")

def preprocess_data(df):
    df['Price_Change'] = df['Close'] - df['Open']
    df['Percent_Change'] = (df['Price_Change'] / df['Open']) * 100
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    features = ['Percent_Change', 'MA5', 'MA10', 'Volume_Change']
    X = df[features]
    y = df['Target']
    return X, y, df

def predict_next_move(model, X):
    return model.predict(X)