import yfinance as yf
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

tickers = ["RELIANCE.NS", "HAL.NS", "SBIN.NS", "TATAMOTORS.NS", "CUPID.NS", "^NSEI"]
frames = []

for t in tickers:
    df = yf.download(t, period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]].dropna()
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["SMA_Ratio"] = df["Close"] / df["Close"].rolling(20).mean()
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    frames.append(df.dropna())

data = pd.concat(frames)
features = ["Return_1d", "Return_5d", "SMA_Ratio", "Volatility_10d"]
X, y = data[features], data["Target"]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)
acc = round(accuracy_score(y, model.predict(X)) * 100, 2)

joblib.dump(model, "stock_model.pkl")
print(f"Trained Stock Trend Model (Accuracy: {acc}%) & saved stock_model.pkl")
