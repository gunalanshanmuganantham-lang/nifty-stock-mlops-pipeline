import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

tickers = ["RELIANCE.NS", "HDFCBANK.NS", "HAL.NS", "SBIN.NS", "TATAMOTORS.NS", "CUPID.NS", "INFY.NS", "ITC.NS", "^NSEI"]
frames = []

for t in tickers:
    df = yf.download(t, period="3y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "High", "Low", "Volume"]].dropna()
    df["Ret_1d"] = df["Close"].pct_change()
    df["Ret_5d"] = df["Close"].pct_change(5)
    df["SMA_Ratio"] = df["Close"] / df["Close"].rolling(20).mean()
    df["EMA_Ratio"] = df["Close"] / df["Close"].ewm(span=12).mean()
    df["Vol_10d"] = df["Ret_1d"].rolling(10).std()
    df["Range_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    frames.append(df.dropna())

data = pd.concat(frames)
features = ["Ret_1d", "Ret_5d", "SMA_Ratio", "EMA_Ratio", "Vol_10d", "Range_Pct", "Vol_Ratio"]
X, y = data[features], data["Target"]

rf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
lr = make_pipeline(StandardScaler(), LogisticRegression())

ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb), ("lr", lr)], voting="soft")
ensemble.fit(X, y)

rf.fit(X, y)
gb.fit(X, y)

meta = {
    "model": ensemble,
    "features": features,
    "metrics": {
        "ensemble_acc": round(accuracy_score(y, ensemble.predict(X)) * 100, 2),
        "rf_acc": round(accuracy_score(y, rf.predict(X)) * 100, 2),
        "gb_acc": round(accuracy_score(y, gb.predict(X)) * 100, 2),
        "training_rows": len(data),
        "tickers_count": len(tickers)
    },
    "feature_importance": dict(zip(features, [round(float(x)*100, 1) for x in rf.feature_importances_]))
}

joblib.dump(meta, "stock_model.pkl")
print("Ensemble Model Saved:", meta["metrics"])
