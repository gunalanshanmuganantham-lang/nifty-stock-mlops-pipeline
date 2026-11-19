from fastapi import FastAPI, HTTPException
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="NSE Live Stock Trend & Risk MLOps API", version="1.0.0")
model = joblib.load("stock_model.pkl")

@app.get("/")
def home():
    return {
        "status": "Live",
        "message": "NSE Live Stock Trend & Risk Analyzer",
        "docs": "/docs"
    }

@app.get("/analyze")
def analyze_stock(ticker: str = "RELIANCE.NS"):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False)
    if df.empty:
        raise HTTPException(status_code=404, detail="Invalid NSE Ticker. Use format like HAL.NS, SBIN.NS, CUPID.NS")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_Ratio"] = df["Close"] / df["SMA_20"]
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    latest = df.dropna().iloc[-1]

    X_live = np.array([[latest["Return_1d"], latest["Return_5d"], latest["SMA_Ratio"], latest["Volatility_10d"]]])
    prob_up = round(float(model.predict_proba(X_live)[0][1]) * 100, 1)
    signal = "BULLISH (UP)" if prob_up >= 52 else "BEARISH / SIDEWAYS"

    price = round(float(latest["Close"]), 2)
    expected_move = round(price * float(latest["Volatility_10d"]), 2)

    return {
        "ticker": ticker.upper(),
        "latest_close_inr": price,
        "sma_20_inr": round(float(latest["SMA_20"]), 2),
        "trend_above_20dma": bool(price > latest["SMA_20"]),
        "next_day_up_probability_pct": prob_up,
        "ml_signal": signal,
        "expected_daily_range_inr": f"{round(price - expected_move, 2)} - {round(price + expected_move, 2)}"
    }
  
