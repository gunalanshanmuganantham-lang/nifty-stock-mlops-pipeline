from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="NSE Live Stock Trend & Risk MLOps API", version="2.0.0")
model = joblib.load("stock_model.pkl")

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSE AI Stock Analyzer</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; display: flex; justify-content: center; }
  .card { background: #1e293b; border-radius: 16px; padding: 24px; max-width: 440px; width: 100%; box-shadow: 0 10px 25px rgba(0,0,0,0.4); border: 1px solid #334155; }
  h1 { font-size: 1.4rem; margin: 0 0 6px; color: #38bdf8; }
  p.sub { font-size: 0.85rem; color: #94a3b8; margin: 0 0 18px; }
  .search-box { display: flex; gap: 8px; margin-bottom: 12px; }
  input { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #475569; background: #0f172a; color: #fff; font-size: 1rem; text-transform: uppercase; }
  button { padding: 12px 18px; border-radius: 10px; border: none; background: #0284c7; color: #fff; font-weight: 600; cursor: pointer; font-size: 0.95rem; }
  button:hover { background: #0369a1; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px; }
  .chip { background: #334155; color: #e2e8f0; padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; cursor: pointer; border: none; }
  .chip:hover { background: #475569; }
  .result { background: #0f172a; border-radius: 12px; padding: 16px; border: 1px solid #334155; display: none; }
  .row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e293b; font-size: 0.92rem; }
  .row:last-child { border-bottom: none; }
  .label { color: #94a3b8; }
  .val { font-weight: 600; }
  .bull { color: #4ade80; }
  .bear { color: #f87171; }
  .loader { text-align: center; color: #38bdf8; display: none; padding: 15px; font-size: 0.9rem; }
  .err { color: #f87171; font-size: 0.9rem; margin-top: 10px; display: none; }
</style>
</head>
<body>
<div class="card">
  <h1>📈 NSE AI Stock Analyzer</h1>
  <p class="sub">Real-Time ML Trend & Volatility Range (Powered by MLOps)</p>
  <div class="search-box">
    <input type="text" id="ticker" placeholder="Enter Stock (e.g. SBIN, HAL)" value="SBIN">
    <button onclick="analyze()">Analyze</button>
  </div>
  <div class="chips">
    <button class="chip" onclick="quick('RELIANCE')">RELIANCE</button>
    <button class="chip" onclick="quick('HAL')">HAL</button>
    <button class="chip" onclick="quick('SBIN')">SBIN</button>
    <button class="chip" onclick="quick('CUPID')">CUPID</button>
    <button class="chip" onclick="quick('TATAMOTORS')">TATAMOTORS</button>
    <button class="chip" onclick="quick('^NSEI')">NIFTY 50</button>
  </div>
  <div class="loader" id="loader">Fetching live NSE market data & running ML model...</div>
  <div class="err" id="err"></div>
  <div class="result" id="result">
    <div class="row"><span class="label">Stock Ticker</span><span class="val" id="r-ticker"></span></div>
    <div class="row"><span class="label">Latest Close</span><span class="val" id="r-price"></span></div>
    <div class="row"><span class="label">20-Day Moving Avg</span><span class="val" id="r-sma"></span></div>
    <div class="row"><span class="label">20-DMA Trend</span><span class="val" id="r-dma"></span></div>
    <div class="row"><span class="label">Next-Day UP Chance</span><span class="val" id="r-prob"></span></div>
    <div class="row"><span class="label">ML Signal</span><span class="val" id="r-sig"></span></div>
    <div class="row"><span class="label">Expected Daily Range</span><span class="val" id="r-range"></span></div>
  </div>
</div>
<script>
function quick(sym) {
  document.getElementById('ticker').value = sym;
  analyze();
}
async function analyze() {
  const sym = document.getElementById('ticker').value.trim();
  if (!sym) return;
  document.getElementById('loader').style.display = 'block';
  document.getElementById('result').style.display = 'none';
  document.getElementById('err').style.display = 'none';
  try {
    const res = await fetch('/analyze?ticker=' + encodeURIComponent(sym));
    const data = await res.json();
    document.getElementById('loader').style.display = 'none';
    if (!res.ok) {
      document.getElementById('err').innerText = data.detail || 'Error fetching stock data';
      document.getElementById('err').style.display = 'block';
      return;
    }
    document.getElementById('r-ticker').innerText = data.ticker;
    document.getElementById('r-price').innerText = '₹ ' + data.latest_close_inr;
    document.getElementById('r-sma').innerText = '₹ ' + data.sma_20_inr;
    document.getElementById('r-dma').innerHTML = data.trend_above_20dma ? '<span class="bull">Above 20-DMA ▲</span>' : '<span class="bear">Below 20-DMA ▼</span>';
    document.getElementById('r-prob').innerText = data.next_day_up_probability_pct + '%';
    const isBull = data.ml_signal.includes('BULLISH');
    document.getElementById('r-sig').innerHTML = `<span class="${isBull ? 'bull' : 'bear'}">${data.ml_signal}</span>`;
    document.getElementById('r-range').innerText = '₹ ' + data.expected_daily_range_inr;
    document.getElementById('result').style.display = 'block';
  } catch (e) {
    document.getElementById('loader').style.display = 'none';
    document.getElementById('err').innerText = 'Network error. Try again.';
    document.getElementById('err').style.display = 'block';
  }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_UI

@app.get("/analyze")
def analyze_stock(ticker: str = "RELIANCE"):
    clean_ticker = ticker.strip().upper()
    if not clean_ticker.startswith("^") and not clean_ticker.endswith(".NS") and not clean_ticker.endswith(".BO"):
        clean_ticker = f"{clean_ticker}.NS"

    df = yf.download(clean_ticker, period="3mo", interval="1d", progress=False)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Stock '{clean_ticker}' not found on NSE. Check symbol spelling.")
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
        "ticker": clean_ticker,
        "latest_close_inr": price,
        "sma_20_inr": round(float(latest["SMA_20"]), 2),
        "trend_above_20dma": bool(price > latest["SMA_20"]),
        "next_day_up_probability_pct": prob_up,
        "ml_signal": signal,
        "expected_daily_range_inr": f"{round(price - expected_move, 2)} - {round(price + expected_move, 2)}"
    }
