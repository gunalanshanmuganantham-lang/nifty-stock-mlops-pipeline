from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="NiftyPulse AI — Groww-Style NSE Quant Analyzer", version="3.0.0")
model = joblib.load("stock_model.pkl")

NSE_DIRECTORY = [
    {"sym": "RELIANCE.NS", "name": "Reliance Industries Ltd", "sector": "Energy & Telecom"},
    {"sym": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking"},
    {"sym": "SBIN.NS", "name": "State Bank of India", "sector": "PSU Bank"},
    {"sym": "HAL.NS", "name": "Hindustan Aeronautics Ltd", "sector": "Defence"},
    {"sym": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Automobile"},
    {"sym": "CUPID.NS", "name": "Cupid Ltd", "sector": "Healthcare"},
    {"sym": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT"},
    {"sym": "INFY.NS", "name": "Infosys Ltd", "sector": "IT"},
    {"sym": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking"},
    {"sym": "IRCTC.NS", "name": "IRCTC Ltd", "sector": "Railways"},
    {"sym": "ZOMATO.NS", "name": "Zomato Ltd", "sector": "Consumer Tech"},
    {"sym": "SUZLON.NS", "name": "Suzlon Energy Ltd", "sector": "Renewable Energy"},
    {"sym": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
    {"sym": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG"},
    {"sym": "LT.NS", "name": "Larsen & Toubro Ltd", "sector": "Infrastructure"},
    {"sym": "^NSEI", "name": "NIFTY 50 Index", "sector": "Benchmark Index"},
    {"sym": "^NSEBANK", "name": "NIFTY BANK Index", "sector": "Benchmark Index"}
]

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NiftyPulse AI | NSE Quant & Trend Terminal</title>
<style>
  :root { --bg: #0b0e14; --card: #121721; --border: #1e2636; --green: #00d09c; --red: #eb5b3c; --text: #f1f5f9; --muted: #8b9bb4; --accent: #38bdf8; }
  * { box-sizing: border-box; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
  body { background: var(--bg); color: var(--text); margin: 0; padding: 16px; display: flex; justify-content: center; }
  .app { max-width: 480px; width: 100%; }
  .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
  .brand { font-size: 1.2rem; font-weight: 700; color: var(--green); display: flex; align-items: center; gap: 6px; }
  .badge { background: rgba(0,208,156,0.12); color: var(--green); font-size: 0.7rem; padding: 3px 8px; border-radius: 12px; font-weight: 600; }
  .search-wrap { position: relative; margin-bottom: 14px; }
  .search-box { display: flex; background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 4px; }
  input { flex: 1; background: transparent; border: none; color: #fff; padding: 10px 12px; font-size: 0.95rem; outline: none; }
  .btn-go { background: var(--green); color: #052e24; border: none; border-radius: 9px; padding: 0 16px; font-weight: 700; cursor: pointer; font-size: 0.9rem; }
  .dropdown { position: absolute; top: 48px; left: 0; right: 0; background: #171e2c; border: 1px solid var(--border); border-radius: 12px; z-index: 50; max-height: 220px; overflow-y: auto; display: none; box-shadow: 0 12px 30px rgba(0,0,0,0.6); }
  .dd-item { padding: 10px 14px; border-bottom: 1px solid var(--border); cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
  .dd-item:hover { background: #1f293d; }
  .dd-name { font-size: 0.88rem; font-weight: 600; }
  .dd-sec { font-size: 0.72rem; color: var(--muted); }
  .dd-sym { font-size: 0.78rem; color: var(--green); font-weight: 600; }
  .chips { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 8px; margin-bottom: 12px; scrollbar-width: none; }
  .chip { background: var(--card); border: 1px solid var(--border); color: var(--text); padding: 6px 12px; border-radius: 20px; font-size: 0.78rem; white-space: nowrap; cursor: pointer; }
  .chip:hover { border-color: var(--green); color: var(--green); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 18px; margin-bottom: 14px; }
  .stock-hdr { display: flex; justify-content: space-between; align-items: flex-start; }
  .s-title { font-size: 1.15rem; font-weight: 700; }
  .s-sub { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
  .price { font-size: 1.65rem; font-weight: 800; margin: 10px 0 2px; }
  .chg { font-size: 0.85rem; font-weight: 600; }
  .up { color: var(--green); }
  .dn { color: var(--red); }
  .chart-box { margin: 14px 0 8px; height: 150px; width: 100%; }
  .tf-row { display: flex; justify-content: space-around; background: var(--bg); padding: 4px; border-radius: 10px; margin-top: 8px; }
  .tf { background: transparent; border: none; color: var(--muted); font-size: 0.78rem; font-weight: 600; padding: 6px 14px; border-radius: 8px; cursor: pointer; }
  .tf.active { background: var(--card); color: var(--green); }
  .ai-box { background: linear-gradient(135deg, rgba(0,208,156,0.08), rgba(56,189,248,0.06)); border: 1px solid rgba(0,208,156,0.25); }
  .ai-hdr { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .ai-title { font-size: 0.92rem; font-weight: 700; color: var(--accent); }
  .sig-pill { padding: 4px 10px; border-radius: 8px; font-size: 0.78rem; font-weight: 700; }
  .sig-bull { background: rgba(0,208,156,0.18); color: var(--green); }
  .sig-bear { background: rgba(235,91,60,0.18); color: var(--red); }
  .meter-bg { background: #1e2636; height: 8px; border-radius: 6px; overflow: hidden; margin: 8px 0 14px; }
  .meter-fill { height: 100%; background: var(--green); transition: width 0.5s ease; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat { background: var(--bg); padding: 10px 12px; border-radius: 10px; border: 1px solid var(--border); }
  .st-lbl { font-size: 0.72rem; color: var(--muted); }
  .st-val { font-size: 0.95rem; font-weight: 700; margin-top: 3px; }
  .loader { text-align: center; padding: 30px; color: var(--green); font-size: 0.9rem; display: none; }
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="brand">⚡ NiftyPulse AI</div>
    <div class="badge">MLOps v3.0 LIVE</div>
  </div>

  <div class="search-wrap">
    <div class="search-box">
      <input type="text" id="q" placeholder="Search HDFC, Reliance, HAL, Cupid..." autocomplete="off" oninput="filterList()" onfocus="filterList()">
      <button class="btn-go" onclick="loadStock(document.getElementById('q').value)">Analyze</button>
    </div>
    <div class="dropdown" id="dd"></div>
  </div>

  <div class="chips">
    <button class="chip" onclick="loadStock('HDFCBANK.NS')">HDFC Bank</button>
    <button class="chip" onclick="loadStock('RELIANCE.NS')">Reliance</button>
    <button class="chip" onclick="loadStock('HAL.NS')">HAL</button>
    <button class="chip" onclick="loadStock('SBIN.NS')">SBI</button>
    <button class="chip" onclick="loadStock('CUPID.NS')">Cupid</button>
    <button class="chip" onclick="loadStock('^NSEI')">Nifty 50</button>
  </div>

  <div class="loader" id="loader">Syncing NSE feed & running RandomForest quant model...</div>

  <div id="content" style="display:none;">
    <div class="card">
      <div class="stock-hdr">
        <div>
          <div class="s-title" id="c-name"></div>
          <div class="s-sub" id="c-sym"></div>
        </div>
        <div class="badge" id="c-dma"></div>
      </div>
      <div class="price" id="c-price"></div>
      <div class="chg" id="c-chg"></div>

      <svg class="chart-box" id="chart" viewBox="0 0 400 150" preserveAspectRatio="none"></svg>

      <div class="tf-row">
        <button class="tf" onclick="setTF('1mo', this)">1M</button>
        <button class="tf active" onclick="setTF('3mo', this)">3M</button>
        <button class="tf" onclick="setTF('6mo', this)">6M</button>
        <button class="tf" onclick="setTF('1y', this)">1Y</button>
      </div>
    </div>

    <div class="card ai-box">
      <div class="ai-hdr">
        <div class="ai-title">🤖 AI Next-Day Quant Forecast</div>
        <div class="sig-pill" id="ai-sig"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
        <span style="color:var(--muted);">Bullish Probability</span>
        <span style="font-weight:700;" id="ai-prob"></span>
      </div>
      <div class="meter-bg"><div class="meter-fill" id="ai-bar"></div></div>

      <div class="grid">
        <div class="stat"><div class="st-lbl">Support (Stop-Loss Zone)</div><div class="st-val dn" id="st-sup"></div></div>
        <div class="stat"><div class="st-lbl">Resistance (Target Zone)</div><div class="st-val up" id="st-res"></div></div>
        <div class="stat"><div class="st-lbl">14-Day RSI Momentum</div><div class="st-val" id="st-rsi"></div></div>
        <div class="stat"><div class="st-lbl">20-Day Moving Avg</div><div class="st-val" id="st-sma"></div></div>
      </div>
    </div>
  </div>
</div>

<script>
let STOCKS = [];
let currentSym = "HDFCBANK.NS";
let currentPeriod = "3mo";

fetch('/stocks').then(r => r.json()).then(d => { STOCKS = d; loadStock('HDFCBANK.NS'); });

function filterList() {
  const val = document.getElementById('q').value.trim().toLowerCase();
  const dd = document.getElementById('dd');
  if (!val) { dd.style.display = 'none'; return; }
  const matches = STOCKS.filter(s => s.name.toLowerCase().includes(val) || s.sym.toLowerCase().includes(val));
  if (!matches.length) { dd.style.display = 'none'; return; }
  dd.innerHTML = matches.map(m => `
    <div class="dd-item" onclick="selectStock('${m.sym}', '${m.name}')">
      <div><div class="dd-name">${m.name}</div><div class="dd-sec">${m.sector}</div></div>
      <div class="dd-sym">${m.sym.replace('.NS','')}</div>
    </div>
  `).join('');
  dd.style.display = 'block';
}

function selectStock(sym, name) {
  document.getElementById('q').value = name;
  document.getElementById('dd').style.display = 'none';
  loadStock(sym);
}

function setTF(p, btn) {
  document.querySelectorAll('.tf').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentPeriod = p;
  loadStock(currentSym);
}

async function loadStock(sym) {
  if (!sym) return;
  document.getElementById('dd').style.display = 'none';
  currentSym = sym;
  document.getElementById('loader').style.display = 'block';
  document.getElementById('content').style.display = 'none';

  const res = await fetch(`/analyze?ticker=${encodeURIComponent(sym)}&period=${currentPeriod}`);
  const d = await res.json();
  document.getElementById('loader').style.display = 'none';
  if (!res.ok) { alert(d.detail || 'Stock not found'); return; }

  document.getElementById('c-name').innerText = d.company_name;
  document.getElementById('c-sym').innerText = `${d.ticker} • NSE`;
  document.getElementById('c-price').innerText = `₹${d.latest_close_inr.toLocaleString('en-IN')}`;
  const isUp = d.day_change_inr >= 0;
  document.getElementById('c-chg').innerHTML = `<span class="${isUp ? 'up' : 'dn'}">${isUp ? '+' : ''}₹${d.day_change_inr} (${isUp ? '+' : ''}${d.day_change_pct}%) Today</span>`;
  document.getElementById('c-dma').innerText = d.trend_above_20dma ? '▲ Above 20-DMA' : '▼ Below 20-DMA';

  const bull = d.ml_signal.includes('BULLISH');
  const sigEl = document.getElementById('ai-sig');
  sigEl.innerText = d.ml_signal;
  sigEl.className = `sig-pill ${bull ? 'sig-bull' : 'sig-bear'}`;
  document.getElementById('ai-prob').innerText = `${d.next_day_up_probability_pct}%`;
  const bar = document.getElementById('ai-bar');
  bar.style.width = `${d.next_day_up_probability_pct}%`;
  bar.style.background = bull ? 'var(--green)' : 'var(--red)';

  document.getElementById('st-sup').innerText = `₹${d.support_inr}`;
  document.getElementById('st-res').innerText = `₹${d.resistance_inr}`;
  document.getElementById('st-rsi').innerText = `${d.rsi_14} (${d.rsi_zone})`;
  document.getElementById('st-sma').innerText = `₹${d.sma_20_inr}`;

  drawChart(d.chart_prices, isUp);
  document.getElementById('content').style.display = 'block';
}

function drawChart(prices, isUp) {
  const svg = document.getElementById('chart');
  const min = Math.min(...prices), max = Math.max(...prices);
  const span = (max - min) || 1;
  const pts = prices.map((p, i) => {
    const x = (i / (prices.length - 1)) * 400;
    const y = 140 - ((p - min) / span) * 125;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const color = isUp ? '#00d09c' : '#eb5b3c';
  svg.innerHTML = `
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${color}" stop-opacity="0.32"/>
        <stop offset="100%" stop-color="${color}" stop-opacity="0.0"/>
      </linearGradient>
    </defs>
    <polygon points="0,150 ${pts} 400,150" fill="url(#g)" />
    <polyline fill="none" stroke="${color}" stroke-width="2.5" points="${pts}" />
  `;
}

document.addEventListener('click', e => { if (!e.target.closest('.search-wrap')) document.getElementById('dd').style.display = 'none'; });
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_UI

@app.get("/stocks")
def get_stocks():
    return NSE_DIRECTORY

@app.get("/analyze")
def analyze_stock(ticker: str = "HDFCBANK.NS", period: str = "3mo"):
    q = ticker.strip().upper()
    matched_name = q
    for item in NSE_DIRECTORY:
        if q in item["sym"].upper() or q in item["name"].upper():
            q = item["sym"]
            matched_name = item["name"]
            break
    if not q.startswith("^") and not q.endswith(".NS") and not q.endswith(".BO"):
        q = f"{q}.NS"

    valid_periods = {"1mo": "3mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y"}
    fetch_period = valid_periods.get(period, "3mo")

    df = yf.download(q, period=fetch_period, interval="1d", progress=False)
    if df.empty or len(df) < 22:
        raise HTTPException(status_code=404, detail=f"Stock '{q}' not found on NSE.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_Ratio"] = df["Close"] / df["SMA_20"]
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    clean_df = df.dropna()
    latest = clean_df.iloc[-1]
    prev = clean_df.iloc[-2]

    X_live = np.array([[latest["Return_1d"], latest["Return_5d"], latest["SMA_Ratio"], latest["Volatility_10d"]]])
    prob_up = round(float(model.predict_proba(X_live)[0][1]) * 100, 1)
    signal = "BULLISH (UP)" if prob_up >= 52 else "BEARISH / SIDEWAYS"

    price = round(float(latest["Close"]), 2)
    prev_close = round(float(prev["Close"]), 2)
    day_chg = round(price - prev_close, 2)
    day_pct = round((day_chg / prev_close) * 100, 2)
    move = round(price * float(latest["Volatility_10d"]), 2)
    rsi_val = round(float(latest["RSI_14"]), 1)
    rsi_zone = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")

    chart_slice = df["Close"].tail(22 if period == "1mo" else len(df)).dropna()

    return {
        "ticker": q,
        "company_name": matched_name,
        "latest_close_inr": price,
        "day_change_inr": day_chg,
        "day_change_pct": day_pct,
        "sma_20_inr": round(float(latest["SMA_20"]), 2),
        "trend_above_20dma": bool(price > latest["SMA_20"]),
        "next_day_up_probability_pct": prob_up,
        "ml_signal": signal,
        "support_inr": round(price - move, 2),
        "resistance_inr": round(price + move, 2),
        "rsi_14": rsi_val,
        "rsi_zone": rsi_zone,
        "chart_prices": [round(float(x), 2) for x in chart_slice.tolist()]
    }
  
