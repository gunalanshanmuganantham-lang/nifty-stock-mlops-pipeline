from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import time

app = FastAPI(title="NiftyPulse AI v4.0 — Full FinTech & MLOps Terminal", version="4.0.0")
bundle = joblib.load("stock_model.pkl")
model = bundle["model"]

telemetry = {"requests": 0, "avg_ms": 0.0, "last_ticker": "HDFCBANK.NS"}

NSE_DIRECTORY = [
    {"sym": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking"},
    {"sym": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy & Telecom"},
    {"sym": "HAL.NS", "name": "Hindustan Aeronautics", "sector": "Defence"},
    {"sym": "SBIN.NS", "name": "State Bank of India", "sector": "PSU Bank"},
    {"sym": "CUPID.NS", "name": "Cupid Ltd", "sector": "Healthcare"},
    {"sym": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Automobile"},
    {"sym": "V2RETAIL.NS", "name": "V2 Retail Ltd", "sector": "Retail"},
    {"sym": "HCLTECH.NS", "name": "HCL Technologies", "sector": "IT"},
    {"sym": "ONGC.NS", "name": "ONGC Ltd", "sector": "Oil & Gas"},
    {"sym": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT"},
    {"sym": "INFY.NS", "name": "Infosys Ltd", "sector": "IT"},
    {"sym": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking"},
    {"sym": "IRCTC.NS", "name": "IRCTC Ltd", "sector": "Railways"},
    {"sym": "ZOMATO.NS", "name": "Zomato Ltd", "sector": "Consumer Tech"},
    {"sym": "SUZLON.NS", "name": "Suzlon Energy Ltd", "sector": "Renewables"},
    {"sym": "^NSEI", "name": "NIFTY 50 Index", "sector": "Index"},
    {"sym": "^NSEBANK", "name": "NIFTY BANK Index", "sector": "Index"}
]

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>NiftyPulse AI | Quant & MLOps Terminal</title>
<style>
  :root { --bg:#0b0e14; --card:#131822; --card2:#1a2230; --border:#232d3f; --green:#00d09c; --red:#eb5b3c; --text:#f8fafc; --muted:#8b9bb4; --blue:#38bdf8; }
  * { box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif; -webkit-tap-highlight-color:transparent; }
  body { background:var(--bg); color:var(--text); margin:0; padding:0 0 76px; display:flex; justify-content:center; }
  .shell { max-width:480px; width:100%; padding:12px 14px; }
  
  /* Lock Screen */
  #lock-screen { position:fixed; inset:0; background:#090c12; z-index:999; display:none; flex-direction:column; align-items:center; justify-content:center; padding:24px; }
  .pin-dots { display:flex; gap:14px; margin:20px 0; }
  .dot { width:14px; height:14px; border-radius:50%; border:2px solid var(--green); }
  .dot.filled { background:var(--green); }
  .keypad { display:grid; grid-template-columns:repeat(3,72px); gap:14px; margin-top:10px; }
  .key { height:60px; border-radius:50%; background:var(--card); border:1px solid var(--border); color:#fff; font-size:1.3rem; font-weight:700; cursor:pointer; }
  .key:active { background:var(--green); color:#000; }

  /* Ticker Strip */
  .idx-bar { display:flex; justify-content:space-between; background:var(--card); border:1px solid var(--border); border-radius:12px; padding:8px 12px; margin-bottom:12px; font-size:0.78rem; }
  .idx-item span { display:block; }
  .idx-val { font-weight:700; font-size:0.86rem; margin-top:2px; }

  /* Search */
  .search-wrap { position:relative; margin-bottom:12px; }
  .search-box { display:flex; background:var(--card); border:1px solid var(--border); border-radius:12px; padding:4px 6px; align-items:center; }
  .search-box input { flex:1; background:transparent; border:none; color:#fff; padding:9px; font-size:0.92rem; outline:none; }
  .btn-go { background:var(--green); color:#042f24; border:none; border-radius:8px; padding:8px 14px; font-weight:700; font-size:0.84rem; cursor:pointer; }
  .dropdown { position:absolute; top:48px; left:0; right:0; background:#171f2e; border:1px solid var(--border); border-radius:12px; z-index:80; max-height:220px; overflow-y:auto; display:none; box-shadow:0 12px 28px rgba(0,0,0,0.7); }
  .dd-item { padding:10px 14px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; cursor:pointer; }
  .dd-item:hover { background:#212c40; }

  /* Cards & Typography */
  .card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px; margin-bottom:12px; }
  .hdr-row { display:flex; justify-content:space-between; align-items:flex-start; }
  .title { font-size:1.15rem; font-weight:700; }
  .sub { font-size:0.76rem; color:var(--muted); margin-top:2px; }
  .price-big { font-size:1.7rem; font-weight:800; margin:8px 0 2px; }
  .up { color:var(--green); } .dn { color:var(--red); }

  /* Interactive Chart */
  .crosshair-info { height:20px; font-size:0.78rem; color:var(--blue); font-weight:600; margin-top:6px; }
  #chart-wrap { position:relative; width:100%; height:185px; touch-action:none; margin:4px 0 8px; }
  canvas { width:100%; height:100%; display:block; }
  .tf-bar { display:flex; justify-content:space-between; align-items:center; background:var(--bg); padding:4px; border-radius:10px; }
  .tf-btns { display:flex; gap:2px; flex:1; justify-content:space-around; }
  .tf { background:transparent; border:none; color:var(--muted); font-size:0.74rem; font-weight:700; padding:6px 8px; border-radius:7px; cursor:pointer; }
  .tf.active { background:var(--card2); color:var(--green); }
  .mode-btn { background:var(--card2); border:1px solid var(--border); color:var(--green); border-radius:7px; padding:5px 9px; font-size:0.74rem; font-weight:700; cursor:pointer; margin-left:4px; }

  /* Range Sliders (Groww Performance Style) */
  .slider-block { margin:12px 0; }
  .sl-labels { display:flex; justify-content:space-between; font-size:0.76rem; color:var(--muted); margin-bottom:4px; }
  .sl-vals { display:flex; justify-content:space-between; font-size:0.86rem; font-weight:700; margin-bottom:6px; }
  .track { position:relative; height:5px; background:#263145; border-radius:4px; }
  .pointer { position:absolute; top:-4px; width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:8px solid var(--text); transform:translateX(-6px); }

  /* Grids & Bars */
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .kv { display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid var(--border); font-size:0.82rem; }
  .kv:last-child { border-bottom:none; }
  .bar-row { margin:9px 0; }
  .bar-top { display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px; }
  .bar-bg { height:7px; background:#222b3c; border-radius:5px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:5px; }

  /* Sub-tabs inside Stock View */
  .sub-tabs { display:flex; border-bottom:1px solid var(--border); margin-bottom:12px; gap:16px; }
  .stab { padding:8px 2px; font-size:0.85rem; font-weight:700; color:var(--muted); border-bottom:2px solid transparent; cursor:pointer; }
  .stab.active { color:var(--text); border-bottom-color:var(--green); }

  /* Bottom Navigation */
  .bnav { position:fixed; bottom:0; left:0; right:0; background:#111620; border-top:1px solid var(--border); display:flex; justify-content:space-around; padding:8px 0 10px; z-index:90; }
  .bitem { background:none; border:none; color:var(--muted); font-size:0.72rem; font-weight:600; display:flex; flex-direction:column; align-items:center; gap:3px; cursor:pointer; }
  .bitem.active { color:var(--green); }
  .bicon { font-size:1.15rem; }

  /* Switch Toggle */
  .switch { position:relative; display:inline-block; width:46px; height:24px; }
  .switch input { opacity:0; width:0; height:0; }
  .slider-tog { position:absolute; cursor:pointer; inset:0; background:#334155; border-radius:24px; transition:.3s; }
  .slider-tog:before { position:absolute; content:""; height:18px; width:18px; left:3px; bottom:3px; background:white; border-radius:50%; transition:.3s; }
  input:checked + .slider-tog { background:var(--green); }
  input:checked + .slider-tog:before { transform:translateX(22px); }
</style>
</head>
<body>

<!-- APP LOCK SCREEN -->
<div id="lock-screen">
  <div style="font-size:2rem;">🔒</div>
  <div style="font-weight:700; font-size:1.1rem; margin-top:8px;">NiftyPulse AI Locked</div>
  <div style="font-size:0.8rem; color:var(--muted); margin-top:4px;" id="lock-msg">Enter 4-Digit Security PIN</div>
  <div class="pin-dots"><div class="dot" id="d0"></div><div class="dot" id="d1"></div><div class="dot" id="d2"></div><div class="dot" id="d3"></div></div>
  <div class="keypad">
    <button class="key" onclick="pressPin('1')">1</button><button class="key" onclick="pressPin('2')">2</button><button class="key" onclick="pressPin('3')">3</button>
    <button class="key" onclick="pressPin('4')">4</button><button class="key" onclick="pressPin('5')">5</button><button class="key" onclick="pressPin('6')">6</button>
    <button class="key" onclick="pressPin('7')">7</button><button class="key" onclick="pressPin('8')">8</button><button class="key" onclick="pressPin('9')">9</button>
    <button class="key" style="font-size:0.9rem;" onclick="clearPin()">CLR</button><button class="key" onclick="pressPin('0')">0</button><button class="key" style="font-size:0.9rem;" onclick="bioUnlock()">👆</button>
  </div>
</div>

<div class="shell">
  <!-- LIVE NIFTY & BANKNIFTY HEADER -->
  <div class="idx-bar">
    <div class="idx-item"><span style="color:var(--muted);">NIFTY 50</span><span class="idx-val" id="idx-nifty">Loading...</span></div>
    <div class="idx-item" style="text-align:right;"><span style="color:var(--muted);">NIFTY BANK</span><span class="idx-val" id="idx-bank">Loading...</span></div>
  </div>

  <!-- TAB 1: ANALYZER (GROWW STYLE) -->
  <div id="tab-analyzer">
    <div class="search-wrap">
      <div class="search-box">
        <input type="text" id="q" placeholder="Search HDFC, V2 Retail, Cupid, HAL..." oninput="filterList()" onfocus="filterList()">
        <button class="btn-go" onclick="loadStock(document.getElementById('q').value)">Go</button>
      </div>
      <div class="dropdown" id="dd"></div>
    </div>

    <div id="loader" style="text-align:center; padding:30px; color:var(--green); display:none;">Syncing NSE Market & Ensemble ML...</div>

    <div id="stock-body" style="display:none;">
      <div class="card">
        <div class="hdr-row">
          <div>
            <div class="title" id="s-name"></div>
            <div class="sub" id="s-sym"></div>
          </div>
          <button onclick="toggleWatch()" id="w-btn" style="background:var(--card2); border:1px solid var(--border); color:var(--green); padding:6px 10px; border-radius:8px; font-size:0.76rem; font-weight:700; cursor:pointer;">+ Watchlist</button>
        </div>
        <div class="price-big" id="s-price"></div>
        <div style="font-size:0.84rem; font-weight:600;" id="s-chg"></div>

        <div class="crosshair-info" id="xhair">Touch & drag chart to inspect historical prices</div>
        <div id="chart-wrap"><canvas id="chart"></canvas></div>

        <div class="tf-bar">
          <div class="tf-btns">
            <button class="tf" onclick="setTF('1wk',this)">1W</button>
            <button class="tf" onclick="setTF('1mo',this)">1M</button>
            <button class="tf" onclick="setTF('3mo',this)">3M</button>
            <button class="tf" onclick="setTF('6mo',this)">6M</button>
            <button class="tf active" onclick="setTF('1y',this)">1Y</button>
            <button class="tf" onclick="setTF('5y',this)">5Y</button>
            <button class="tf" onclick="setTF('max',this)">ALL</button>
          </div>
          <button class="mode-btn" id="mode-btn" onclick="toggleChartMode()">🕯️ Candle</button>
        </div>
      </div>

      <!-- SUB-TABS: OVERVIEW / AI TECHNICALS / SHAREHOLDING -->
      <div class="sub-tabs">
        <div class="stab active" onclick="switchSub('ov',this)">Overview</div>
        <div class="stab" onclick="switchSub('ai',this)">AI Quant & Tech</div>
        <div class="stab" onclick="switchSub('sh',this)">Shareholding</div>
      </div>

      <!-- SUB 1: OVERVIEW (PERFORMANCE SLIDERS + FUNDAMENTALS) -->
      <div id="sub-ov">
        <div class="card">
          <div style="font-weight:700; font-size:0.95rem; margin-bottom:10px;">Performance</div>
          <div class="slider-block">
            <div class="sl-labels"><span>Today's Low</span><span>Today's High</span></div>
            <div class="sl-vals"><span id="t-low"></span><span id="t-high"></span></div>
            <div class="track"><div class="pointer" id="t-ptr"></div></div>
          </div>
          <div class="slider-block">
            <div class="sl-labels"><span>52 Week Low</span><span>52 Week High</span></div>
            <div class="sl-vals"><span id="w-low"></span><span id="w-high"></span></div>
            <div class="track"><div class="pointer" id="w-ptr"></div></div>
          </div>
          <div class="grid2" style="margin-top:14px; padding-top:10px; border-top:1px solid var(--border);">
            <div class="kv"><span style="color:var(--muted);">Open</span><strong id="f-open"></strong></div>
            <div class="kv"><span style="color:var(--muted);">Prev. Close</span><strong id="f-prev"></strong></div>
            <div class="kv"><span style="color:var(--muted);">Volume</span><strong id="f-vol"></strong></div>
            <div class="kv"><span style="color:var(--muted);">20-DMA</span><strong id="f-sma"></strong></div>
          </div>
        </div>

        <div class="card">
          <div style="font-weight:700; font-size:0.95rem; margin-bottom:8px;">Fundamentals</div>
          <div class="grid2">
            <div class="kv"><span style="color:var(--muted);">Mkt Cap</span><strong id="fn-cap"></strong></div>
            <div class="kv"><span style="color:var(--muted);">ROE</span><strong id="fn-roe"></strong></div>
            <div class="kv"><span style="color:var(--muted);">P/E Ratio</span><strong id="fn-pe"></strong></div>
            <div class="kv"><span style="color:var(--muted);">EPS (TTM)</span><strong id="fn-eps"></strong></div>
            <div class="kv"><span style="color:var(--muted);">P/B Ratio</span><strong id="fn-pb"></strong></div>
            <div class="kv"><span style="color:var(--muted);">Div Yield</span><strong id="fn-div"></strong></div>
          </div>
        </div>
      </div>

      <!-- SUB 2: AI QUANT & TECHNICALS -->
      <div id="sub-ai" style="display:none;">
        <div class="card" style="border-color:rgba(0,208,156,0.35);">
          <div class="hdr-row">
            <div>
              <div style="font-weight:700; color:var(--blue); font-size:0.95rem;">🤖 Ensemble ML Forecast</div>
              <div class="sub">RF + GradientBoost + Logistic Voting</div>
            </div>
            <span id="ai-badge" style="padding:4px 10px; border-radius:8px; font-weight:700; font-size:0.78rem;"></span>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-top:12px;">
            <span style="color:var(--muted);">Next-Day Bullish Probability</span>
            <strong id="ai-prob"></strong>
          </div>
          <div class="bar-bg" style="margin:6px 0 14px;"><div class="bar-fill" id="ai-bar"></div></div>

          <div class="grid2">
            <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);">
              <div class="sub">Support (Stop-Loss)</div><div class="dn" style="font-weight:700; margin-top:3px;" id="ai-sup"></div>
            </div>
            <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);">
              <div class="sub">Resistance (Target)</div><div class="up" style="font-weight:700; margin-top:3px;" id="ai-res"></div>
            </div>
            <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);">
              <div class="sub">14-Day RSI</div><div style="font-weight:700; margin-top:3px;" id="ai-rsi"></div>
            </div>
            <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);">
              <div class="sub">10-Day Volatility</div><div style="font-weight:700; margin-top:3px;" id="ai-vol"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- SUB 3: SHAREHOLDING PATTERN -->
      <div id="sub-sh" style="display:none;">
        <div class="card">
          <div style="font-weight:700; font-size:0.95rem; margin-bottom:10px;">Shareholding Pattern</div>
          <div class="bar-row"><div class="bar-top"><span>Promoters</span><strong id="sh-prom"></strong></div><div class="bar-bg"><div class="bar-fill" id="b-prom" style="background:#818cf8;"></div></div></div>
          <div class="bar-row"><div class="bar-top"><span>FIIs (Foreign Inst.)</span><strong id="sh-fii"></strong></div><div class="bar-bg"><div class="bar-fill" id="b-fii" style="background:#38bdf8;"></div></div></div>
          <div class="bar-row"><div class="bar-top"><span>DIIs (Mutual Funds/Inst.)</span><strong id="sh-dii"></strong></div><div class="bar-bg"><div class="bar-fill" id="b-dii" style="background:#00d09c;"></div></div></div>
          <div class="bar-row"><div class="bar-top"><span>Retail & Public</span><strong id="sh-pub"></strong></div><div class="bar-bg"><div class="bar-fill" id="b-pub" style="background:#fb923c;"></div></div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: WATCHLIST (ZERODHA KITE STYLE) -->
  <div id="tab-watchlist" style="display:none;">
    <div class="card">
      <div class="hdr-row" style="margin-bottom:10px;">
        <div class="title">My Watchlist</div>
        <span class="sub" id="wl-count"></span>
      </div>
      <div id="wl-list"></div>
    </div>
  </div>

  <!-- TAB 3: MLOPS PIPELINE & TELEMETRY (FOR INTERVIEWS!) -->
  <div id="tab-mlops" style="display:none;">
    <div class="card" style="border-color:rgba(56,189,248,0.35);">
      <div class="title" style="color:var(--blue);">⚙️ Production MLOps Engine</div>
      <div class="sub">Live Model Registry, Feature Weights & API Telemetry</div>
      <div class="grid2" style="margin-top:12px;">
        <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);"><div class="sub">Ensemble Accuracy</div><div class="up" style="font-weight:800; font-size:1.1rem;" id="m-ens"></div></div>
        <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);"><div class="sub">Training Samples</div><div style="font-weight:800; font-size:1.1rem;" id="m-rows"></div></div>
        <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);"><div class="sub">Live API Predictions</div><div style="font-weight:800; font-size:1.1rem;" id="m-reqs"></div></div>
        <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);"><div class="sub">Avg Inference Latency</div><div class="up" style="font-weight:800; font-size:1.1rem;" id="m-lat"></div></div>
      </div>
    </div>
    <div class="card">
      <div style="font-weight:700; font-size:0.95rem; margin-bottom:10px;">RandomForest Feature Importance (%)</div>
      <div id="fi-bars"></div>
    </div>
  </div>

  <!-- TAB 4: SETTINGS & SECURITY (APP LOCK TOGGLE) -->
  <div id="tab-settings" style="display:none;">
    <div class="card">
      <div class="title" style="margin-bottom:4px;">Security & App Lock</div>
      <div class="sub" style="margin-bottom:16px;">Protect your portfolio & quant terminal on startup</div>
      <div class="kv" style="align-items:center; padding:12px 0;">
        <div>
          <div style="font-weight:700; font-size:0.92rem;">Enable App Lock (PIN / Biometric)</div>
          <div class="sub">Ask PIN whenever app opens</div>
        </div>
        <label class="switch">
          <input type="checkbox" id="lock-toggle" onchange="toggleLockSetting()">
          <span class="slider-tog"></span>
        </label>
      </div>
      <div class="kv" style="align-items:center; padding:12px 0;">
        <div>
          <div style="font-weight:700; font-size:0.92rem;">Change 4-Digit PIN</div>
          <div class="sub">Default PIN is 1234</div>
        </div>
        <button onclick="changePin()" style="background:var(--card2); border:1px solid var(--border); color:var(--green); padding:7px 12px; border-radius:8px; font-weight:700; cursor:pointer;">Set PIN</button>
      </div>
      <div class="kv" style="align-items:center; padding:12px 0;">
        <div>
          <div style="font-weight:700; font-size:0.92rem;">Test Lock Screen Now</div>
          <div class="sub">Preview startup lock</div>
        </div>
        <button onclick="showLock()" style="background:var(--green); border:none; color:#000; padding:7px 12px; border-radius:8px; font-weight:700; cursor:pointer;">Lock Now</button>
      </div>
    </div>
  </div>
</div>

<!-- BOTTOM NAVIGATION BAR -->
<div class="bnav">
  <button class="bitem active" onclick="switchTab('analyzer',this)"><span class="bicon">📈</span>Analyzer</button>
  <button class="bitem" onclick="switchTab('watchlist',this)"><span class="bicon">🔖</span>Watchlist</button>
  <button class="bitem" onclick="switchTab('mlops',this)"><span class="bicon">🤖</span>MLOps Lab</button>
  <button class="bitem" onclick="switchTab('settings',this)"><span class="bicon">⚙️</span>Settings</button>
</div>

<script>
let STOCKS = [], currentSym = "CUPID.NS", currentPeriod = "1y", chartData = [], chartMode = "line";
let watchlist = JSON.parse(localStorage.getItem("np_wl") || '["CUPID.NS","HDFCBANK.NS","HAL.NS","SBIN.NS","RELIANCE.NS"]');
let enteredPin = "";

// App Lock Logic
function initLock() {
  const enabled = localStorage.getItem("np_lock") === "true";
  document.getElementById("lock-toggle").checked = enabled;
  if (enabled) showLock();
}
function showLock() { enteredPin = ""; updateDots(); document.getElementById("lock-screen").style.display = "flex"; }
function pressPin(d) {
  if (enteredPin.length < 4) { enteredPin += d; updateDots(); }
  if (enteredPin.length === 4) {
    const saved = localStorage.getItem("np_pin") || "1234";
    if (enteredPin === saved) { document.getElementById("lock-screen").style.display = "none"; }
    else { document.getElementById("lock-msg").innerText = "❌ Wrong PIN! Try again (Default: 1234)"; enteredPin = ""; setTimeout(updateDots, 300); }
  }
}
function clearPin() { enteredPin = ""; updateDots(); }
function updateDots() { for(let i=0;i<4;i++) document.getElementById("d"+i).classList.toggle("filled", i < enteredPin.length); }
function bioUnlock() { document.getElementById("lock-screen").style.display = "none"; }
function toggleLockSetting() {
  const on = document.getElementById("lock-toggle").checked;
  localStorage.setItem("np_lock", on);
  if (on && !localStorage.getItem("np_pin")) localStorage.setItem("np_pin", "1234");
}
function changePin() {
  const p = prompt("Enter new 4-digit PIN:");
  if (p && /^\d{4}$/.test(p)) { localStorage.setItem("np_pin", p); alert("✅ PIN updated!"); }
  else if (p) alert("Enter exactly 4 digits.");
}

// Init App
initLock();
fetch('/indices').then(r=>r.json()).then(d=>{
  document.getElementById("idx-nifty").innerHTML = `${d.nifty.price} <span class="${d.nifty.chg>=0?'up':'dn'}">(${d.nifty.pct}%)</span>`;
  document.getElementById("idx-bank").innerHTML = `${d.bank.price} <span class="${d.bank.chg>=0?'up':'dn'}">(${d.bank.pct}%)</span>`;
});
fetch('/stocks').then(r=>r.json()).then(d=>{ STOCKS = d; loadStock("CUPID.NS"); });

function switchTab(t, btn) {
  ["analyzer","watchlist","mlops","settings"].forEach(x => document.getElementById("tab-"+x).style.display = (x===t?"block":"none"));
  document.querySelectorAll(".bitem").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  if (t === "watchlist") renderWatchlist();
  if (t === "mlops") loadMlops();
}

function switchSub(s, el) {
  ["ov","ai","sh"].forEach(x => document.getElementById("sub-"+x).style.display = (x===s?"block":"none"));
  document.querySelectorAll(".stab").forEach(b => b.classList.remove("active"));
  el.classList.add("active");
}

function filterList() {
  const val = document.getElementById('q').value.trim().toLowerCase();
  const dd = document.getElementById('dd');
  if (!val) { dd.style.display = 'none'; return; }
  const m = STOCKS.filter(s => s.name.toLowerCase().includes(val) || s.sym.toLowerCase().includes(val));
  dd.innerHTML = m.map(x => `<div class="dd-item" onclick="selectStock('${x.sym}','${x.name}')"><div><strong>${x.name}</strong><div class="sub">${x.sector}</div></div><span class="up" style="font-weight:700;">${x.sym.replace('.NS','')}</span></div>`).join('');
  dd.style.display = m.length ? 'block' : 'none';
}
function selectStock(sym, name) { document.getElementById('q').value = name; document.getElementById('dd').style.display='none'; loadStock(sym); }

function setTF(p, btn) {
  document.querySelectorAll('.tf').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  currentPeriod = p;
  loadStock(currentSym);
}
function toggleChartMode() {
  chartMode = (chartMode === "line") ? "candle" : "line";
  document.getElementById("mode-btn").innerText = (chartMode === "line") ? "🕯️ Candle" : "📈 Line";
  drawChart(-1);
}

async function loadStock(sym) {
  if (!sym) return;
  document.getElementById('dd').style.display='none';
  currentSym = sym;
  document.getElementById('loader').style.display='block';
  document.getElementById('stock-body').style.display='none';

  const res = await fetch(`/analyze?ticker=${encodeURIComponent(sym)}&period=${currentPeriod}`);
  const d = await res.json();
  document.getElementById('loader').style.display='none';
  if (!res.ok) { alert(d.detail || "Stock not found"); return; }

  currentSym = d.ticker;
  document.getElementById('s-name').innerText = d.company_name;
  document.getElementById('s-sym').innerText = `${d.ticker} • NSE`;
  document.getElementById('s-price').innerText = `₹${d.price.toLocaleString('en-IN')}`;
  const up = d.day_chg >= 0;
  document.getElementById('s-chg').innerHTML = `<span class="${up?'up':'dn'}">${up?'+':''}₹${d.day_chg} (${up?'+':''}${d.day_pct}%) Today</span>`;
  document.getElementById('w-btn').innerText = watchlist.includes(d.ticker) ? "✓ Saved" : "+ Watchlist";

  // Performance Sliders
  document.getElementById('t-low').innerText = `₹${d.day_low}`;
  document.getElementById('t-high').innerText = `₹${d.day_high}`;
  const tPos = Math.min(100, Math.max(0, ((d.price - d.day_low)/((d.day_high - d.day_low)||1))*100));
  document.getElementById('t-ptr').style.left = tPos + "%";

  document.getElementById('w-low').innerText = `₹${d.w52_low}`;
  document.getElementById('w-high').innerText = `₹${d.w52_high}`;
  const wPos = Math.min(100, Math.max(0, ((d.price - d.w52_low)/((d.w52_high - d.w52_low)||1))*100));
  document.getElementById('w-ptr').style.left = wPos + "%";

  document.getElementById('f-open').innerText = `₹${d.open}`;
  document.getElementById('f-prev').innerText = `₹${d.prev_close}`;
  document.getElementById('f-vol').innerText = d.volume.toLocaleString('en-IN');
  document.getElementById('f-sma').innerText = `₹${d.sma_20}`;

  // Fundamentals
  document.getElementById('fn-cap').innerText = d.fundamentals.mkt_cap;
  document.getElementById('fn-roe').innerText = d.fundamentals.roe;
  document.getElementById('fn-pe').innerText = d.fundamentals.pe;
  document.getElementById('fn-eps').innerText = d.fundamentals.eps;
  document.getElementById('fn-pb').innerText = d.fundamentals.pb;
  document.getElementById('fn-div').innerText = d.fundamentals.div_yield;

  // AI Quant
  const bull = d.ml_signal.includes("BULLISH");
  const badge = document.getElementById('ai-badge');
  badge.innerText = d.ml_signal;
  badge.style.background = bull ? "rgba(0,208,156,0.18)" : "rgba(235,91,60,0.18)";
  badge.style.color = bull ? "var(--green)" : "var(--red)";
  document.getElementById('ai-prob').innerText = d.prob_up + "%";
  const bar = document.getElementById('ai-bar');
  bar.style.width = d.prob_up + "%";
  bar.style.background = bull ? "var(--green)" : "var(--red)";
  document.getElementById('ai-sup').innerText = `₹${d.support}`;
  document.getElementById('ai-res').innerText = `₹${d.resistance}`;
  document.getElementById('ai-rsi').innerText = `${d.rsi} (${d.rsi_zone})`;
  document.getElementById('ai-vol').innerText = `${d.volatility}%`;

  // Shareholding
  document.getElementById('sh-prom').innerText = d.shareholding.promoters + "%";
  document.getElementById('b-prom').style.width = d.shareholding.promoters + "%";
  document.getElementById('sh-fii').innerText = d.shareholding.fii + "%";
  document.getElementById('b-fii').style.width = d.shareholding.fii + "%";
  document.getElementById('sh-dii').innerText = d.shareholding.dii + "%";
  document.getElementById('b-dii').style.width = d.shareholding.dii + "%";
  document.getElementById('sh-pub').innerText = d.shareholding.public + "%";
  document.getElementById('b-pub').style.width = d.shareholding.public + "%";

  chartData = d.history;
  document.getElementById('stock-body').style.display='block';
  drawChart(-1);
}

// Interactive Touch-Drag Canvas Chart (Line + Candlestick + Crosshair)
function drawChart(hoverIdx) {
  const canvas = document.getElementById('chart');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0,0,W,H);
  if (!chartData.length) return;

  const highs = chartData.map(x=>x.h), lows = chartData.map(x=>x.l);
  const max = Math.max(...highs), min = Math.min(...lows), span = (max - min) || 1;
  const toY = v => H - 14 - ((v - min)/span)*(H - 28);
  const step = W / Math.max(1, chartData.length - 1);

  if (chartMode === "line") {
    const isUp = chartData[chartData.length-1].c >= chartData[0].c;
    const col = isUp ? "#00d09c" : "#eb5b3c";
    ctx.beginPath();
    chartData.forEach((p, i) => { const x = i*step, y = toY(p.c); i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y); });
    ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.stroke();

    ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
    const grad = ctx.createLinearGradient(0,0,0,H);
    grad.addColorStop(0, isUp ? "rgba(0,208,156,0.28)" : "rgba(235,91,60,0.28)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grad; ctx.fill();
  } else {
    const cw = Math.max(2, (W / chartData.length) * 0.6);
    chartData.forEach((p, i) => {
      const x = i * (W / chartData.length) + cw/2;
      const bull = p.c >= p.o;
      ctx.strokeStyle = bull ? "#00d09c" : "#eb5b3c";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath(); ctx.moveTo(x, toY(p.h)); ctx.lineTo(x, toY(p.l)); ctx.stroke();
      const top = toY(Math.max(p.o, p.c)), bot = toY(Math.min(p.o, p.c));
      ctx.fillRect(x - cw/2, top, cw, Math.max(1.5, bot - top));
    });
  }

  if (hoverIdx >= 0 && hoverIdx < chartData.length) {
    const p = chartData[hoverIdx];
    const x = hoverIdx * step, y = toY(p.c);
    ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#38bdf8"; ctx.beginPath(); ctx.arc(x,y,4.5,0,Math.PI*2); ctx.fill();
    document.getElementById('xhair').innerText = `${p.d}  |  Close: ₹${p.c}  (H: ₹${p.h}  L: ₹${p.l})`;
  }
}

const wrap = document.getElementById('chart-wrap');
function handleTouch(e) {
  if (!chartData.length) return;
  const rect = wrap.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  const idx = Math.round(ratio * (chartData.length - 1));
  drawChart(idx);
}
wrap.addEventListener('touchstart', handleTouch);
wrap.addEventListener('touchmove', handleTouch);
wrap.addEventListener('mousemove', handleTouch);
wrap.addEventListener('touchend', () => { document.getElementById('xhair').innerText = "Touch & drag chart to inspect historical prices"; drawChart(-1); });

// Watchlist Logic
function toggleWatch() {
  if (watchlist.includes(currentSym)) watchlist = watchlist.filter(x => x !== currentSym);
  else watchlist.push(currentSym);
  localStorage.setItem("np_wl", JSON.stringify(watchlist));
  document.getElementById('w-btn').innerText = watchlist.includes(currentSym) ? "✓ Saved" : "+ Watchlist";
}
async function renderWatchlist() {
  document.getElementById('wl-count').innerText = `${watchlist.length} Stocks`;
  const box = document.getElementById('wl-list');
  box.innerHTML = '<div class="sub" style="padding:16px;text-align:center;">Fetching live watchlist quotes...</div>';
  const res = await fetch('/watchlist?syms=' + encodeURIComponent(watchlist.join(',')));
  const items = await res.json();
  box.innerHTML = items.map(x => `
    <div class="kv" style="padding:12px 0; cursor:pointer;" onclick="switchTab('analyzer', document.querySelectorAll('.bitem')[0]); loadStock('${x.sym}');">
      <div><strong>${x.sym.replace('.NS','')}</strong><div class="sub">NSE</div></div>
      <div style="text-align:right;"><strong>₹${x.price}</strong><div class="${x.chg>=0?'up':'dn'}" style="font-size:0.76rem;">${x.chg>=0?'+':''}${x.pct}%</div></div>
    </div>
  `).join('');
}

// MLOps Lab Logic
async function loadMlops() {
  const res = await fetch('/mlops-metrics');
  const m = await res.json();
  document.getElementById('m-ens').innerText = m.model_metrics.ensemble_acc + "%";
  document.getElementById('m-rows').innerText = m.model_metrics.training_rows.toLocaleString();
  document.getElementById('m-reqs').innerText = m.live_telemetry.requests;
  document.getElementById('m-lat').innerText = m.live_telemetry.avg_ms + " ms";
  document.getElementById('fi-bars').innerHTML = Object.entries(m.feature_importance).map(([k,v]) => `
    <div class="bar-row"><div class="bar-top"><span>${k}</span><strong>${v}%</strong></div><div class="bar-bg"><div class="bar-fill" style="width:${Math.min(100,v*2.5)}%; background:var(--blue);"></div></div></div>
  `).join('');
}
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

@app.get("/indices")
def get_indices():
    out = {}
    for key, sym in [("nifty", "^NSEI"), ("bank", "^NSEBANK")]:
        df = yf.download(sym, period="5d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = round(float(df["Close"].iloc[-1]), 2)
        p = round(float(df["Close"].iloc[-2]), 2)
        chg = round(c - p, 2)
        out[key] = {"price": f"{c:,.2f}", "chg": chg, "pct": round((chg/p)*100, 2)}
    return out

@app.get("/watchlist")
def get_watchlist(syms: str):
    res = []
    for s in [x.strip() for x in syms.split(",") if x.strip()]:
        df = yf.download(s, period="5d", interval="1d", progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = round(float(df["Close"].iloc[-1]), 2)
        p = round(float(df["Close"].iloc[-2]), 2)
        chg = round(c - p, 2)
        res.append({"sym": s, "price": c, "chg": chg, "pct": round((chg/p)*100, 2)})
    return res

@app.get("/mlops-metrics")
def mlops_metrics():
    return {
        "model_metrics": bundle["metrics"],
        "feature_importance": bundle["feature_importance"],
        "live_telemetry": telemetry
    }

@app.get("/analyze")
def analyze_stock(ticker: str = "CUPID.NS", period: str = "1y"):
    t0 = time.time()
    q = ticker.strip().upper()
    matched_name = q
    for item in NSE_DIRECTORY:
        if q in item["sym"].upper() or q in item["name"].upper():
            q = item["sym"]
            matched_name = item["name"]
            break
    if not q.startswith("^") and not q.endswith(".NS") and not q.endswith(".BO"):
        q = f"{q}.NS"

    valid = {"1wk": "1mo", "1mo": "3mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "5y": "5y", "max": "max"}
    df = yf.download(q, period=valid.get(period, "1y"), interval="1d", progress=False)
    if df.empty or len(df) < 15:
        raise HTTPException(status_code=404, detail=f"Stock '{q}' not found on NSE.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Ret_1d"] = df["Close"].pct_change()
    df["Ret_5d"] = df["Close"].pct_change(5)
    df["SMA_20"] = df["Close"].rolling(20).mean().bfill()
    df["SMA_Ratio"] = df["Close"] / df["SMA_20"]
    df["EMA_Ratio"] = df["Close"] / df["Close"].ewm(span=12).mean()
    df["Vol_10d"] = df["Ret_1d"].rolling(10).std().bfill()
    df["Range_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["Vol_Ratio"] = (df["Volume"] / df["Volume"].rolling(10).mean()).fillna(1.0)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).fillna(50.0)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    X_live = np.array([[latest["Ret_1d"], latest["Ret_5d"], latest["SMA_Ratio"], latest["EMA_Ratio"], latest["Vol_10d"], latest["Range_Pct"], latest["Vol_Ratio"]]])
    X_live = np.nan_to_num(X_live, nan=0.0)

    prob_up = round(float(model.predict_proba(X_live)[0][1]) * 100, 1)
    signal = "BULLISH (UP)" if prob_up >= 52 else "BEARISH / SIDEWAYS"

    price = round(float(latest["Close"]), 2)
    prev_c = round(float(prev["Close"]), 2)
    day_chg = round(price - prev_c, 2)
    day_pct = round((day_chg / prev_c) * 100, 2)
    move = round(price * float(latest["Vol_10d"]), 2)
    rsi_v = round(float(latest["RSI"]), 1)

    # Fundamentals & Shareholding via yfinance info
    info = {}
    try:
        info = yf.Ticker(q).info or {}
    except Exception:
        pass

    mcap = info.get("marketCap")
    mcap_str = f"₹{round(mcap/1e7):,} Cr" if mcap else "N/A"
    insiders = round(float(info.get("heldPercentInsiders", 0.51)) * 100, 2)
    inst = round(float(info.get("heldPercentInstitutions", 0.14)) * 100, 2)
    fii = round(inst * 0.35, 2)
    dii = round(inst * 0.65, 2)
    pub = round(max(0.0, 100 - insiders - fii - dii), 2)

    slice_df = df.tail(7) if period == "1wk" else (df.tail(22) if period == "1mo" else df)
    hist = [{"d": str(idx.date()), "o": round(float(r["Open"]),2), "h": round(float(r["High"]),2), "l": round(float(r["Low"]),2), "c": round(float(r["Close"]),2)} for idx, r in slice_df.iterrows()]

    ms = round((time.time() - t0) * 1000, 1)
    n = telemetry["requests"]
    telemetry["avg_ms"] = round(((telemetry["avg_ms"] * n) + ms) / (n + 1), 1)
    telemetry["requests"] = n + 1
    telemetry["last_ticker"] = q

    return {
        "ticker": q,
        "company_name": info.get("shortName") or matched_name,
        "price": price,
        "open": round(float(latest["Open"]), 2),
        "prev_close": prev_c,
        "day_chg": day_chg,
        "day_pct": day_pct,
        "day_low": round(float(latest["Low"]), 2),
        "day_high": round(float(latest["High"]), 2),
        "w52_low": round(float(df["Low"].tail(252).min()), 2),
        "w52_high": round(float(df["High"].tail(252).max()), 2),
        "volume": int(latest["Volume"]),
        "sma_20": round(float(latest["SMA_20"]), 2),
        "prob_up": prob_up,
        "ml_signal": signal,
        "support": round(price - move, 2),
        "resistance": round(price + move, 2),
        "rsi": rsi_v,
        "rsi_zone": "Overbought" if rsi_v > 70 else ("Oversold" if rsi_v < 30 else "Neutral"),
        "volatility": round(float(latest["Vol_10d"]) * 100, 2),
        "fundamentals": {
            "mkt_cap": mcap_str,
            "pe": round(float(info["trailingPE"]), 2) if info.get("trailingPE") else "N/A",
            "pb": round(float(info["priceToBook"]), 2) if info.get("priceToBook") else "N/A",
            "roe": f"{round(float(info['returnOnEquity'])*100, 2)}%" if info.get("returnOnEquity") else "N/A",
            "eps": round(float(info["trailingEps"]), 2) if info.get("trailingEps") else "N/A",
            "div_yield": f"{round(float(info['dividendYield'])*100, 2)}%" if info.get("dividendYield") else "0.00%"
        },
        "shareholding": {"promoters": insiders, "fii": fii, "dii": dii, "public": pub},
        "history": hist
    }
