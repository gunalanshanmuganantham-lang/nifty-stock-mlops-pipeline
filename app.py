from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import requests
import time

app = FastAPI(title="NiftyPulse AI v5.0 — Pro Quant & MLOps Terminal", version="5.0.0")
bundle = joblib.load("stock_model.pkl")
model = bundle["model"]

telemetry = {"requests": 0, "avg_ms": 0.0, "cache_hits": 0, "last_ticker": "HDFCBANK.NS"}
CACHE = {}
CACHE_TTL = 300  # 5 mins ultra-fast memory cache

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

POPULAR_NSE = [
    {"sym": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking"},
    {"sym": "RELIANCE.NS", "name": "Reliance Industries Ltd", "sector": "Energy & Telecom"},
    {"sym": "HAL.NS", "name": "Hindustan Aeronautics Ltd", "sector": "Defence"},
    {"sym": "SBIN.NS", "name": "State Bank of India", "sector": "PSU Bank"},
    {"sym": "CUPID.NS", "name": "Cupid Ltd", "sector": "Healthcare"},
    {"sym": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Automobile"},
    {"sym": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT"},
    {"sym": "INFY.NS", "name": "Infosys Ltd", "sector": "IT"},
    {"sym": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking"},
    {"sym": "BEL.NS", "name": "Bharat Electronics Ltd", "sector": "Defence"},
    {"sym": "TRENT.NS", "name": "Trent Ltd", "sector": "Retail"},
    {"sym": "IRCTC.NS", "name": "IRCTC Ltd", "sector": "Railways"},
    {"sym": "ZOMATO.NS", "name": "Zomato Ltd", "sector": "Consumer Tech"},
    {"sym": "CDSL.NS", "name": "Central Depository Services", "sector": "Capital Markets"}
]

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>NiftyPulse AI v5.0 | Quant & MLOps Terminal</title>
<style>
  :root { --bg:#090c12; --card:#121722; --card2:#192130; --border:#222d40; --green:#00d09c; --red:#eb5b3c; --text:#f8fafc; --muted:#8b9bb4; --blue:#38bdf8; --gold:#f59e0b; }
  * { box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif; -webkit-tap-highlight-color:transparent; }
  body { background:var(--bg); color:var(--text); margin:0; padding:0 0 78px; display:flex; justify-content:center; }
  .shell { max-width:480px; width:100%; padding:10px 12px; }

  /* Lock Screen */
  #lock-screen { position:fixed; inset:0; background:#070a0f; z-index:999; display:none; flex-direction:column; align-items:center; justify-content:center; padding:24px; }
  .pin-dots { display:flex; gap:14px; margin:18px 0; }
  .dot { width:13px; height:13px; border-radius:50%; border:2px solid var(--green); }
  .dot.filled { background:var(--green); }
  .keypad { display:grid; grid-template-columns:repeat(3,70px); gap:14px; margin-top:8px; }
  .key { height:58px; border-radius:50%; background:var(--card); border:1px solid var(--border); color:#fff; font-size:1.25rem; font-weight:700; cursor:pointer; }
  .key:active { background:var(--green); color:#000; }

  /* Multi-Index Horizontal Ticker Strip */
  .idx-strip { display:flex; gap:8px; overflow-x:auto; padding-bottom:6px; margin-bottom:10px; scrollbar-width:none; }
  .idx-strip::-webkit-scrollbar { display:none; }
  .idx-pill { background:var(--card); border:1px solid var(--border); border-radius:11px; padding:7px 11px; min-width:128px; flex-shrink:0; cursor:pointer; }
  .idx-pill:hover { border-color:var(--green); }
  .idx-name { font-size:0.68rem; color:var(--muted); font-weight:700; letter-spacing:0.3px; }
  .idx-num { font-size:0.84rem; font-weight:800; margin-top:2px; }
  .idx-chg { font-size:0.7rem; font-weight:600; margin-top:1px; }

  /* Universal Live Search */
  .search-wrap { position:relative; margin-bottom:10px; }
  .search-box { display:flex; background:var(--card); border:1px solid var(--border); border-radius:12px; padding:3px 6px; align-items:center; }
  .search-box input { flex:1; background:transparent; border:none; color:#fff; padding:9px; font-size:0.9rem; outline:none; }
  .btn-go { background:var(--green); color:#042f24; border:none; border-radius:8px; padding:8px 13px; font-weight:700; font-size:0.82rem; cursor:pointer; }
  .dropdown { position:absolute; top:46px; left:0; right:0; background:#151d2b; border:1px solid var(--border); border-radius:12px; z-index:85; max-height:250px; overflow-y:auto; display:none; box-shadow:0 14px 32px rgba(0,0,0,0.8); }
  .dd-item { padding:10px 14px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; cursor:pointer; }
  .dd-item:hover { background:#1f2a3d; }

  /* Cards & Layout */
  .card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:15px; margin-bottom:11px; }
  .hdr-row { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }
  .title { font-size:1.12rem; font-weight:700; }
  .sub { font-size:0.74rem; color:var(--muted); margin-top:2px; }
  .price-big { font-size:1.68rem; font-weight:800; margin:6px 0 2px; }
  .up { color:var(--green); } .dn { color:var(--red); }

  /* Zoomable Chart */
  .chart-top { display:flex; justify-content:space-between; align-items:center; margin-top:6px; gap:6px; }
  .crosshair-info { font-size:0.74rem; color:var(--blue); font-weight:600; flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .zoom-btns { display:flex; gap:4px; }
  .zbtn { background:var(--card2); border:1px solid var(--border); color:var(--text); border-radius:6px; padding:3px 8px; font-size:0.74rem; font-weight:700; cursor:pointer; }
  .zbtn:active { background:var(--green); color:#000; }
  #chart-wrap { position:relative; width:100%; height:195px; touch-action:none; margin:6px 0 8px; cursor:crosshair; }
  canvas { width:100%; height:100%; display:block; }
  .tf-bar { display:flex; justify-content:space-between; align-items:center; background:var(--bg); padding:4px; border-radius:10px; }
  .tf-btns { display:flex; gap:1px; flex:1; justify-content:space-around; }
  .tf { background:transparent; border:none; color:var(--muted); font-size:0.72rem; font-weight:700; padding:6px 7px; border-radius:7px; cursor:pointer; }
  .tf.active { background:var(--card2); color:var(--green); }
  .mode-btn { background:var(--card2); border:1px solid var(--border); color:var(--green); border-radius:7px; padding:5px 8px; font-size:0.72rem; font-weight:700; cursor:pointer; margin-left:4px; }

  /* Sliders & Grids */
  .slider-block { margin:11px 0; }
  .sl-labels { display:flex; justify-content:space-between; font-size:0.74rem; color:var(--muted); margin-bottom:3px; }
  .sl-vals { display:flex; justify-content:space-between; font-size:0.84rem; font-weight:700; margin-bottom:6px; }
  .track { position:relative; height:5px; background:#263145; border-radius:4px; }
  .pointer { position:absolute; top:-4px; width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:8px solid var(--text); transform:translateX(-6px); }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .kv { display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid var(--border); font-size:0.81rem; }
  .kv:last-child { border-bottom:none; }
  .bar-row { margin:9px 0; }
  .bar-top { display:flex; justify-content:space-between; font-size:0.79rem; margin-bottom:4px; }
  .bar-bg { height:7px; background:#222b3c; border-radius:5px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:5px; }

  /* Sub-tabs & Bottom Nav */
  .sub-tabs { display:flex; border-bottom:1px solid var(--border); margin-bottom:11px; gap:16px; }
  .stab { padding:8px 2px; font-size:0.84rem; font-weight:700; color:var(--muted); border-bottom:2px solid transparent; cursor:pointer; }
  .stab.active { color:var(--text); border-bottom-color:var(--green); }
  .bnav { position:fixed; bottom:0; left:0; right:0; background:#10151f; border-top:1px solid var(--border); display:flex; justify-content:space-around; padding:8px 0 10px; z-index:90; }
  .bitem { background:none; border:none; color:var(--muted); font-size:0.71rem; font-weight:600; display:flex; flex-direction:column; align-items:center; gap:3px; cursor:pointer; }
  .bitem.active { color:var(--green); }
  .bicon { font-size:1.12rem; }

  /* Switch */
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
    <button class="key" style="font-size:0.85rem;" onclick="clearPin()">CLR</button><button class="key" onclick="pressPin('0')">0</button><button class="key" style="font-size:0.9rem;" onclick="bioUnlock()">👆</button>
  </div>
</div>

<div class="shell">
  <!-- LIVE MULTI-INDEX STRIP (NIFTY 50, NEXT 50, BANK NIFTY, IT, MIDCAP, VIX) -->
  <div class="idx-strip" id="idx-strip">
    <div class="idx-pill"><div class="idx-name">SYNCING INDICES...</div><div class="idx-num">NSE Live</div></div>
  </div>

  <!-- TAB 1: ANALYZER -->
  <div id="tab-analyzer">
    <div class="search-wrap">
      <div class="search-box">
        <input type="text" id="q" placeholder="Search ANY NSE stock (MRF, BEL, Trent, Cupid...)" oninput="liveSearch()" onfocus="liveSearch()">
        <button class="btn-go" onclick="loadStock(document.getElementById('q').value)">Go</button>
      </div>
      <div class="dropdown" id="dd"></div>
    </div>

    <div id="loader" style="text-align:center; padding:26px; color:var(--green); font-size:0.88rem; display:none;">⚡ Fetching NSE Data & Quant Signals...</div>

    <div id="stock-body" style="display:none;">
      <div class="card">
        <div class="hdr-row">
          <div>
            <div class="title" id="s-name"></div>
            <div class="sub" id="s-sym"></div>
          </div>
          <div style="display:flex; gap:6px; align-items:center;">
            <span id="lat-pill" style="font-size:0.68rem; color:var(--blue); background:rgba(56,189,248,0.12); padding:4px 7px; border-radius:6px; font-weight:700;"></span>
            <button onclick="toggleWatch()" id="w-btn" style="background:var(--card2); border:1px solid var(--border); color:var(--green); padding:6px 10px; border-radius:8px; font-size:0.75rem; font-weight:700; cursor:pointer;">+ Watch</button>
          </div>
        </div>
        <div class="price-big" id="s-price"></div>
        <div style="font-size:0.83rem; font-weight:600;" id="s-chg"></div>

        <div class="chart-top">
          <div class="crosshair-info" id="xhair">Pinch or use +/- to zoom • Drag to inspect</div>
          <div class="zoom-btns">
            <button class="zbtn" onclick="zoomChart(0.7)">+</button>
            <button class="zbtn" onclick="zoomChart(1.4)">−</button>
            <button class="zbtn" onclick="resetZoom()">Reset</button>
          </div>
        </div>
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

      <!-- SUB-TABS -->
      <div class="sub-tabs">
        <div class="stab active" onclick="switchSub('ov',this)">Overview</div>
        <div class="stab" onclick="switchSub('ai',this)">AI Quant & Tech</div>
        <div class="stab" onclick="switchSub('sh',this)">Shareholding</div>
      </div>

      <!-- SUB 1: OVERVIEW -->
      <div id="sub-ov">
        <div class="card">
          <div style="font-weight:700; font-size:0.94rem; margin-bottom:8px;">Performance</div>
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
          <div class="grid2" style="margin-top:12px; padding-top:10px; border-top:1px solid var(--border);">
            <div class="kv"><span style="color:var(--muted);">Open</span><strong id="f-open"></strong></div>
            <div class="kv"><span style="color:var(--muted);">Prev. Close</span><strong id="f-prev"></strong></div>
            <div class="kv"><span style="color:var(--muted);">Volume</span><strong id="f-vol"></strong></div>
            <div class="kv"><span style="color:var(--muted);">20-DMA</span><strong id="f-sma"></strong></div>
          </div>
        </div>

        <div class="card">
          <div style="font-weight:700; font-size:0.94rem; margin-bottom:6px;">Fundamentals</div>
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
              <div style="font-weight:700; color:var(--blue); font-size:0.95rem;">🤖 Ensemble ML & Technical Setup</div>
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
              <div class="sub">Stop-Loss (Support)</div><div class="dn" style="font-weight:700; margin-top:3px;" id="ai-sup"></div>
            </div>
            <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);">
              <div class="sub">Target (Resistance)</div><div class="up" style="font-weight:700; margin-top:3px;" id="ai-res"></div>
            </div>
            <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);">
              <div class="sub">14-Day RSI</div><div style="font-weight:700; margin-top:3px;" id="ai-rsi"></div>
            </div>
            <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);">
              <div class="sub">MACD Momentum</div><div style="font-weight:700; margin-top:3px;" id="ai-macd"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- SUB 3: SHAREHOLDING -->
      <div id="sub-sh" style="display:none;">
        <div class="card">
          <div style="font-weight:700; font-size:0.94rem; margin-bottom:10px;">Shareholding Pattern</div>
          <div class="bar-row"><div class="bar-top"><span>Promoters</span><strong id="sh-prom"></strong></div><div class="bar-bg"><div class="bar-fill" id="b-prom" style="background:#818cf8;"></div></div></div>
          <div class="bar-row"><div class="bar-top"><span>FIIs (Foreign Inst.)</span><strong id="sh-fii"></strong></div><div class="bar-bg"><div class="bar-fill" id="b-fii" style="background:#38bdf8;"></div></div></div>
          <div class="bar-row"><div class="bar-top"><span>DIIs (Mutual Funds)</span><strong id="sh-dii"></strong></div><div class="bar-bg"><div class="bar-fill" id="b-dii" style="background:#00d09c;"></div></div></div>
          <div class="bar-row"><div class="bar-top"><span>Retail & Public</span><strong id="sh-pub"></strong></div><div class="bar-bg"><div class="bar-fill" id="b-pub" style="background:#fb923c;"></div></div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: WATCHLIST -->
  <div id="tab-watchlist" style="display:none;">
    <div class="card">
      <div class="hdr-row" style="margin-bottom:10px;">
        <div class="title">My Watchlist</div>
        <span class="sub" id="wl-count"></span>
      </div>
      <div id="wl-list"></div>
    </div>
  </div>

  <!-- TAB 3: MLOPS LAB -->
  <div id="tab-mlops" style="display:none;">
    <div class="card" style="border-color:rgba(56,189,248,0.35);">
      <div class="title" style="color:var(--blue);">⚙️ Production MLOps Engine</div>
      <div class="sub">Ensemble Model Registry, Cache & Latency Telemetry</div>
      <div class="grid2" style="margin-top:12px;">
        <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);"><div class="sub">Ensemble Accuracy</div><div class="up" style="font-weight:800; font-size:1.1rem;" id="m-ens"></div></div>
        <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--border);"><div class="sub">Cache Hit Speed</div><div class="up" style="font-weight:800; font-size:1.1rem;" id="m-hits"></div></div>
        <div style="background:var(--bg); padding:10px; border-radius:10px; border:1px solid var(--
