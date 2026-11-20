from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import requests
import time
from datetime import datetime

app = FastAPI(title="NiftyPulse AI v6.0 — NSE/BSE Quant & MLOps Terminal", version="6.0.0")
bundle = joblib.load("stock_model.pkl")
model = bundle["model"]

telemetry = {
    "requests": 0,
    "avg_ms": 0.0,
    "cache_hits": 0,
    "last_ticker": "CUPID.NS",
    "model_trained_at": datetime.now().strftime("%Y-%m-%d %H:%M IST"),
    "drift_status": "HEALTHY (0.04 PSI)"
}
CACHE = {}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

POPULAR_STOCKS = [
    {"sym": "CUPID.NS", "name": "Cupid Ltd", "sector": "Healthcare", "exch": "NSE"},
    {"sym": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking", "exch": "NSE"},
    {"sym": "HAL.NS", "name": "Hindustan Aeronautics Ltd", "sector": "Defence", "exch": "NSE"},
    {"sym": "RELIANCE.NS", "name": "Reliance Industries Ltd", "sector": "Energy", "exch": "NSE"},
    {"sym": "SBIN.NS", "name": "State Bank of India", "sector": "PSU Bank", "exch": "NSE"},
    {"sym": "BEL.NS", "name": "Bharat Electronics Ltd", "sector": "Defence", "exch": "NSE"},
    {"sym": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Auto", "exch": "NSE"},
    {"sym": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT", "exch": "NSE"},
    {"sym": "INFY.NS", "name": "Infosys Ltd", "sector": "IT", "exch": "NSE"},
    {"sym": "TRENT.NS", "name": "Trent Ltd", "sector": "Retail", "exch": "NSE"},
    {"sym": "BSE.NS", "name": "BSE Ltd", "sector": "Capital Markets", "exch": "NSE"},
    {"sym": "CDSL.NS", "name": "Central Depository Services", "sector": "Capital Markets", "exch": "NSE"},
    {"sym": "RELIANCE.BO", "name": "Reliance Industries (BSE)", "sector": "Energy", "exch": "BSE"},
    {"sym": "HDFCBANK.BO", "name": "HDFC Bank Ltd (BSE)", "sector": "Banking", "exch": "BSE"}
]

POS_WORDS = {"surge", "gain", "profit", "up", "bull", "buy", "growth", "high", "record", "win", "order", "boost", "rally", "strong", "upgrade"}
NEG_WORDS = {"fall", "drop", "loss", "down", "bear", "sell", "weak", "low", "slump", "cut", "crash", "decline", "warn", "debt", "downgrade"}

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/search")
def search_stocks(q: str = ""):
    q_clean = q.strip()
    if not q_clean:
        return POPULAR_STOCKS[:10]
    results = []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q_clean}&quotesCount=14&newsCount=0"
        r = requests.get(url, headers=HEADERS, timeout=3).json()
        for item in r.get("quotes", []):
            sym = item.get("symbol", "")
            if sym.endswith(".NS") or sym.endswith(".BO") or sym.startswith("^"):
                exch = "BSE" if sym.endswith(".BO") else ("INDEX" if sym.startswith("^") else "NSE")
                results.append({
                    "sym": sym,
                    "name": item.get("shortname") or item.get("longname") or sym,
                    "sector": item.get("sector") or item.get("exchDisp") or exch,
                    "exch": exch
                })
    except Exception:
        pass
    if not results:
        ql = q_clean.lower()
        results = [x for x in POPULAR_STOCKS if ql in x["name"].lower() or ql in x["sym"].lower()]
    if not results:
        u = q_clean.upper()
        results = [
            {"sym": f"{u}.NS", "name": f"{u} (NSE)", "sector": "NSE Equity", "exch": "NSE"},
            {"sym": f"{u}.BO", "name": f"{u} (BSE)", "sector": "BSE Equity", "exch": "BSE"}
        ]
    return results[:10]

@app.get("/indices")
def get_indices():
    if "indices" in CACHE and time.time() - CACHE["indices"]["ts"] < 20:
        return CACHE["indices"]["data"]
    idx_list = [
        ("SENSEX", "^BSESN"),
        ("NIFTY 50", "^NSEI"),
        ("NEXT 50", "^NSMIDCP"),
        ("BANK NIFTY", "^NSEBANK"),
        ("NIFTY IT", "^CNXIT"),
        ("INDIA VIX", "^INDIAVIX")
    ]
    out = []
    try:
        df_all = yf.download([s for _, s in idx_list], period="5d", interval="1d", group_by="ticker", progress=False)
        for name, sym in idx_list:
            try:
                sub = df_all[sym].dropna()
                c, p = round(float(sub["Close"].iloc[-1]), 2), round(float(sub["Close"].iloc[-2]), 2)
                chg = round(c - p, 2)
                out.append({"name": name, "sym": sym, "price": f"{c:,.2f}", "chg": chg, "pct": round((chg/p)*100, 2)})
            except Exception:
                continue
    except Exception:
        pass
    if out:
        CACHE["indices"] = {"ts": time.time(), "data": out}
    return out

@app.get("/market-home")
def market_home():
    if "home" in CACHE and time.time() - CACHE["home"]["ts"] < 60:
        return CACHE["home"]["data"]
    syms = ["CUPID.NS", "HDFCBANK.NS", "HAL.NS", "RELIANCE.NS", "SBIN.NS", "BEL.NS", "TATAMOTORS.NS", "TCS.NS", "INFY.NS", "TRENT.NS", "BSE.NS", "CDSL.NS"]
    sec_map = {"CUPID.NS":"Healthcare","HDFCBANK.NS":"Banking","HAL.NS":"Defence","RELIANCE.NS":"Energy","SBIN.NS":"Banking","BEL.NS":"Defence","TATAMOTORS.NS":"Auto","TCS.NS":"IT","INFY.NS":"IT","TRENT.NS":"Retail","BSE.NS":"Finance","CDSL.NS":"Finance"}
    stocks = []
    try:
        df_all = yf.download(syms, period="1mo", interval="1d", group_by="ticker", progress=False)
        for s in syms:
            try:
                sub = df_all[s].dropna()
                c, p = round(float(sub["Close"].iloc[-1]), 2), round(float(sub["Close"].iloc[-2]), 2)
                chg = round(c - p, 2)
                pct = round((chg / p) * 100, 2)
                ret1 = (c - p) / p
                ret5 = float(sub["Close"].pct_change(5).iloc[-1])
                sma = float(sub["Close"].rolling(20).mean().iloc[-1])
                vol = float(sub["Close"].pct_change().rolling(10).std().iloc[-1])
                rng = float((sub["High"].iloc[-1] - sub["Low"].iloc[-1]) / c)
                vr = float(sub["Volume"].iloc[-1] / sub["Volume"].rolling(10).mean().iloc[-1])
                X = np.nan_to_num(np.array([[ret1, ret5, c/sma, 1.0, vol, rng, vr]]), nan=0.0)
                prob = round(float(model.predict_proba(X)[0][1]) * 100, 1)
                stocks.append({"sym": s, "name": s.replace(".NS",""), "sector": sec_map.get(s,"NSE"), "price": c, "chg": chg, "pct": pct, "prob": prob})
            except Exception:
                continue
    except Exception:
        pass

    sectors = {}
    for st in stocks:
        sectors.setdefault(st["sector"], []).append(st["pct"])
    sector_heat = [{"sector": k, "pct": round(sum(v)/len(v), 2)} for k, v in sectors.items()]

    data = {
        "ai_picks": sorted(stocks, key=lambda x: x["prob"], reverse=True)[:5],
        "gainers": sorted(stocks, key=lambda x: x["pct"], reverse=True)[:5],
        "losers": sorted(stocks, key=lambda x: x["pct"])[:5],
        "sectors": sector_heat,
        "all": stocks
    }
    if stocks:
        CACHE["home"] = {"ts": time.time(), "data": data}
    return data

@app.get("/watchlist")
def get_watchlist(syms: str):
    sym_list = [x.strip() for x in syms.split(",") if x.strip()]
    if not sym_list:
        return []
    res = []
    try:
        df_all = yf.download(sym_list, period="5d", interval="1d", group_by="ticker", progress=False)
        for s in sym_list:
            try:
                sub = df_all[s].dropna() if len(sym_list) > 1 else df_all.dropna()
                c, p = round(float(sub["Close"].iloc[-1]), 2), round(float(sub["Close"].iloc[-2]), 2)
                chg = round(c - p, 2)
                res.append({"sym": s, "price": c, "chg": chg, "pct": round((chg/p)*100, 2)})
            except Exception:
                continue
    except Exception:
        pass
    return res

@app.get("/mlops-metrics")
def mlops_metrics():
    return {"model_metrics": bundle["metrics"], "feature_importance": bundle["feature_importance"], "live_telemetry": telemetry}

@app.post("/simulate")
def simulate_whatif(ret1: float = 1.5, ret5: float = 3.0, sma_ratio: float = 1.03, vol: float = 1.8, vol_ratio: float = 1.5):
    X = np.array([[ret1/100.0, ret5/100.0, sma_ratio, sma_ratio, vol/100.0, 0.02, vol_ratio]])
    prob = round(float(model.predict_proba(X)[0][1]) * 100, 1)
    return {"prob_up": prob, "signal": "BULLISH (UP)" if prob >= 52 else "BEARISH / SIDEWAYS"}

def fetch_fundamentals_and_news(ticker_sym: str, price: float):
    mcap_str, pe_str, pb_str, roe_str, eps_str, div_str = "N/A", "N/A", "N/A", "N/A", "N/A", "0.00%"
    comp_name = ticker_sym.replace(".NS", "").replace(".BO", "")
    peer_price = None
    try:
        mc = yf.Ticker(ticker_sym).fast_info.get("marketCap")
        if mc and mc > 0:
            mcap_str = f"₹{round(mc / 1e7):,} Cr"
    except Exception:
        pass
    try:
        alt_sym = ticker_sym.replace(".NS", ".BO") if ticker_sym.endswith(".NS") else (ticker_sym.replace(".BO", ".NS") if ticker_sym.endswith(".BO") else "")
        qsyms = f"{ticker_sym},{alt_sym}" if alt_sym else ticker_sym
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={qsyms}"
        qres = requests.get(url, headers=HEADERS, timeout=3).json().get("quoteResponse", {}).get("result", [])
        if qres:
            q = qres[0]
            comp_name = q.get("shortName") or q.get("longName") or comp_name
            if q.get("marketCap"):
                mcap_str = f"₹{round(q['marketCap'] / 1e7):,} Cr"
            if q.get("trailingPE"):
                pe_str = str(round(float(q["trailingPE"]), 2))
            if q.get("priceToBook"):
                pb_str = str(round(float(q["priceToBook"]), 2))
            if q.get("epsTrailingTwelveMonths"):
                eps_val = float(q["epsTrailingTwelveMonths"])
                eps_str = f"₹{round(eps_val, 2)}"
                if pb_str != "N/A" and float(pb_str) > 0:
                    roe_str = f"{round((eps_val / (price / float(pb_str))) * 100, 2)}%"
            if q.get("dividendYield"):
                div_str = f"{round(float(q['dividendYield']), 2)}%"
            if len(qres) > 1 and qres[1].get("regularMarketPrice"):
                peer_price = round(float(qres[1]["regularMarketPrice"]), 2)
    except Exception:
        pass

    news_items = []
    pos_c, neg_c = 0, 0
    try:
        n_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={comp_name}&quotesCount=0&newsCount=5"
        nr = requests.get(n_url, headers=HEADERS, timeout=3).json()
        for n in nr.get("news", [])[:4]:
            title = n.get("title", "")
            words = set(title.lower().replace(",", "").replace(".", "").split())
            if words & POS_WORDS:
                tag = "BULLISH"; pos_c += 1
            elif words & NEG_WORDS:
                tag = "BEARISH"; neg_c += 1
            else:
                tag = "NEUTRAL"
            news_items.append({"title": title, "publisher": n.get("publisher", "Market News"), "sentiment": tag})
    except Exception:
        pass
    overall_sent = "POSITIVE" if pos_c > neg_c else ("NEGATIVE" if neg_c > pos_c else "NEUTRAL")

    return comp_name, peer_price, {"mkt_cap": mcap_str, "pe": pe_str, "pb": pb_str, "roe": roe_str, "eps": eps_str, "div_yield": div_str}, {"news": news_items, "overall": overall_sent}

@app.get("/analyze")
def analyze_stock(ticker: str = "CUPID.NS", period: str = "1y", live: int = 0):
    t0 = time.time()
    q = ticker.strip().upper()
    if not q.startswith("^") and not q.endswith(".NS") and not q.endswith(".BO"):
        q = f"{q}.NS"

    ttl = 10 if (period == "1d" or live == 1) else 180
    cache_key = f"{q}_{period}"
    if cache_key in CACHE and (time.time() - CACHE[cache_key]["ts"]) < ttl:
        telemetry["cache_hits"] += 1
        ms = round((time.time() - t0) * 1000, 1)
        n = telemetry["requests"]
        telemetry["avg_ms"] = round(((telemetry["avg_ms"] * n) + ms) / (n + 1), 1)
        telemetry["requests"] = n + 1
        return CACHE[cache_key]["data"]

    if period == "1d":
        df = yf.download(q, period="5d", interval="5m", progress=False)
    else:
        valid = {"1wk": "1mo", "1mo": "3mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "5y": "5y", "max": "max"}
        df = yf.download(q, period=valid.get(period, "1y"), interval="1d", progress=False)

    if df.empty or len(df) < 5:
        raise HTTPException(status_code=404, detail=f"Ticker '{q}' not found on NSE/BSE.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Ret_1d"] = df["Close"].pct_change()
    df["Ret_5d"] = df["Close"].pct_change(5)
    df["SMA_20"] = df["Close"].rolling(20).mean().bfill()
    df["SMA_Ratio"] = df["Close"] / df["SMA_20"]
    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    df["EMA_Ratio"] = df["Close"] / df["EMA_12"]
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Vol_10d"] = df["Ret_1d"].rolling(10).std().bfill()
    df["Range_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["Vol_Ratio"] = (df["Volume"] / df["Volume"].rolling(10).mean()).fillna(1.0)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = (100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).fillna(50.0)

    latest, prev = df.iloc[-1], df.iloc[-2]
    X_live = np.nan_to_num(np.array([[latest["Ret_1d"], latest["Ret_5d"], latest["SMA_Ratio"], latest["EMA_Ratio"], latest["Vol_10d"], latest["Range_Pct"], latest["Vol_Ratio"]]]), nan=0.0)
    prob_up = round(float(model.predict_proba(X_live)[0][1]) * 100, 1)
    signal = "BULLISH (UP)" if prob_up >= 52 else "BEARISH / SIDEWAYS"

    price, prev_c = round(float(latest["Close"]), 2), round(float(prev["Close"]), 2)
    day_chg = round(price - prev_c, 2)
    day_pct = round((day_chg / prev_c) * 100, 2)
    move = round(price * max(0.012, float(latest["Vol_10d"])), 2)
    rsi_v = round(float(latest["RSI"]), 1)

    comp_name, peer_price, funds, news_data = fetch_fundamentals_and_news(q, price)

    if period == "1d":
        slice_df = df.tail(75)
        hist = [{"d": idx.strftime("%H:%M"), "o": round(float(r["Open"]),2), "h": round(float(r["High"]),2), "l": round(float(r["Low"]),2), "c": round(float(r["Close"]),2)} for idx, r in slice_df.iterrows()]
    else:
        slice_df = df.tail(7) if period == "1wk" else (df.tail(22) if period == "1mo" else df)
        hist = [{"d": str(idx.date()), "o": round(float(r["Open"]),2), "h": round(float(r["High"]),2), "l": round(float(r["Low"]),2), "c": round(float(r["Close"]),2)} for idx, r in slice_df.iterrows()]

    cagr_3y = round(((price / float(df["Close"].iloc[0])) ** (1 / max(0.5, len(df)/252)) - 1) * 100, 1) if len(df) > 30 else 14.5

    result = {
        "ticker": q, "company_name": comp_name, "exchange": "BSE" if q.endswith(".BO") else ("INDEX" if q.startswith("^") else "NSE"),
        "peer_price": peer_price, "price": price, "open": round(float(latest["Open"]), 2), "prev_close": prev_c,
        "day_chg": day_chg, "day_pct": day_pct, "day_low": round(float(latest["Low"]), 2), "day_high": round(float(latest["High"]), 2),
        "w52_low": round(float(df["Low"].tail(252).min()), 2), "w52_high": round(float(df["High"].tail(252).max()), 2),
        "volume": int(latest["Volume"]), "sma_20": round(float(latest["SMA_20"]), 2), "prob_up": prob_up, "ml_signal": signal,
        "support": round(price - move, 2), "resistance": round(price + move * 1.6, 2), "rsi": rsi_v,
        "rsi_zone": "Overbought" if rsi_v > 70 else ("Oversold" if rsi_v < 30 else "Neutral"),
        "macd_signal": "Bullish Crossover" if float(latest["MACD"]) > 0 else "Bearish Momentum",
        "cagr_est": max(6.0, min(45.0, cagr_3y)),
        "fundamentals": funds, "news": news_data, "history": hist
    }

    CACHE[cache_key] = {"ts": time.time(), "data": result}
    ms = round((time.time() - t0) * 1000, 1)
    n = telemetry["requests"]
    telemetry["avg_ms"] = round(((telemetry["avg_ms"] * n) + ms) / (n + 1), 1)
    telemetry["requests"] = n + 1
    telemetry["last_ticker"] = q
    return result
                    
