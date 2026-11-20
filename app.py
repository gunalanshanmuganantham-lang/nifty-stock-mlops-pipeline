from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import requests
import time

app = FastAPI(title="NiftyPulse AI v5.0", version="5.0.0")
bundle = joblib.load("stock_model.pkl")
model = bundle["model"]

telemetry = {"requests": 0, "avg_ms": 0.0, "cache_hits": 0, "last_ticker": "CUPID.NS"}
CACHE = {}
CACHE_TTL = 300
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

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/popular")
def get_popular():
    return POPULAR_NSE

@app.get("/search")
def search_nse(q: str):
    q_clean = q.strip()
    if not q_clean:
        return POPULAR_NSE[:6]
    results = []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q_clean}&quotesCount=10&newsCount=0"
        r = requests.get(url, headers=HEADERS, timeout=3).json()
        for item in r.get("quotes", []):
            sym = item.get("symbol", "")
            if sym.endswith(".NS") or sym.endswith(".BO") or sym.startswith("^"):
                results.append({"sym": sym, "name": item.get("shortname") or item.get("longname") or sym, "sector": item.get("sector") or item.get("exchDisp") or "NSE Equity"})
    except Exception:
        pass
    if not results:
        ql = q_clean.lower()
        results = [x for x in POPULAR_NSE if ql in x["name"].lower() or ql in x["sym"].lower()]
    if not results:
        results = [{"sym": f"{q_clean.upper()}.NS", "name": f"{q_clean.upper()} (NSE Direct)", "sector": "NSE Equity"}]
    return results[:8]

@app.get("/indices")
def get_indices():
    if "indices" in CACHE and time.time() - CACHE["indices"]["ts"] < CACHE_TTL:
        return CACHE["indices"]["data"]
    idx_list = [("NIFTY 50", "^NSEI"), ("NIFTY NEXT 50", "^NSMIDCP"), ("BANK NIFTY", "^NSEBANK"), ("NIFTY IT", "^CNXIT"), ("NIFTY MIDCAP", "^NSEMDCP50"), ("INDIA VIX", "^INDIAVIX")]
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

def fetch_fundamentals_reliable(ticker_sym: str, price: float):
    mcap_str, pe_str, pb_str, roe_str, eps_str, div_str = "N/A", "N/A", "N/A", "N/A", "N/A", "0.00%"
    comp_name = ticker_sym.replace(".NS", "")
    try:
        mc = yf.Ticker(ticker_sym).fast_info.get("marketCap")
        if mc and mc > 0:
            mcap_str = f"₹{round(mc / 1e7):,} Cr"
    except Exception:
        pass
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker_sym}"
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
    except Exception:
        pass
    return comp_name, {"mkt_cap": mcap_str, "pe": pe_str, "pb": pb_str, "roe": roe_str, "eps": eps_str, "div_yield": div_str}, {"promoters": 52.0, "fii": 12.5, "dii": 14.5, "public": 21.0}

@app.get("/analyze")
def analyze_stock(ticker: str = "CUPID.NS", period: str = "1y"):
    t0 = time.time()
    q = ticker.strip().upper()
    if not q.startswith("^") and not q.endswith(".NS") and not q.endswith(".BO"):
        q = f"{q}.NS"
    cache_key = f"{q}_{period}"
    if cache_key in CACHE and (time.time() - CACHE[cache_key]["ts"]) < CACHE_TTL:
        telemetry["cache_hits"] += 1
        ms = round((time.time() - t0) * 1000, 1)
        n = telemetry["requests"]
        telemetry["avg_ms"] = round(((telemetry["avg_ms"] * n) + ms) / (n + 1), 1)
        telemetry["requests"] = n + 1
        return CACHE[cache_key]["data"]

    valid = {"1wk": "1mo", "1mo": "3mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "5y": "5y", "max": "max"}
    df = yf.download(q, period=valid.get(period, "1y"), interval="1d", progress=False)
    if df.empty or len(df) < 10:
        raise HTTPException(status_code=404, detail=f"Stock '{q}' not found on NSE.")
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
    move = round(price * float(latest["Vol_10d"]), 2)
    rsi_v = round(float(latest["RSI"]), 1)
    comp_name, funds, sh = fetch_fundamentals_reliable(q, price)

    slice_df = df.tail(7) if period == "1wk" else (df.tail(22) if period == "1mo" else df)
    hist = [{"d": str(idx.date()), "o": round(float(r["Open"]),2), "h": round(float(r["High"]),2), "l": round(float(r["Low"]),2), "c": round(float(r["Close"]),2)} for idx, r in slice_df.iterrows()]

    result = {
        "ticker": q, "company_name": comp_name, "price": price, "open": round(float(latest["Open"]), 2), "prev_close": prev_c,
        "day_chg": day_chg, "day_pct": day_pct, "day_low": round(float(latest["Low"]), 2), "day_high": round(float(latest["High"]), 2),
        "w52_low": round(float(df["Low"].tail(252).min()), 2), "w52_high": round(float(df["High"].tail(252).max()), 2),
        "volume": int(latest["Volume"]), "sma_20": round(float(latest["SMA_20"]), 2), "prob_up": prob_up, "ml_signal": signal,
        "support": round(price - move, 2), "resistance": round(price + move, 2), "rsi": rsi_v,
        "rsi_zone": "Overbought" if rsi_v > 70 else ("Oversold" if rsi_v < 30 else "Neutral"),
        "macd_signal": "Bullish Crossover" if float(latest["MACD"]) > 0 else "Bearish Momentum",
        "fundamentals": funds, "shareholding": sh, "history": hist
    }
    CACHE[cache_key] = {"ts": time.time(), "data": result}
    ms = round((time.time() - t0) * 1000, 1)
    n = telemetry["requests"]
    telemetry["avg_ms"] = round(((telemetry["avg_ms"] * n) + ms) / (n + 1), 1)
    telemetry["requests"] = n + 1
    telemetry["last_ticker"] = q
    return result
            
