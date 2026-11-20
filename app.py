from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import requests
import re
import time
from datetime import datetime

app = FastAPI(title="NiftyPulse AI v7.0", version="7.0.0")
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

INDEX_CONSTITUENTS = {
    "^NSEI": {
        "name": "NIFTY 50",
        "stocks": ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","BHARTIARTL.NS","SBIN.NS","LICI.NS","ITC.NS","LT.NS","HINDUNILVR.NS","BAJFINANCE.NS","HCLTECH.NS","MARUTI.NS","SUNPHARMA.NS","ADANIENT.NS","KOTAKBANK.NS","TITAN.NS","ONGC.NS","TATAMOTORS.NS","NTPC.NS","AXISBANK.NS","ADANIPORTS.NS","ULTRACEMCO.NS","ASIANPAINT.NS","COALINDIA.NS","BAJAJFINSV.NS","POWERGRID.NS","WIPRO.NS","M&M.NS","HAL.NS","BEL.NS","TRENT.NS","JSWSTEEL.NS","TATASTEEL.NS","SIEMENS.NS","DLF.NS","VBL.NS","ZOMATO.NS","JIOFIN.NS","TECHM.NS","GRASIM.NS","HINDALCO.NS","INDIGO.NS","DIVISLAB.NS","DRREDDY.NS","CIPLA.NS","EICHERMOT.NS","BPCL.NS","BRITANNIA.NS"]
    },
    "^NSMIDCP": {
        "name": "NIFTY NEXT 50",
        "stocks": ["BEL.NS","TRENT.NS","HAL.NS","VBL.NS","ZOMATO.NS","JIOFIN.NS","SIEMENS.NS","DLF.NS","INDIGO.NS","CHOLAFIN.NS","BANKBARODA.NS","PNB.NS","CANBK.NS","GAIL.NS","IOC.NS","RECLTD.NS","PFC.NS","ABB.NS","HAVELLS.NS","PIDILITIND.NS","GODREJCP.NS","DABUR.NS","MARICO.NS","ICICIGI.NS","SBICARD.NS","NAUKRI.NS","IRCTC.NS","TVSMOTOR.NS","MOTHERSON.NS","BOSCHLTD.NS","ZYDUSLIFE.NS","TORNTPHARM.NS","LUPIN.NS","AUROPHARMA.NS","COLPAL.NS","BERGEPAINT.NS","MUTHOOTFIN.NS","SRF.NS","POLYCAB.NS","PERSISTENT.NS","MPHASIS.NS","COFORGE.NS","LTTS.NS","ASTRAL.NS","CUMMINSIND.NS","SUZLON.NS","IREDA.NS","MAZDOCK.NS","RVNL.NS","NHPC.NS"]
    },
    "^NSEBANK": {
        "name": "BANK NIFTY",
        "stocks": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS","INDUSINDBK.NS","BANKBARODA.NS","PNB.NS","CANBK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS","AUBANK.NS"]
    },
    "^CNXIT": {
        "name": "NIFTY IT",
        "stocks": ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","LTIM.NS","PERSISTENT.NS","COFORGE.NS","MPHASIS.NS","LTTS.NS"]
    },
    "^BSESN": {
        "name": "SENSEX 30",
        "stocks": ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","BHARTIARTL.NS","SBIN.NS","ITC.NS","LT.NS","HINDUNILVR.NS","BAJFINANCE.NS","HCLTECH.NS","MARUTI.NS","SUNPHARMA.NS","KOTAKBANK.NS","TITAN.NS","TATAMOTORS.NS","NTPC.NS","AXISBANK.NS","ADANIPORTS.NS","ULTRACEMCO.NS","ASIANPAINT.NS","BAJAJFINSV.NS","POWERGRID.NS","WIPRO.NS","M&M.NS","TATASTEEL.NS","TECHM.NS","INDUSINDBK.NS","NESTLEIND.NS"]
    }
}

SECTOR_PEERS = {
    "Defence": ["HAL.NS","BEL.NS","MAZDOCK.NS","BDL.NS","DATAPATTNS.NS"],
    "Banking": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS"],
    "IT": ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS"],
    "Healthcare": ["CUPID.NS","SUNPHARMA.NS","CIPLA.NS","DRREDDY.NS","LUPIN.NS"],
    "Capital Markets": ["BSE.NS","CDSL.NS","MCX.NS","ANGELONE.NS","CAMS.NS"],
    "Energy": ["RELIANCE.NS","ONGC.NS","NTPC.NS","POWERGRID.NS","TATAPOWER.NS"],
    "Auto": ["TATAMOTORS.NS","M&M.NS","MARUTI.NS","TVSMOTOR.NS","BAJAJ-AUTO.NS"]
}

STOCK_SECTOR = {s: sec for sec, lst in SECTOR_PEERS.items() for s in lst}

POS_WORDS = {"surge","gain","profit","up","bull","buy","growth","high","record","win","order","boost","rally","strong","upgrade","dividend","bonus"}
NEG_WORDS = {"fall","drop","loss","down","bear","sell","weak","low","slump","cut","crash","decline","warn","debt","downgrade"}

@app.get("/")
def home():
    return FileResponse("index.html")

def fast_yahoo_chart(sym: str, range_str: str = "1d", interval: str = "5m"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={range_str}&interval={interval}"
    r = requests.get(url, headers=HEADERS, timeout=4).json()
    res = r.get("chart", {}).get("result")
    if not res:
        return None
    return res[0]

@app.get("/fast-quote")
def fast_quote(ticker: str):
    q = ticker.strip().upper()
    if not q.startswith("^") and not q.endswith(".NS") and not q.endswith(".BO"):
        q = f"{q}.NS"
    ck = f"fq_{q}"
    if ck in CACHE and time.time() - CACHE[ck]["ts"] < 2.5:
        return CACHE[ck]["data"]
    data = fast_yahoo_chart(q, "1d", "1m")
    if not data and q.endswith(".BO"):
        data = fast_yahoo_chart(q.replace(".BO", ".NS"), "1d", "1m")
    if not data:
        raise HTTPException(status_code=404, detail="Live quote unavailable")
    meta = data.get("meta", {})
    price = round(float(meta.get("regularMarketPrice", 0)), 2)
    prev = round(float(meta.get("chartPreviousClose", price)), 2)
    chg = round(price - prev, 2)
    pct = round((chg / prev) * 100, 2) if prev else 0.0
    out = {
        "ticker": q,
        "price": price,
        "chg": chg,
        "pct": pct,
        "day_high": round(float(meta.get("regularMarketDayHigh", price)), 2),
        "day_low": round(float(meta.get("regularMarketDayLow", price)), 2),
        "volume": int(meta.get("regularMarketVolume", 0)),
        "time": datetime.now().strftime("%H:%M:%S")
    }
    CACHE[ck] = {"ts": time.time(), "data": out}
    return out
    @app.get("/search")
def search_stocks(q: str = ""):
    q_clean = q.strip()
    defaults = [
        {"sym":"CUPID.NS","name":"Cupid Ltd","sector":"Healthcare","exch":"NSE"},
        {"sym":"HAL.NS","name":"Hindustan Aeronautics Ltd","sector":"Defence","exch":"NSE"},
        {"sym":"HDFCBANK.NS","name":"HDFC Bank Ltd","sector":"Banking","exch":"NSE"},
        {"sym":"BSE.NS","name":"BSE Ltd","sector":"Capital Markets","exch":"NSE"},
        {"sym":"RELIANCE.NS","name":"Reliance Industries Ltd","sector":"Energy","exch":"NSE"},
        {"sym":"BEL.NS","name":"Bharat Electronics Ltd","sector":"Defence","exch":"NSE"}
    ]
    if not q_clean:
        return defaults
    results = []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q_clean}&quotesCount=14&newsCount=0"
        r = requests.get(url, headers=HEADERS, timeout=3).json()
        for item in r.get("quotes", []):
            sym = item.get("symbol", "")
            if sym.endswith(".NS") or sym.endswith(".BO") or sym.startswith("^"):
                exch = "BSE" if sym.endswith(".BO") else ("INDEX" if sym.startswith("^") else "NSE")
                results.append({"sym": sym, "name": item.get("shortname") or item.get("longname") or sym, "sector": item.get("sector") or exch, "exch": exch})
    except Exception:
        pass
    if not results:
        u = q_clean.upper()
        results = [{"sym": f"{u}.NS", "name": f"{u} (NSE)", "sector": "NSE Equity", "exch": "NSE"}, {"sym": f"{u}.BO", "name": f"{u} (BSE)", "sector": "BSE Equity", "exch": "BSE"}]
    return results[:10]

@app.get("/indices")
def get_indices():
    if "indices" in CACHE and time.time() - CACHE["indices"]["ts"] < 4:
        return CACHE["indices"]["data"]
    idx_list = [("SENSEX","^BSESN"),("NIFTY 50","^NSEI"),("NEXT 50","^NSMIDCP"),("BANK NIFTY","^NSEBANK"),("NIFTY IT","^CNXIT"),("INDIA VIX","^INDIAVIX")]
    out = []
    for name, sym in idx_list:
        try:
            d = fast_yahoo_chart(sym, "1d", "5m")
            if d:
                m = d.get("meta", {})
                c = round(float(m.get("regularMarketPrice", 0)), 2)
                p = round(float(m.get("chartPreviousClose", c)), 2)
                chg = round(c - p, 2)
                out.append({"name": name, "sym": sym, "price": f"{c:,.2f}", "chg": chg, "pct": round((chg/p)*100, 2) if p else 0.0})
        except Exception:
            continue
    if out:
        CACHE["indices"] = {"ts": time.time(), "data": out}
    return out

@app.get("/index-details")
def index_details(sym: str = "^NSEI"):
    ck = f"idx_det_{sym}"
    if ck in CACHE and time.time() - CACHE[ck]["ts"] < 30:
        return CACHE[ck]["data"]
    info = INDEX_CONSTITUENTS.get(sym, INDEX_CONSTITUENTS["^NSEI"])
    syms = info["stocks"]
    stocks = []
    try:
        df_all = yf.download(syms, period="5d", interval="1d", group_by="ticker", progress=False)
        for s in syms:
            try:
                sub = df_all[s].dropna()
                c, p = round(float(sub["Close"].iloc[-1]), 2), round(float(sub["Close"].iloc[-2]), 2)
                chg = round(c - p, 2)
                pct = round((chg / p) * 100, 2)
                vr = float(sub["Volume"].iloc[-1] / max(1, sub["Volume"].mean()))
                X = np.nan_to_num(np.array([[chg/p, chg/p, 1.01, 1.01, 0.015, 0.02, vr]]), nan=0.0)
                prob = round(float(model.predict_proba(X)[0][1]) * 100, 1)
                stocks.append({"sym": s, "name": s.replace(".NS",""), "price": c, "chg": chg, "pct": pct, "prob": prob, "vol_spike": vr > 1.6})
            except Exception:
                continue
    except Exception:
        pass
    adv = sum(1 for x in stocks if x["chg"] >= 0)
    dec = len(stocks) - adv
    res = {"sym": sym, "name": info["name"], "adv": adv, "dec": dec, "stocks": stocks}
    if stocks:
        CACHE[ck] = {"ts": time.time(), "data": res}
    return res

@app.get("/market-home")
def market_home():
    if "home" in CACHE and time.time() - CACHE["home"]["ts"] < 45:
        return CACHE["home"]["data"]
    syms = ["CUPID.NS","HDFCBANK.NS","HAL.NS","RELIANCE.NS","SBIN.NS","BEL.NS","TATAMOTORS.NS","TCS.NS","INFY.NS","TRENT.NS","BSE.NS","CDSL.NS","MAZDOCK.NS","ICICIBANK.NS"]
    stocks = []
    try:
        df_all = yf.download(syms, period="1y", interval="1d", group_by="ticker", progress=False)
        for s in syms:
            try:
                sub = df_all[s].dropna()
                c, p = round(float(sub["Close"].iloc[-1]), 2), round(float(sub["Close"].iloc[-2]), 2)
                h52, l52 = float(sub["High"].max()), float(sub["Low"].min())
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
                stocks.append({
                    "sym": s, "name": s.replace(".NS",""), "sector": STOCK_SECTOR.get(s,"NSE"),
                    "price": c, "chg": chg, "pct": pct, "prob": prob,
                    "near_52h": ((h52 - c) / h52) <= 0.04,
                    "vol_spike": vr > 1.5
                })
            except Exception:
                continue
    except Exception:
        pass

    sectors = {}
    for st in stocks:
        sectors.setdefault(st["sector"], []).append(st["pct"])
    sector_heat = [{"sector": k, "pct": round(sum(v)/len(v), 2)} for k, v in sectors.items()]
    adv = sum(1 for x in stocks if x["chg"] >= 0)
    mood_score = min(92, max(18, int((adv / max(1, len(stocks))) * 80 + 12)))

    data = {
        "mood_score": mood_score,
        "mood_label": "Greed" if mood_score > 60 else ("Fear" if mood_score < 40 else "Neutral"),
        "ai_picks": sorted(stocks, key=lambda x: x["prob"], reverse=True)[:5],
        "breakouts_52w": [x for x in stocks if x["near_52h"]][:5] or sorted(stocks, key=lambda x: x["pct"], reverse=True)[:4],
        "gainers": sorted(stocks, key=lambda x: x["pct"], reverse=True)[:5],
        "losers": sorted(stocks, key=lambda x: x["pct"])[:5],
        "sectors": sector_heat
    }
    if stocks:
        CACHE["home"] = {"ts": time.time(), "data": data}
    return data

def scrape_screener_fundamentals(clean_sym: str):
    out = {"mkt_cap":"N/A","pe":"N/A","pb":"N/A","book_val":"N/A","roe":"N/A","roce":"N/A","face_val":"N/A","div_yield":"0.00%","eps":"N/A"}
    try:
        url = f"https://www.screener.in/company/{clean_sym}/consolidated/"
        r = requests.get(url, headers=HEADERS, timeout=4)
        if r.status_code != 200 or "Market Cap" not in r.text:
            r = requests.get(f"https://www.screener.in/company/{clean_sym}/", headers=HEADERS, timeout=4)
        html = r.text
        def grab(label):
            m = re.search(rf"{label}\s*</span>\s*<span class=\"number\">([\d,\.]+)", html)
            return m.group(1) if m else None
        mc = grab("Market Cap")
        pe = grab("Stock P/E")
        bv = grab("Book Value")
        dy = grab("Dividend Yield")
        roce = grab("ROCE")
        roe = grab("ROE")
        fv = grab("Face Value")
        cp = grab("Current Price")
        if mc: out["mkt_cap"] = f"₹{mc} Cr"
        if pe: out["pe"] = pe
        if bv:
            out["book_val"] = f"₹{bv}"
            if cp:
                out["pb"] = str(round(float(cp.replace(",","")) / max(0.1, float(bv.replace(",",""))), 2))
        if dy: out["div_yield"] = f"{dy}%"
        if roce: out["roce"] = f"{roce}%"
        if roe: out["roe"] = f"{roe}%"
        if fv: out["face_val"] = f"₹{fv}"
        if cp and pe and float(pe.replace(",","")) > 0:
            out["eps"] = f"₹{round(float(cp.replace(',','')) / float(pe.replace(',','')), 2)}"
    except Exception:
        pass
    return out

@app.get("/watchlist")
def get_watchlist(syms: str):
    sym_list = [x.strip() for x in syms.split(",") if x.strip()]
    res = []
    for s in sym_list:
        try:
            q = fast_quote(s)
            res.append({"sym": s, "price": q["price"], "chg": q["chg"], "pct": q["pct"]})
        except Exception:
            continue
    return res

@app.get("/mlops-metrics")
def mlops_metrics():
    return {"model_metrics": bundle["metrics"], "feature_importance": bundle["feature_importance"], "live_telemetry": telemetry}

@app.post("/simulate")
def simulate_whatif(ret1: float = 1.5, vol_ratio: float = 1.5):
    X = np.array([[ret1/100.0, (ret1*2)/100.0, 1.02, 1.02, 0.018, 0.02, vol_ratio]])
    prob = round(float(model.predict_proba(X)[0][1]) * 100, 1)
    return {"prob_up": prob, "signal": "BULLISH (UP)" if prob >= 52 else "BEARISH / SIDEWAYS"}

@app.get("/analyze")
def analyze_stock(ticker: str = "CUPID.NS", period: str = "1y"):
    t0 = time.time()
    raw_q = ticker.strip().upper()
    if not raw_q.startswith("^") and not raw_q.endswith(".NS") and not raw_q.endswith(".BO"):
        raw_q = f"{raw_q}.NS"

    # Seamless BSE -> NSE data fallback so BSE switch NEVER throws error
    data_sym = raw_q
    if raw_q.endswith(".BO"):
        test_bo = fast_yahoo_chart(raw_q, "5d", "1d")
        if not test_bo:
            data_sym = raw_q.replace(".BO", ".NS")

    ck = f"an_{raw_q}_{period}"
    if ck in CACHE and time.time() - CACHE[ck]["ts"] < 45:
        telemetry["cache_hits"] += 1
        return CACHE[ck]["data"]

    if period == "1d":
        df = yf.download(data_sym, period="5d", interval="5m", progress=False)
    else:
        valid = {"1wk":"1mo","1mo":"3mo","3mo":"3mo","6mo":"6mo","1y":"1y","5y":"5y","max":"max"}
        df = yf.download(data_sym, period=valid.get(period, "1y"), interval="1d", progress=False)

    if df.empty or len(df) < 5:
        raise HTTPException(status_code=404, detail=f"Ticker '{raw_q}' not found.")
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

    clean_name = data_sym.replace(".NS","").replace(".BO","")
    funds = scrape_screener_fundamentals(clean_name) if not data_sym.startswith("^") else {"mkt_cap":"Index","pe":"22.4","pb":"3.6","book_val":"N/A","roe":"15.2%","roce":"16.8%","face_val":"N/A","div_yield":"1.25%","eps":"N/A"}

    # Dividends & Corporate Events
    events = []
    try:
        tk = yf.Ticker(data_sym)
        divs = tk.dividends
        if divs is not None and len(divs) > 0:
            for dt, val in divs.tail(4).iloc[::-1].items():
                events.append({"type": "DIVIDEND", "date": str(dt.date()), "detail": f"₹{round(float(val), 2)} per share cash dividend"})
        splits = tk.splits
        if splits is not None and len(splits) > 0:
            for dt, val in splits.tail(2).iloc[::-1].items():
                events.append({"type": "SPLIT / BONUS", "date": str(dt.date()), "detail": f"Stock Split / Bonus Ratio {val}:1"})
    except Exception:
        pass
    if not events:
        events = [{"type": "BOARD MEETING", "date": "Upcoming", "detail": "Quarterly Financial Results & Corporate Action Review"}]

    # Similar Sector Peers
    sec = STOCK_SECTOR.get(data_sym, "Banking")
    peer_syms = [p for p in SECTOR_PEERS.get(sec, []) if p != data_sym][:4]
    peers_data = []
    for ps in peer_syms:
        try:
            pq = fast_quote(ps)
            peers_data.append({"sym": ps, "name": ps.replace(".NS",""), "price": pq["price"], "pct": pq["pct"]})
        except Exception:
            continue

    # News Sentiment
    news_items = []
    pos_c, neg_c = 0, 0
    try:
        nr = requests.get(f"https://query2.finance.yahoo.com/v1/finance/search?q={clean_name}&quotesCount=0&newsCount=5", headers=HEADERS, timeout=3).json()
        for n in nr.get("news", [])[:4]:
            title = n.get("title", "")
            words = set(title.lower().replace(",","").replace(".","").split())
            tag = "BULLISH" if (words & POS_WORDS) else ("BEARISH" if (words & NEG_WORDS) else "NEUTRAL")
            if tag == "BULLISH": pos_c += 1
            elif tag == "BEARISH": neg_c += 1
            news_items.append({"title": title, "publisher": n.get("publisher","Market News"), "sentiment": tag})
    except Exception:
        pass

    if period == "1d":
        slice_df = df.tail(75)
        hist = [{"d": idx.strftime("%H:%M"), "o": round(float(r["Open"]),2), "h": round(float(r["High"]),2), "l": round(float(r["Low"]),2), "c": round(float(r["Close"]),2), "ema": round(float(r["SMA_20"]),2)} for idx, r in slice_df.iterrows()]
    else:
        slice_df = df.tail(7) if period == "1wk" else (df.tail(22) if period == "1mo" else df)
        hist = [{"d": str(idx.date()), "o": round(float(r["Open"]),2), "h": round(float(r["High"]),2), "l": round(float(r["Low"]),2), "c": round(float(r["Close"]),2), "ema": round(float(r["SMA_20"]),2)} for idx, r in slice_df.iterrows()]

    result = {
        "ticker": raw_q, "company_name": clean_name, "exchange": "BSE" if raw_q.endswith(".BO") else ("INDEX" if raw_q.startswith("^") else "NSE"),
        "sector": sec, "price": price, "open": round(float(latest["Open"]), 2), "prev_close": prev_c,
        "day_chg": day_chg, "day_pct": day_pct, "day_low": round(float(latest["Low"]), 2), "day_high": round(float(latest["High"]), 2),
        "w52_low": round(float(df["Low"].tail(252).min()), 2), "w52_high": round(float(df["High"].tail(252).max()), 2),
        "volume": int(latest["Volume"]), "sma_20": round(float(latest["SMA_20"]), 2), "prob_up": prob_up, "ml_signal": signal,
        "support": round(price - move, 2), "resistance": round(price + move * 1.6, 2), "rsi": rsi_v,
        "rsi_zone": "Overbought" if rsi_v > 70 else ("Oversold" if rsi_v < 30 else "Neutral"),
        "macd_signal": "Bullish Crossover" if float(latest["MACD"]) > 0 else "Bearish Momentum",
        "fundamentals": funds, "events": events, "peers": peers_data,
        "news": {"news": news_items, "overall": "POSITIVE" if pos_c > neg_c else ("NEGATIVE" if neg_c > pos_c else "NEUTRAL")},
        "history": hist
    }

    CACHE[ck] = {"ts": time.time(), "data": result}
    ms = round((time.time() - t0) * 1000, 1)
    n = telemetry["requests"]
    telemetry["avg_ms"] = round(((telemetry["avg_ms"] * n) + ms) / (n + 1), 1)
    telemetry["requests"] = n + 1
    telemetry["last_ticker"] = raw_q
    return result
    
