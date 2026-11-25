from flask import Flask, jsonify, render_template, request
from datetime import datetime, timezone
import json
import time
import threading
import urllib.request
from urllib.parse import urlencode
from pathlib import Path
import pandas as pd
import os
import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from dotenv import load_dotenv

from trading_engine import ScalpingStrategy, InsufficientDataError

load_dotenv(dotenv_path=Path('/workspace/.env'))

app = Flask(__name__)

strategy = ScalpingStrategy()

BOT_STATE = {
    "running": False,
    "mode": "PAPER",
    "last_signal": "WAITING",
    "confidence": 0,
    "buy_confidence": 50,
    "sell_confidence": 50,
    "entry": 0,
    "stop_loss": 0,
    "take_profit": 0,
    "trend": "UNKNOWN",
    "reasons": [],
    "message": "Waiting for market data."
}



# ---------------- PAPER TRADING ENGINE ----------------
PAPER_STATE = {
    "starting_balance": 10.0,
    "balance": 10.0,
    "lot_size": 0.01,
    "contract_size": 100.0,
    "open_trade": None,
    "open_trades": [],
    "max_open_trades": 5,
    "history": [],
    "last_entry_candle": None,
    "total_profit": 0.0,
    "total_trades": 0,
    "wins": 0,
    "losses": 0
}


LIVE_PRICE_HISTORY = []

def update_direction_confidence(price):
    LIVE_PRICE_HISTORY.append(float(price))
    del LIVE_PRICE_HISTORY[:-30]
    if len(LIVE_PRICE_HISTORY) < 4:
        return 50.0, 50.0

    prices = LIVE_PRICE_HISTORY
    changes = []
    for n, weight in ((3, 0.5), (8, 0.3), (15, 0.2)):
        if len(prices) > n:
            changes.append(((prices[-1] - prices[-1-n]) / prices[-1-n]) * weight * 10000)

    momentum = sum(changes) / max(sum((0.5, 0.3, 0.2)[:len(changes)]), 0.01)
    score = max(-40.0, min(40.0, momentum * 8))
    buy = round(50 + score, 1)
    return buy, round(100 - buy, 1)


def sync_open_trade_compat():
    PAPER_STATE["open_trade"] = (
        PAPER_STATE["open_trades"][0] if PAPER_STATE["open_trades"] else None
    )


def paper_close_trade(trade, price, reason):
    if not trade or trade not in PAPER_STATE["open_trades"]:
        return

    direction = 1 if trade["direction"] == "BUY" else -1
    pnl = ((float(price) - trade["entry"]) * direction *
           PAPER_STATE["contract_size"] * trade["lot_size"])

    trade["exit"] = round(float(price), 3)
    trade["profit_loss"] = round(pnl, 2)
    trade["result"] = reason
    trade["closed_at"] = datetime.now(timezone.utc).isoformat()
    trade["status"] = "CLOSED"

    PAPER_STATE["balance"] = round(PAPER_STATE["balance"] + pnl, 2)
    PAPER_STATE["total_profit"] = round(PAPER_STATE["total_profit"] + pnl, 2)
    PAPER_STATE["total_trades"] += 1
    if pnl >= 0:
        PAPER_STATE["wins"] += 1
    else:
        PAPER_STATE["losses"] += 1

    PAPER_STATE["history"].insert(0, trade.copy())
    PAPER_STATE["history"] = PAPER_STATE["history"][:50]
    PAPER_STATE["open_trades"].remove(trade)
    sync_open_trade_compat()
    BOT_STATE["message"] = f"Paper trade closed: {reason}. P/L: ${pnl:.2f}"


def update_paper_market(df, market_result):
    if df is None or df.empty:
        return

    candle_time = str(df.index[-1].isoformat())
    candle = df.iloc[-1]
    high = float(candle["high"])
    low = float(candle["low"])
    price = float(market_result.get("price") or candle["close"])

    # Manage every open position independently.
    for trade in list(PAPER_STATE["open_trades"]):
        direction = trade["direction"]
        entry = float(trade["entry"])
        sl = float(trade["stop_loss"])
        tp = float(trade["take_profit"])
        risk = float(trade.get("risk_distance", abs(entry - sl)))
        lot = float(trade["lot_size"])
        current_pnl = ((price - entry) * (1 if direction == "BUY" else -1)
                       * PAPER_STATE["contract_size"] * lot)

        opposite = market_result.get("signal") == ("SELL" if direction == "BUY" else "BUY")
        opposite_conf = float(
            market_result.get("sell_confidence" if direction == "BUY" else "buy_confidence", 50)
        )
        if opposite and current_pnl > 0.02 and opposite_conf >= 65:
            paper_close_trade(trade, price, "REVERSAL PROFIT LOCK")
            continue

        if direction == "BUY":
            trade["best_price"] = max(float(trade.get("best_price", entry)), high)
            best_profit = trade["best_price"] - entry
            if risk > 0 and best_profit >= 0.20:
                trade["stop_loss"] = round(max(sl, entry, trade["best_price"] - 0.10), 3)
            sl = float(trade["stop_loss"])
            if low <= sl or price <= sl:
                paper_close_trade(trade, sl, "TRAILING/STOP LOSS")
            elif high >= tp or price >= tp:
                paper_close_trade(trade, tp, "TAKE PROFIT")
        else:
            trade["best_price"] = min(float(trade.get("best_price", entry)), low)
            best_profit = entry - trade["best_price"]
            if risk > 0 and best_profit >= 0.20:
                trade["stop_loss"] = round(min(sl, entry, trade["best_price"] + 0.10), 3)
            sl = float(trade["stop_loss"])
            if high >= sl or price >= sl:
                paper_close_trade(trade, sl, "TRAILING/STOP LOSS")
            elif low <= tp or price <= tp:
                paper_close_trade(trade, tp, "TAKE PROFIT")

    sync_open_trade_compat()

    if not BOT_STATE["running"]:
        return

    buy_conf = float(market_result.get("buy_confidence", 50) or 50)
    sell_conf = float(market_result.get("sell_confidence", 50) or 50)

    # Re-arm each direction only after confidence falls below 75%.
    if buy_conf < 75:
        PAPER_STATE["buy_armed"] = True
    if sell_conf < 75:
        PAPER_STATE["sell_armed"] = True

    if len(PAPER_STATE["open_trades"]) >= PAPER_STATE["max_open_trades"]:
        return

    buy_ready = buy_conf >= 80 and PAPER_STATE.get("buy_armed", True)
    sell_ready = sell_conf >= 80 and PAPER_STATE.get("sell_armed", True)

    if not buy_ready and not sell_ready:
        return

    direction = "BUY" if buy_ready and (not sell_ready or buy_conf >= sell_conf) else "SELL"
    entry = price

    # Fixed XAUUSD price distances: small target, wider protective stop.
    target_distance = 0.50
    stop_distance = 0.75

    if entry <= 0:
        return

    sl = entry - stop_distance if direction == "BUY" else entry + stop_distance
    tp = entry + target_distance if direction == "BUY" else entry - target_distance

    if direction == "BUY":
        PAPER_STATE["buy_armed"] = False
    else:
        PAPER_STATE["sell_armed"] = False

    trade = {
        "direction": direction,
        "entry": round(entry, 3),
        "stop_loss": round(sl, 3),
        "take_profit": round(tp, 3),
        "risk_distance": round(abs(entry - sl), 3),
        "best_price": entry,
        "lot_size": PAPER_STATE["lot_size"],
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "status": "OPEN"
    }
    PAPER_STATE["open_trades"].append(trade)
    sync_open_trade_compat()
    PAPER_STATE["last_entry_candle"] = candle_time
    BOT_STATE["message"] = (
        f"Paper {direction} opened at {entry:.3f} "
        f"with {PAPER_STATE['lot_size']:.2f} lots."
    )

# -------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "success": True,
        "service": "XAUUSD AI Scalper",
        "status": "healthy"
    })




@app.route("/api/live-quote")
def live_quote():
    try:
        q = get_biquote_live()
        return jsonify({"success": True, "quote": q, "source": "BiQuote"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502

@app.route("/api/status")
def status():
    return jsonify({
        "success": True,
        "server": "online",
        "bot": BOT_STATE,
        "paper": PAPER_STATE,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })




# BiQuote live market feed - builds candles from live quotes
def get_biquote_live():
    url = "https://biquote.io/api/XAUUSD"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode())
    if data.get("stale"):
        raise ValueError("BiQuote returned stale data")
    bid = float(data["bid"])
    ask = float(data["ask"])
    mid = float(data.get("mid") or (bid + ask) / 2)
    return {
        "symbol": data.get("symbol", "XAUUSD"),
        "bid": bid,
        "ask": ask,
        "price": mid,
        "timestamp": data.get("timestamp"),
        "quote_age": data.get("quoteAgeSeconds"),
        "market_state": data.get("marketState"),
        "stale": data.get("stale", False)
    }

LIVE_CACHE = {"time": 0, "data": None}
LIVE_CACHE_SECONDS = 1

QUOTE_CANDLES = {}
QUOTE_CANDLES_LOCK = threading.Lock()
QUOTE_HISTORY = []
QUOTE_HISTORY_LOCK = threading.Lock()
QUOTE_COLLECTOR_STARTED = False



def get_live_candles():
    env_path = Path(__file__).resolve().parent / ".env"
    key = ""
    for line in env_path.read_text().splitlines():
        if line.strip().startswith("TWELVE_DATA_API_KEY="):
            key = line.split("=", 1)[1].strip()
            break
    if not key:
        raise RuntimeError("Twelve Data API key missing")

    params = urlencode({
        "symbol": "XAU/USD",
        "interval": "1min",
        "outputsize": 400,
        "apikey": key
    })
    url = "https://api.twelvedata.com/time_series?" + params
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req, timeout=20) as response:
        data = json.loads(response.read().decode())

    if "values" not in data:
        raise RuntimeError(data.get("message", "Historical candles unavailable"))

    df = pd.DataFrame(data["values"])
    df["timestamp"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = 0
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.set_index("timestamp").sort_index()
    return df[~df.index.duplicated(keep="last")]



def fetch_startup_history():
    global QUOTE_HISTORY
    try:
        df = get_live_candles().tail(400)

        records = []
        for timestamp, row in df.iterrows():
            records.append({
                "timestamp": timestamp,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": 0
            })

        with QUOTE_HISTORY_LOCK:
            QUOTE_HISTORY = records

        print(f"[BOOT] Loaded {len(records)} historical M1 candles.")

        df5 = df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        if len(df) >= 60 and len(df5) >= 60:
            signal = strategy.analyze(df, df5)
            entry = signal.price
            sl = 0.0
            tp = 0.0

            if signal.action == "BUY":
                sl = entry - signal.stop_distance
                tp = entry + signal.stop_distance * 0.9
            elif signal.action == "SELL":
                sl = entry + signal.stop_distance
                tp = entry - signal.stop_distance * 0.9

            BOT_STATE.update({
                "last_signal": signal.action,
                "confidence": round(signal.confidence * 100, 2),
                "entry": round(entry, 3),
                "stop_loss": round(sl, 3),
                "take_profit": round(tp, 3),
                "trend": signal.trend_state,
                "reasons": signal.reasons,
                "market_data": "BiQuote",
                "message": "Initial startup analysis completed."
            })

            print(
                f"[BOOT] Initial analysis: {signal.action} | "
                f"Trend: {signal.trend_state} | "
                f"Confidence: {signal.confidence * 100:.1f}%"
            )
        else:
            print(f"[BOOT] Insufficient data: M1={len(df)}, M5={len(df5)}")

    except Exception as e:
        print("[BOOT] Historical loading/analysis failed:", str(e)[:250])


def collect_biquote():
    global QUOTE_HISTORY
    fetch_startup_history()
    while True:
        try:
            q = get_biquote_live()
            price = float(q["price"])
            now = pd.to_datetime(q.get("timestamp"), utc=True) if q.get("timestamp") else pd.Timestamp.now(tz="UTC")
            minute = now.floor("min")

            with QUOTE_CANDLES_LOCK:
                candle = QUOTE_CANDLES.get(str(minute))
                if candle is None:
                    candle = {
                        "timestamp": minute,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 0
                    }
                    QUOTE_CANDLES[str(minute)] = candle
                else:
                    candle["high"] = max(candle["high"], price)
                    candle["low"] = min(candle["low"], price)
                    candle["close"] = price

                completed = sorted(
                    QUOTE_CANDLES.values(),
                    key=lambda x: x["timestamp"]
                )
                with QUOTE_HISTORY_LOCK:
                    merged = {
                        str(pd.Timestamp(item["timestamp"])):
                        item for item in QUOTE_HISTORY
                    }
                    for item in completed:
                        merged[str(pd.Timestamp(item["timestamp"]))] = item
                    QUOTE_HISTORY = sorted(
                        merged.values(),
                        key=lambda x: pd.Timestamp(x["timestamp"])
                    )[-500:]

            # Manage existing paper positions on every newly received quote.
            live_df = get_biquote_candles()
            if live_df is not None and not live_df.empty:
                live_result = {
                    "price": price,
                    "bid": float(q["bid"]),
                    "ask": float(q["ask"]),
                    "signal": BOT_STATE.get("last_signal", "WAITING"),
                    "buy_confidence": BOT_STATE.get("buy_confidence", 50),
                    "sell_confidence": BOT_STATE.get("sell_confidence", 50),
                }
                update_paper_market(live_df, live_result)

        except Exception as e:
            print("BiQuote collector:", str(e)[:150])

        time.sleep(1)

def start_biquote_collector():
    global QUOTE_COLLECTOR_STARTED
    if not QUOTE_COLLECTOR_STARTED:
        QUOTE_COLLECTOR_STARTED = True
        threading.Thread(
            target=collect_biquote,
            daemon=True
        ).start()

def get_biquote_candles():
    with QUOTE_HISTORY_LOCK:
        candles = list(QUOTE_HISTORY)

    if not candles:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC")
        )

    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]

start_biquote_collector()

@app.route("/api/live-market")
def live_market():
    now = time.time()

    if LIVE_CACHE["data"] and now - LIVE_CACHE["time"] < LIVE_CACHE_SECONDS:
        return jsonify(LIVE_CACHE["data"])

    try:
        quote = get_biquote_live()
        df = get_biquote_candles()
        latest_price = float(quote["price"])
        buy_conf, sell_conf = update_direction_confidence(latest_price)
        BOT_STATE["buy_confidence"] = buy_conf
        BOT_STATE["sell_confidence"] = sell_conf


        result = {
            "success": True,
            "symbol": "XAUUSD",
            "source": "BiQuote",
            "price": latest_price,
            "buy_confidence": buy_conf,
            "sell_confidence": sell_conf,
            "bid": quote["bid"],
            "ask": quote["ask"],
            "timestamp": quote["timestamp"],
            "candles": len(df),
            "signal": "WAITING"
        }

        if len(df):
            df5 = df.resample("5min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()

            if len(df) >= 60 and len(df5) >= 60:
                signal = strategy.analyze(df, df5)
                entry = signal.price
                sl = 0.0
                tp = 0.0

                if signal.action == "BUY":
                    sl = entry - signal.stop_distance
                    tp = entry + signal.stop_distance * 0.9
                elif signal.action == "SELL":
                    sl = entry + signal.stop_distance
                    tp = entry - signal.stop_distance * 0.9

                BOT_STATE.update({
                    "last_signal": signal.action,
                    "confidence": round(signal.confidence * 100, 2),
                    "entry": round(entry, 3),
                    "stop_loss": round(sl, 3),
                    "take_profit": round(tp, 3),
                    "trend": signal.trend_state,
                    "reasons": signal.reasons,
                    "market_data": "BiQuote",
                    "message": "BiQuote market analysis updated."
                })

                result["signal"] = signal.action
                result["confidence"] = BOT_STATE["confidence"]
                result["stop_distance"] = float(signal.stop_distance)
                result["trend"] = signal.trend_state
            else:
                result["analysis_message"] = (
                    f"Collecting BiQuote candle history: "
                    f"{len(df)} M1 candles and {len(df5)} M5 candles."
                )
        else:
            result["analysis_message"] = "Waiting for BiQuote candle collection."

        LIVE_CACHE.update({"time": now, "data": result})
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)[:250],
            "source": "BiQuote"
        }), 502

@app.route("/api/paper")
def paper_status():
    sync_open_trade_compat()
    return jsonify({"success": True, "paper": PAPER_STATE})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True) or {}
        candles = data.get("candles", [])

        if not isinstance(candles, list) or not candles:
            return jsonify({
                "success": False,
                "error": "Provide a non-empty candles list."
            }), 400

        df = pd.DataFrame(candles)

        required = ["timestamp", "open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing fields: {missing}"
            }), 400

        if "volume" not in df.columns:
            df["volume"] = 0

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce"
        )

        df = df.dropna(subset=required)
        df = df.set_index("timestamp").sort_index()

        if df.index.has_duplicates:
            return jsonify({
                "success": False,
                "error": "Duplicate candle timestamps."
            }), 400

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()

        if len(df) < 300:
            return jsonify({
                "success": False,
                "error": f"Need at least 300 one-minute candles. Received {len(df)}."
            }), 400

        df5 = df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        signal = strategy.analyze(df, df5)

        entry = signal.price
        stop_loss = 0
        take_profit = 0

        if signal.action == "BUY":
            stop_loss = entry - signal.stop_distance
            take_profit = entry + (signal.stop_distance * 0.9)

        elif signal.action == "SELL":
            stop_loss = entry + signal.stop_distance
            take_profit = entry - (signal.stop_distance * 0.9)

        BOT_STATE.update({
            "last_signal": signal.action,
            "confidence": round(signal.confidence * 100, 2),
            "entry": round(entry, 3),
            "stop_loss": round(stop_loss, 3),
            "take_profit": round(take_profit, 3),
            "trend": signal.trend_state,
            "reasons": signal.reasons,
            "message": "Analysis completed.",
            "market_data": "Submitted candles",
        })

        return jsonify({
            "success": True,
            "signal": signal.action,
            "confidence": BOT_STATE["confidence"],
            "entry": BOT_STATE["entry"],
            "stop_loss": BOT_STATE["stop_loss"],
            "take_profit": BOT_STATE["take_profit"],
            "trend": signal.trend_state,
            "reasons": signal.reasons,
            "timestamp": str(signal.timestamp)
        })

    except InsufficientDataError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/api/start", methods=["POST"])
def start_bot():
    BOT_STATE["running"] = True
    BOT_STATE["message"] = (
        "Paper trading active. Simulated orders will be opened when signals qualify."
    )
    return jsonify({"success": True, "bot": BOT_STATE})


@app.route("/api/stop", methods=["POST"])
def stop_bot():
    BOT_STATE["running"] = False
    BOT_STATE["message"] = "Bot stopped."
    return jsonify({"success": True, "bot": BOT_STATE})




# METAAPI_DASHBOARD_INTEGRATION
META_LOOP = asyncio.new_event_loop()
META_CONNECTION = None
META_ACCOUNT = None
META_STATUS = {"connected": False, "message": "Not connected", "account": None, "quote": None}
META_THREAD = None

def meta_loop_worker():
    asyncio.set_event_loop(META_LOOP)
    META_LOOP.run_forever()

def ensure_meta_loop():
    global META_THREAD
    if META_THREAD is None or not META_THREAD.is_alive():
        META_THREAD = threading.Thread(target=meta_loop_worker, daemon=True)
        META_THREAD.start()

def run_meta(coro, timeout=180):
    ensure_meta_loop()
    future = asyncio.run_coroutine_threadsafe(coro, META_LOOP)
    return future.result(timeout=timeout)

async def meta_connect_account(login, password, server, broker):
    global META_ACCOUNT, META_CONNECTION
    from metaapi_cloud_sdk import MetaApi

    token = os.getenv("METAAPI_TOKEN")
    if not token:
        raise RuntimeError("METAAPI_TOKEN is missing from .env")

    api = MetaApi(token=token)
    accounts = await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
    account = next(
        (a for a in accounts if str(a.login) == str(login)
         and str(a.server) == server),
        None
    )

    if account is None:
        account = await api.metatrader_account_api.create_account({
            "name": "EGM Securities Demo",
            "type": "cloud",
            "login": str(login),
            "password": password,
            "server": server,
            "platform": "mt5",
            "application": "MetaApi",
            "magic": 6467001,
            "quoteStreamingIntervalInSeconds": 1
        })

    if account.state != "DEPLOYED":
        await account.deploy()

    if account.connection_status != "CONNECTED":
        await account.wait_connected()

    connection = account.get_streaming_connection()
    await connection.connect()
    await connection.wait_synchronized()

    try:
        await connection.subscribe_to_market_data("XAUUSD", [
            {"type": "quotes", "intervalInMilliseconds": 1000},
            {"type": "candles", "timeframe": "1m", "intervalInMilliseconds": 1000}
        ])
    except Exception:
        pass

    META_ACCOUNT = account
    META_CONNECTION = connection
    info = connection.terminal_state.account_information
    META_STATUS.update({
        "connected": True,
        "message": "MT5 connected",
        "account": {
            "login": str(login),
            "server": server,
            "balance": info.get("balance"),
            "equity": info.get("equity"),
            "currency": info.get("currency"),
            "name": info.get("name")
        }
    })
    return META_STATUS


@app.route("/api/mt5/brokers")
def mt5_brokers():
    import urllib.parse
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify({"success": True, "brokers": {}})
    token = os.getenv("METAAPI_TOKEN")
    url = "https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai/known-mt-servers/5/search?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, headers={"auth-token": token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return jsonify({"success": True, "brokers": json.loads(response.read().decode())})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)[:300]}), 400


@app.route("/api/mt5/connect", methods=["POST"])
def mt5_connect():
    data = request.get_json(silent=True) or {}
    login = str(data.get("login", "")).strip()
    password = str(data.get("password", ""))
    server = str(data.get("server", "")).strip()
    broker = str(data.get("broker", "")).strip()

    if not login or not password or not server:
        return jsonify({"success": False, "error": "Enter login, password and select an MT5 server."}), 400

    if not login or not password:
        return jsonify({"success": False, "error": "Enter your MT5 login and password."}), 400

    try:
        result = run_meta(meta_connect_account(login, password, server, broker), timeout=240)
        return jsonify({"success": True, "mt5": result})
    except FutureTimeout:
        META_STATUS.update({"connected": False, "message": "Connection timed out"})
        return jsonify({"success": False, "error": "MetaApi connection timed out. Check account status in MetaApi."}), 504
    except Exception as e:
        META_STATUS.update({"connected": False, "message": str(e)[:250]})
        return jsonify({"success": False, "error": str(e)[:400]}), 400

@app.route("/api/mt5/status")
def mt5_status():
    if META_CONNECTION and META_ACCOUNT:
        try:
            state = META_CONNECTION.terminal_state
            info = state.account_information
            quote = state.price("XAUUSD")
            META_STATUS["connected"] = bool(state.connected and state.connected_to_broker)
            META_STATUS["account"] = {
                "login": str(META_ACCOUNT.login),
                "server": str(META_ACCOUNT.server),
                "balance": info.get("balance"),
                "equity": info.get("equity"),
                "currency": info.get("currency")
            }
            META_STATUS["quote"] = quote
        except Exception as e:
            META_STATUS["message"] = str(e)[:200]
    return jsonify({"success": True, "mt5": META_STATUS})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
