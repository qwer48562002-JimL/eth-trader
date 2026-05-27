"""
ETH 多空雙向模擬交易機器人
使用 MAX 交易所公開 API 取得 K 線資料
指標：Stoch RSI、MACD、BB 線
紀錄儲存：PostgreSQL（Railway 提供）
"""

import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# ── PostgreSQL ───────────────────────────────────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    HAS_PG = True
except ImportError:
    HAS_PG = False

# ── 設定區 ──────────────────────────────────────────────────────────────────

CONFIG = {
    "initial_capital_long":  1000,
    "initial_capital_short": 1000,
    "trade_ratio": 1.0,
    "symbol": "ethtwd",
    "quote_currency": "TWD",
    "main_period":     "30m",
    "trend_period_4h": "4h",
    "trend_period_1d": "1d",
    "stop_loss_pct":   0.10,
    "trail_start_pct": 0.03,
    "trail_drop_pct":  0.015,
    "stoch_rsi_period":   14,
    "stoch_rsi_smooth_k":  3,
    "stoch_rsi_smooth_d":  3,
    "macd_fast":   12,
    "macd_slow":   26,
    "macd_signal":  9,
    "bb_period":   20,
    "bb_std":       2.0,
    "check_interval_sec": 60,
}

MAX_API_BASE = "https://max-api.maicoin.com/api/v2"

# ── 資料庫連線 ───────────────────────────────────────────────────────────────

def get_db_conn():
    """從環境變數 DATABASE_URL 取得連線"""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url or not HAS_PG:
        return None
    try:
        conn = psycopg2.connect(db_url, sslmode="require")
        return conn
    except Exception as e:
        print(f"[DB] 連線失敗：{e}")
        return None


def init_db():
    """建立資料表（若不存在）"""
    conn = get_db_conn()
    if not conn:
        print("[DB] 無法連線，將使用本機檔案儲存")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMP NOT NULL,
                    side TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price FLOAT NOT NULL,
                    pnl_pct FLOAT NOT NULL,
                    reason TEXT NOT NULL,
                    signals TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
        conn.commit()
        print("[DB] 資料表初始化完成")
    except Exception as e:
        print(f"[DB] 初始化失敗：{e}")
    finally:
        conn.close()


def db_get(key: str, default=None):
    """從資料庫讀取狀態"""
    import json
    conn = get_db_conn()
    if not conn:
        return default
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM state WHERE key = %s", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else default
    except Exception as e:
        print(f"[DB] 讀取失敗 {key}：{e}")
        return default
    finally:
        conn.close()


def db_set(key: str, value):
    """寫入狀態到資料庫"""
    import json

    def serialize(obj):
        if isinstance(obj, bool):
            return bool(obj)
        if isinstance(obj, (int, float, str, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [serialize(i) for i in obj]
        return str(obj)

    conn = get_db_conn()
    if not conn:
        return
    try:
        serialized = json.dumps(serialize(value))
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO state (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
            """, (key, serialized))
        conn.commit()
    except Exception as e:
        print(f"[DB] 寫入失敗 {key}：{e}")
    finally:
        conn.close()


def db_log_trade(side: str, action: str, price: float, pnl: float,
                 reason: str, signals: dict):
    """記錄交易到資料庫"""
    import json

    def serialize(obj):
        if isinstance(obj, bool):
            return bool(obj)
        if isinstance(obj, (int, float, str, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [serialize(i) for i in obj]
        return str(obj)

    conn = get_db_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trade_log (time, side, action, price, pnl_pct, reason, signals)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                datetime.now(),
                side, action,
                float(price),
                round(float(pnl) * 100, 2),
                reason,
                json.dumps(serialize(signals))
            ))
        conn.commit()
    except Exception as e:
        print(f"[DB] 寫入交易紀錄失敗：{e}")
    finally:
        conn.close()

# ── MAX API ──────────────────────────────────────────────────────────────────

def fetch_klines(symbol: str, period: str, limit: int = 200) -> pd.DataFrame:
    period_map = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
    minutes = period_map.get(period, 30)
    try:
        resp = requests.get(f"{MAX_API_BASE}/k",
                            params={"market": symbol, "period": minutes, "limit": limit},
                            timeout=10)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json(),
                          columns=["timestamp","open","high","low","close","volume"])
        df = df.astype({c: float for c in ["open","high","low","close","volume"]})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] 取得 K 線失敗 ({symbol} {period}): {e}")
        return pd.DataFrame()

# ── 技術指標 ─────────────────────────────────────────────────────────────────

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_stoch_rsi(close, period=14, smooth_k=3, smooth_d=3):
    rsi = calc_rsi(close, period)
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    k = stoch.rolling(smooth_k).mean()
    return k, k.rolling(smooth_d).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calc_bb(close, period=20, std_mult=2.0):
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    return middle + std_mult*std, middle, middle - std_mult*std

def add_indicators(df):
    cfg = CONFIG
    k, d = calc_stoch_rsi(df["close"], cfg["stoch_rsi_period"],
                           cfg["stoch_rsi_smooth_k"], cfg["stoch_rsi_smooth_d"])
    df["stoch_k"] = k
    df["stoch_d"] = d
    macd, sig, hist = calc_macd(df["close"], cfg["macd_fast"],
                                 cfg["macd_slow"], cfg["macd_signal"])
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = hist
    upper, mid, lower = calc_bb(df["close"], cfg["bb_period"], cfg["bb_std"])
    df["bb_upper"] = upper
    df["bb_middle"] = mid
    df["bb_lower"] = lower
    return df

# ── 趨勢判斷 ─────────────────────────────────────────────────────────────────

def check_trend(df_4h, df_1d):
    if df_4h.empty or df_1d.empty:
        return "neutral"
    df_4h = add_indicators(df_4h)
    df_1d = add_indicators(df_1d)
    m4  = float(df_4h["macd"].iloc[-1])
    h4  = float(df_4h["macd_hist"].iloc[-1])
    m1d = float(df_1d["macd"].iloc[-1])
    if m4 > 0 and h4 > 0 and m1d > 0:
        return "bull"
    if m4 < 0 and h4 < 0 and m1d < 0:
        return "bear"
    return "neutral"

# ── 進場訊號 ─────────────────────────────────────────────────────────────────

def check_long_signals(df):
    row  = df.iloc[-1]
    prev = df.iloc[-2]
    s_stoch = bool(prev["stoch_k"] < 20 and
                   row["stoch_k"] > row["stoch_d"] and
                   prev["stoch_k"] <= prev["stoch_d"])
    s_macd  = bool(row["macd"] > row["macd_signal"] and
                   prev["macd"] <= prev["macd_signal"] and
                   row["macd_hist"] > 0)
    s_bb    = bool(prev["close"] <= prev["bb_lower"] * 1.005 and
                   row["close"] > row["bb_lower"] and
                   row["close"] > prev["close"])
    count = sum([s_stoch, s_macd, s_bb])
    return {"stoch_rsi": s_stoch, "macd": s_macd, "bb": s_bb,
            "count": count, "entry": bool(count >= 2)}

def check_short_signals(df):
    row  = df.iloc[-1]
    prev = df.iloc[-2]
    s_stoch = bool(prev["stoch_k"] > 80 and
                   row["stoch_k"] < row["stoch_d"] and
                   prev["stoch_k"] >= prev["stoch_d"])
    s_macd  = bool(row["macd"] < row["macd_signal"] and
                   prev["macd"] >= prev["macd_signal"] and
                   row["macd_hist"] < 0)
    s_bb    = bool(prev["close"] >= prev["bb_upper"] * 0.995 and
                   row["close"] < row["bb_upper"] and
                   row["close"] < prev["close"])
    count = sum([s_stoch, s_macd, s_bb])
    return {"stoch_rsi": s_stoch, "macd": s_macd, "bb": s_bb,
            "count": count, "entry": bool(count >= 2)}

# ── 出場訊號 ─────────────────────────────────────────────────────────────────

def check_long_exit_signals(df):
    row  = df.iloc[-1]
    prev = df.iloc[-2]
    weak = {
        "stoch_rsi": bool(prev["stoch_k"] > 80 and row["stoch_k"] < row["stoch_d"]),
        "macd":      bool((row["macd"] < row["macd_signal"] and prev["macd"] >= prev["macd_signal"]) or
                          (row["macd_hist"] < 0 and prev["macd_hist"] >= 0)),
        "bb":        bool(prev["close"] >= prev["bb_upper"] * 0.995 and row["close"] < row["bb_middle"]),
    }
    weak["any"] = bool(any([weak["stoch_rsi"], weak["macd"], weak["bb"]]))
    return weak

def check_short_exit_signals(df):
    row  = df.iloc[-1]
    prev = df.iloc[-2]
    weak = {
        "stoch_rsi": bool(prev["stoch_k"] < 20 and row["stoch_k"] > row["stoch_d"]),
        "macd":      bool((row["macd"] > row["macd_signal"] and prev["macd"] <= prev["macd_signal"]) or
                          (row["macd_hist"] > 0 and prev["macd_hist"] <= 0)),
        "bb":        bool(prev["close"] <= prev["bb_lower"] * 1.005 and row["close"] > row["bb_middle"]),
    }
    weak["any"] = bool(any([weak["stoch_rsi"], weak["macd"], weak["bb"]]))
    return weak

# ── 持倉類別 ─────────────────────────────────────────────────────────────────

class Position:
    def __init__(self, side, entry_price, capital, entry_time, signals):
        self.side          = side
        self.entry_price   = float(entry_price)
        self.capital       = float(capital)
        self.entry_time    = entry_time
        self.entry_signals = signals
        self.quantity      = float(capital) / float(entry_price)
        self.trailing_active = False
        self.extreme_price = float(entry_price)

    def update_extreme(self, price):
        price = float(price)
        if self.side == "long":
            if price > self.extreme_price:
                self.extreme_price = price
        else:
            if price < self.extreme_price:
                self.extreme_price = price

    def current_pnl_pct(self, price):
        price = float(price)
        if self.side == "long":
            return (price - self.entry_price) / self.entry_price
        return (self.entry_price - price) / self.entry_price

    def should_stop_loss(self, price):
        return self.current_pnl_pct(price) <= -CONFIG["stop_loss_pct"]

    def check_trailing(self, price):
        price = float(price)
        pnl = self.current_pnl_pct(price)
        if pnl >= CONFIG["trail_start_pct"]:
            self.trailing_active = True
        if self.trailing_active:
            if self.side == "long":
                retrace = (self.extreme_price - price) / self.extreme_price
            else:
                retrace = (price - self.extreme_price) / self.extreme_price
            return retrace >= CONFIG["trail_drop_pct"]
        return False

    def to_dict(self):
        return {
            "side":            self.side,
            "entry_price":     self.entry_price,
            "capital":         self.capital,
            "quantity":        self.quantity,
            "entry_time":      self.entry_time.isoformat(),
            "extreme_price":   self.extreme_price,
            "trailing_active": bool(self.trailing_active),
            "entry_signals":   {k: bool(v) if isinstance(v, (bool, np.bool_)) else v
                                for k, v in self.entry_signals.items()},
        }

    @classmethod
    def from_dict(cls, d):
        p = cls(d["side"], d["entry_price"], d["capital"],
                datetime.fromisoformat(d["entry_time"]), d.get("entry_signals", {}))
        p.quantity        = d["quantity"]
        p.extreme_price   = d["extreme_price"]
        p.trailing_active = d["trailing_active"]
        return p

# ── 狀態管理（DB 優先，fallback 本機 JSON）────────────────────────────────────

import json

def load_state():
    data = db_get("bot_state")
    if data:
        return data
    if os.path.exists("state.json"):
        with open("state.json") as f:
            return json.load(f)
    return {
        "capital_long":   CONFIG["initial_capital_long"],
        "capital_short":  CONFIG["initial_capital_short"],
        "position_long":  None,
        "position_short": None,
        "total_trades": 0, "win_trades": 0, "total_pnl": 0.0,
        "long_trades":  0, "long_wins":  0, "long_pnl":  0.0,
        "short_trades": 0, "short_wins": 0, "short_pnl": 0.0,
    }

def save_state(state):
    db_set("bot_state", state)
    with open("state.json", "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def log_trade(side, action, price, pnl, reason, signals):
    db_log_trade(side, action, price, pnl, reason, signals)

# ── 顯示狀態 ─────────────────────────────────────────────────────────────────

def print_status(state, price, trend, pos_long, pos_short):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = state["total_trades"]
    wr = state["win_trades"] / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"  ETH 多空模擬交易  |  {now}")
    print(f"{'='*60}")
    print(f"  當前價格   : {price:,.0f} {CONFIG['quote_currency']}")
    print(f"  大趨勢方向 : {'🟢 多頭' if trend=='bull' else '🔴 空頭' if trend=='bear' else '⚪ 中性'}")
    print(f"  累計交易   : {total} 筆  勝率 {wr:.1f}%  損益 {state['total_pnl']:+.2f} USDT")
    print(f"\n  【做多】資金 {state['capital_long']:,.2f}  {state['long_trades']}筆  {state['long_pnl']:+.2f} USDT")
    if pos_long:
        pnl = pos_long.current_pnl_pct(price) * 100
        print(f"    ▶ 持倉  進場 {pos_long.entry_price:,.0f}  損益 {pnl:+.2f}%  "
              f"極值 {pos_long.extreme_price:,.0f}  滾動 {'✅' if pos_long.trailing_active else '⏳'}")
    else:
        print(f"    ▶ 空倉，等待多頭訊號")
    print(f"\n  【做空】資金 {state['capital_short']:,.2f}  {state['short_trades']}筆  {state['short_pnl']:+.2f} USDT")
    if pos_short:
        pnl = pos_short.current_pnl_pct(price) * 100
        print(f"    ▶ 持倉  進場 {pos_short.entry_price:,.0f}  損益 {pnl:+.2f}%  "
              f"極值 {pos_short.extreme_price:,.0f}  滾動 {'✅' if pos_short.trailing_active else '⏳'}")
    else:
        print(f"    ▶ 空倉，等待空頭訊號")
    print(f"{'='*60}")

# ── 出場處理 ─────────────────────────────────────────────────────────────────

def process_exit(state, pos, price, exit_reason):
    pnl      = pos.current_pnl_pct(price)
    pnl_usdt = pnl * pos.capital
    sk       = pos.side
    state[f"capital_{sk}"]  += pos.capital + pnl_usdt
    state[f"{sk}_trades"]   += 1
    state[f"{sk}_pnl"]      += pnl_usdt
    state["total_trades"]   += 1
    state["total_pnl"]      += pnl_usdt
    if pnl > 0:
        state["win_trades"]     += 1
        state[f"{sk}_wins"]     += 1
    icon  = "📤" if sk == "long" else "📥"
    label = "多單" if sk == "long" else "空單"
    print(f"\n  {icon} {label}出場！{exit_reason}")
    print(f"     {pos.entry_price:,.0f} → {price:,.0f}  {pnl*100:+.2f}% ({pnl_usdt:+.2f} USDT)")
    log_trade(sk, "EXIT", price, pnl, exit_reason, {})

# ── 主循環 ───────────────────────────────────────────────────────────────────

def run():
    print("🚀 ETH 多空雙向模擬交易機器人啟動")
    init_db()

    state     = load_state()
    pos_long  = Position.from_dict(state["position_long"])  if state.get("position_long")  else None
    pos_short = Position.from_dict(state["position_short"]) if state.get("position_short") else None

    while True:
        try:
            df_30m = fetch_klines(CONFIG["symbol"], CONFIG["main_period"], limit=100)
            df_4h  = fetch_klines(CONFIG["symbol"], CONFIG["trend_period_4h"], limit=100)
            df_1d  = fetch_klines(CONFIG["symbol"], CONFIG["trend_period_1d"], limit=100)

            if df_30m.empty:
                print("[WARN] 無法取得 K 線，等待重試...")
                time.sleep(CONFIG["check_interval_sec"])
                continue

            price  = float(df_30m["close"].iloc[-1])
            df_30m = add_indicators(df_30m)
            trend  = check_trend(df_4h, df_1d)
            print_status(state, price, trend, pos_long, pos_short)

            # ── 多單管理 ──
            if pos_long:
                pos_long.update_extreme(price)
                exit_reason = None
                if pos_long.should_stop_loss(price):
                    exit_reason = "停損"
                else:
                    weak = check_long_exit_signals(df_30m)
                    if pos_long.trailing_active and weak["any"]:
                        names = [k for k, v in weak.items() if v and k != "any"]
                        exit_reason = f"指標轉弱（{'、'.join(names)}）"
                    elif pos_long.check_trailing(price):
                        exit_reason = "滾動停利觸發"
                if exit_reason:
                    process_exit(state, pos_long, price, exit_reason)
                    pos_long = None
                    state["position_long"] = None
            else:
                if trend in ("bull", "neutral"):
                    sig = check_long_signals(df_30m)
                    triggered = [k for k, v in sig.items() if v is True]
                    print(f"  多頭訊號：{', '.join(triggered) if triggered else '無'} ({sig['count']}/3)")
                    if sig["entry"]:
                        cap = state["capital_long"] * CONFIG["trade_ratio"]
                        pos_long = Position("long", price, cap, datetime.now(), sig)
                        state["capital_long"] -= cap
                        state["position_long"] = pos_long.to_dict()
                        log_trade("long", "ENTRY", price, 0, f"訊號：{', '.join(triggered)}", sig)
                        print(f"\n  📈 多單進場！{price:,.0f}  訊號：{', '.join(triggered)}")
                else:
                    print(f"  空頭趨勢，多單暫停")

            # ── 空單管理 ──
            if pos_short:
                pos_short.update_extreme(price)
                exit_reason = None
                if pos_short.should_stop_loss(price):
                    exit_reason = "停損"
                else:
                    weak = check_short_exit_signals(df_30m)
                    if pos_short.trailing_active and weak["any"]:
                        names = [k for k, v in weak.items() if v and k != "any"]
                        exit_reason = f"指標轉強（{'、'.join(names)}）"
                    elif pos_short.check_trailing(price):
                        exit_reason = "滾動停利觸發"
                if exit_reason:
                    process_exit(state, pos_short, price, exit_reason)
                    pos_short = None
                    state["position_short"] = None
            else:
                if trend in ("bear", "neutral"):
                    sig = check_short_signals(df_30m)
                    triggered = [k for k, v in sig.items() if v is True]
                    print(f"  空頭訊號：{', '.join(triggered) if triggered else '無'} ({sig['count']}/3)")
                    if sig["entry"]:
                        cap = state["capital_short"] * CONFIG["trade_ratio"]
                        pos_short = Position("short", price, cap, datetime.now(), sig)
                        state["capital_short"] -= cap
                        state["position_short"] = pos_short.to_dict()
                        log_trade("short", "ENTRY", price, 0, f"訊號：{', '.join(triggered)}", sig)
                        print(f"\n  📉 空單進場！{price:,.0f}  訊號：{', '.join(triggered)}")
                else:
                    print(f"  多頭趨勢，空單暫停")

            if pos_long:
                state["position_long"]  = pos_long.to_dict()
            if pos_short:
                state["position_short"] = pos_short.to_dict()
            save_state(state)

        except KeyboardInterrupt:
            print("\n⏹ 手動停止")
            break
        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(CONFIG["check_interval_sec"])

if __name__ == "__main__":
    run()
