"""
ETH 多空雙向模擬交易機器人 v3
- 多空各自最多 4 筆持倉
- 每筆進場使用當前該方向資金的 1/4
- 紀錄儲存於 PostgreSQL
"""

import requests
import pandas as pd
import numpy as np
import os, time, json
from datetime import datetime

try:
    import psycopg2
    HAS_PG = True
except ImportError:
    HAS_PG = False

# ── 設定區 ───────────────────────────────────────────────────────────────────

CONFIG = {
    "initial_capital_long":  1000,
    "initial_capital_short": 1000,
    "trade_ratio":   0.25,       # 每次進場使用該方向資金的 1/4
    "max_positions": 4,          # 同方向最多持倉數
    "symbol":        "ethtwd",
    "quote_currency":"TWD",
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

# ── 資料庫 ───────────────────────────────────────────────────────────────────

def get_db_conn():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url or not HAS_PG:
        return None
    try:
        return psycopg2.connect(db_url, sslmode="require")
    except Exception as e:
        print(f"[DB] 連線失敗：{e}")
        return None

def init_db():
    conn = get_db_conn()
    if not conn:
        print("[DB] 無法連線，使用本機檔案儲存")
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
                    position_id TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            # 補欄位（舊資料庫相容）
            cur.execute("""
                ALTER TABLE trade_log
                ADD COLUMN IF NOT EXISTS position_id TEXT;
            """)
        conn.commit()
        print("[DB] 資料表初始化完成")
    except Exception as e:
        print(f"[DB] 初始化失敗：{e}")
    finally:
        conn.close()

def _serialize(obj):
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    return obj

def db_get(key, default=None):
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

def db_set(key, value):
    conn = get_db_conn()
    if not conn:
        return
    try:
        serialized = json.dumps(_serialize(value))
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

def db_log_trade(side, action, price, pnl, reason, signals, position_id=""):
    conn = get_db_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trade_log
                (time, side, action, price, pnl_pct, reason, signals, position_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                datetime.now(), side, action,
                float(price), round(float(pnl) * 100, 2),
                reason, json.dumps(_serialize(signals)), position_id
            ))
        conn.commit()
    except Exception as e:
        print(f"[DB] 寫入交易紀錄失敗：{e}")
    finally:
        conn.close()

# ── MAX API ──────────────────────────────────────────────────────────────────

def fetch_klines(symbol, period, limit=200):
    period_map = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
    try:
        resp = requests.get(f"{MAX_API_BASE}/k",
                            params={"market": symbol,
                                    "period": period_map.get(period, 30),
                                    "limit": limit},
                            timeout=10)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json(),
                          columns=["timestamp","open","high","low","close","volume"])
        df = df.astype({c: float for c in ["open","high","low","close","volume"]})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] K 線失敗 ({symbol} {period}): {e}")
        return pd.DataFrame()

# ── 技術指標 ─────────────────────────────────────────────────────────────────

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period-1, min_periods=period).mean()
    avg_l = loss.ewm(com=period-1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_stoch_rsi(close, period=14, sk=3, sd=3):
    rsi = calc_rsi(close, period)
    mn  = rsi.rolling(period).min()
    mx  = rsi.rolling(period).max()
    k   = ((rsi - mn) / (mx - mn).replace(0, np.nan) * 100).rolling(sk).mean()
    return k, k.rolling(sd).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    ml = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl

def calc_bb(close, period=20, mult=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + mult*std, mid, mid - mult*std

def add_indicators(df):
    c = CONFIG
    df["stoch_k"], df["stoch_d"] = calc_stoch_rsi(
        df["close"], c["stoch_rsi_period"], c["stoch_rsi_smooth_k"], c["stoch_rsi_smooth_d"])
    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(
        df["close"], c["macd_fast"], c["macd_slow"], c["macd_signal"])
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = calc_bb(
        df["close"], c["bb_period"], c["bb_std"])
    return df

# ── 趨勢 ─────────────────────────────────────────────────────────────────────

def check_trend(df_4h, df_1d):
    if df_4h.empty or df_1d.empty:
        return "neutral"
    df_4h = add_indicators(df_4h)
    df_1d = add_indicators(df_1d)
    m4, h4 = float(df_4h["macd"].iloc[-1]), float(df_4h["macd_hist"].iloc[-1])
    m1d    = float(df_1d["macd"].iloc[-1])
    if m4 > 0 and h4 > 0 and m1d > 0: return "bull"
    if m4 < 0 and h4 < 0 and m1d < 0: return "bear"
    return "neutral"

# ── 進場訊號 ─────────────────────────────────────────────────────────────────

def check_long_signals(df):
    r, p = df.iloc[-1], df.iloc[-2]
    s1 = bool(p["stoch_k"] < 20 and r["stoch_k"] > r["stoch_d"] and p["stoch_k"] <= p["stoch_d"])
    s2 = bool(r["macd"] > r["macd_signal"] and p["macd"] <= p["macd_signal"] and r["macd_hist"] > 0)
    s3 = bool(p["close"] <= p["bb_lower"]*1.005 and r["close"] > r["bb_lower"] and r["close"] > p["close"])
    cnt = sum([s1, s2, s3])
    return {"stoch_rsi": s1, "macd": s2, "bb": s3, "count": cnt, "entry": bool(cnt >= 2)}

def check_short_signals(df):
    r, p = df.iloc[-1], df.iloc[-2]
    s1 = bool(p["stoch_k"] > 80 and r["stoch_k"] < r["stoch_d"] and p["stoch_k"] >= p["stoch_d"])
    s2 = bool(r["macd"] < r["macd_signal"] and p["macd"] >= p["macd_signal"] and r["macd_hist"] < 0)
    s3 = bool(p["close"] >= p["bb_upper"]*0.995 and r["close"] < r["bb_upper"] and r["close"] < p["close"])
    cnt = sum([s1, s2, s3])
    return {"stoch_rsi": s1, "macd": s2, "bb": s3, "count": cnt, "entry": bool(cnt >= 2)}

# ── 出場訊號 ─────────────────────────────────────────────────────────────────

def check_long_exit(df):
    r, p = df.iloc[-1], df.iloc[-2]
    w = {
        "stoch_rsi": bool(p["stoch_k"] > 80 and r["stoch_k"] < r["stoch_d"]),
        "macd":      bool((r["macd"] < r["macd_signal"] and p["macd"] >= p["macd_signal"]) or
                          (r["macd_hist"] < 0 and p["macd_hist"] >= 0)),
        "bb":        bool(p["close"] >= p["bb_upper"]*0.995 and r["close"] < r["bb_middle"]),
    }
    w["any"] = bool(any([w["stoch_rsi"], w["macd"], w["bb"]]))
    return w

def check_short_exit(df):
    r, p = df.iloc[-1], df.iloc[-2]
    w = {
        "stoch_rsi": bool(p["stoch_k"] < 20 and r["stoch_k"] > r["stoch_d"]),
        "macd":      bool((r["macd"] > r["macd_signal"] and p["macd"] <= p["macd_signal"]) or
                          (r["macd_hist"] > 0 and p["macd_hist"] <= 0)),
        "bb":        bool(p["close"] <= p["bb_lower"]*1.005 and r["close"] > r["bb_middle"]),
    }
    w["any"] = bool(any([w["stoch_rsi"], w["macd"], w["bb"]]))
    return w

# ── 持倉類別 ─────────────────────────────────────────────────────────────────

class Position:
    def __init__(self, side, entry_price, capital, entry_time, signals, pos_id=None):
        self.side            = side
        self.entry_price     = float(entry_price)
        self.capital         = float(capital)
        self.entry_time      = entry_time
        self.entry_signals   = signals
        self.quantity        = float(capital) / float(entry_price)
        self.trailing_active = False
        self.extreme_price   = float(entry_price)
        self.pos_id          = pos_id or datetime.now().strftime("%Y%m%d%H%M%S%f")

    def update_extreme(self, price):
        price = float(price)
        if self.side == "long":
            if price > self.extreme_price: self.extreme_price = price
        else:
            if price < self.extreme_price: self.extreme_price = price

    def pnl_pct(self, price):
        price = float(price)
        if self.side == "long":
            return (price - self.entry_price) / self.entry_price
        return (self.entry_price - price) / self.entry_price

    def should_stop_loss(self, price):
        return self.pnl_pct(price) <= -CONFIG["stop_loss_pct"]

    def check_trailing(self, price):
        price = float(price)
        pnl   = self.pnl_pct(price)
        if pnl >= CONFIG["trail_start_pct"]:
            self.trailing_active = True
        if self.trailing_active:
            retrace = ((self.extreme_price - price) / self.extreme_price if self.side == "long"
                       else (price - self.extreme_price) / self.extreme_price)
            return retrace >= CONFIG["trail_drop_pct"]
        return False

    def to_dict(self):
        return {
            "pos_id":          self.pos_id,
            "side":            self.side,
            "entry_price":     self.entry_price,
            "capital":         self.capital,
            "quantity":        self.quantity,
            "entry_time":      self.entry_time.isoformat(),
            "extreme_price":   self.extreme_price,
            "trailing_active": bool(self.trailing_active),
            "entry_signals":   _serialize(self.entry_signals),
        }

    @classmethod
    def from_dict(cls, d):
        p = cls(d["side"], d["entry_price"], d["capital"],
                datetime.fromisoformat(d["entry_time"]),
                d.get("entry_signals", {}), d.get("pos_id"))
        p.quantity        = d["quantity"]
        p.extreme_price   = d["extreme_price"]
        p.trailing_active = d["trailing_active"]
        return p

# ── 狀態管理 ─────────────────────────────────────────────────────────────────

def load_state():
    data = db_get("bot_state")
    if data:
        return data
    if os.path.exists("state.json"):
        with open("state.json") as f:
            return json.load(f)
    return {
        "capital_long":    CONFIG["initial_capital_long"],
        "capital_short":   CONFIG["initial_capital_short"],
        "positions_long":  [],
        "positions_short": [],
        "total_trades": 0, "win_trades": 0, "total_pnl": 0.0,
        "long_trades":  0, "long_wins":  0, "long_pnl":  0.0,
        "short_trades": 0, "short_wins": 0, "short_pnl": 0.0,
    }

def save_state(state):
    db_set("bot_state", state)
    with open("state.json", "w") as f:
        json.dump(_serialize(state), f, ensure_ascii=False, indent=2)

# ── 顯示狀態 ─────────────────────────────────────────────────────────────────

def print_status(state, price, trend, longs, shorts):
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = state["total_trades"]
    wr    = state["win_trades"] / total * 100 if total > 0 else 0
    print(f"\n{'='*62}")
    print(f"  ETH 多空模擬交易  |  {now}")
    print(f"{'='*62}")
    print(f"  價格 {price:,.0f} {CONFIG['quote_currency']}  |  "
          f"趨勢 {'🟢多頭' if trend=='bull' else '🔴空頭' if trend=='bear' else '⚪中性'}  |  "
          f"累計 {total}筆 勝率{wr:.1f}% 損益{state['total_pnl']:+.2f}")

    # 多單
    print(f"\n  【做多】可用資金 {state['capital_long']:,.2f} USDT  "
          f"持倉 {len(longs)}/{CONFIG['max_positions']}  "
          f"累計損益 {state['long_pnl']:+.2f} USDT")
    for i, pos in enumerate(longs, 1):
        pnl = pos.pnl_pct(price) * 100
        print(f"    #{i} 進場 {pos.entry_price:,.0f}  "
              f"資金 {pos.capital:,.0f}  "
              f"損益 {pnl:+.2f}%  "
              f"滾動{'✅' if pos.trailing_active else '⏳'}")
    if not longs:
        print(f"    空倉，等待多頭訊號")

    # 空單
    print(f"\n  【做空】可用資金 {state['capital_short']:,.2f} USDT  "
          f"持倉 {len(shorts)}/{CONFIG['max_positions']}  "
          f"累計損益 {state['short_pnl']:+.2f} USDT")
    for i, pos in enumerate(shorts, 1):
        pnl = pos.pnl_pct(price) * 100
        print(f"    #{i} 進場 {pos.entry_price:,.0f}  "
              f"資金 {pos.capital:,.0f}  "
              f"損益 {pnl:+.2f}%  "
              f"滾動{'✅' if pos.trailing_active else '⏳'}")
    if not shorts:
        print(f"    空倉，等待空頭訊號")
    print(f"{'='*62}")

# ── 出場處理 ─────────────────────────────────────────────────────────────────

def process_exit(state, pos, price, reason):
    pnl      = pos.pnl_pct(price)
    pnl_usdt = pnl * pos.capital
    sk       = pos.side
    state[f"capital_{sk}"] += pos.capital + pnl_usdt
    state[f"{sk}_trades"]  += 1
    state[f"{sk}_pnl"]     += pnl_usdt
    state["total_trades"]  += 1
    state["total_pnl"]     += pnl_usdt
    if pnl > 0:
        state["win_trades"]    += 1
        state[f"{sk}_wins"]    += 1
    label = "多單" if sk == "long" else "空單"
    icon  = "📤" if sk == "long" else "📥"
    print(f"\n  {icon} {label}出場 #{pos.pos_id[-6:]}  {reason}")
    print(f"     {pos.entry_price:,.0f}→{price:,.0f}  {pnl*100:+.2f}% ({pnl_usdt:+.2f} USDT)")
    db_log_trade(sk, "EXIT", price, pnl, reason, {}, pos.pos_id)

# ── 主循環 ───────────────────────────────────────────────────────────────────

def run():
    print("🚀 ETH 多空雙向模擬交易機器人 v3 啟動")
    print(f"   每次進場資金：該方向可用資金的 {CONFIG['trade_ratio']*100:.0f}%  |  "
          f"同方向最多 {CONFIG['max_positions']} 筆")
    init_db()

    state  = load_state()

    # 相容舊版（單一持倉 → 陣列）
    if "position_long" in state and "positions_long" not in state:
        old_l = state.pop("position_long", None)
        old_s = state.pop("position_short", None)
        state["positions_long"]  = [old_l] if old_l else []
        state["positions_short"] = [old_s] if old_s else []

    longs  = [Position.from_dict(d) for d in state.get("positions_long",  [])]
    shorts = [Position.from_dict(d) for d in state.get("positions_short", [])]

    while True:
        try:
            df_30m = fetch_klines(CONFIG["symbol"], CONFIG["main_period"],     limit=100)
            df_4h  = fetch_klines(CONFIG["symbol"], CONFIG["trend_period_4h"], limit=100)
            df_1d  = fetch_klines(CONFIG["symbol"], CONFIG["trend_period_1d"], limit=100)

            if df_30m.empty:
                print("[WARN] 無法取得 K 線，等待重試...")
                time.sleep(CONFIG["check_interval_sec"])
                continue

            price  = float(df_30m["close"].iloc[-1])
            df_30m = add_indicators(df_30m)
            trend  = check_trend(df_4h, df_1d)
            print_status(state, price, trend, longs, shorts)

            # ── 多單管理 ──────────────────────────────────────────────────────
            weak_long = check_long_exit(df_30m)
            exited = []
            for pos in longs:
                pos.update_extreme(price)
                reason = None
                if pos.should_stop_loss(price):
                    reason = "停損"
                else:
                    if pos.trailing_active and weak_long["any"]:
                        names  = [k for k, v in weak_long.items() if v and k != "any"]
                        reason = f"指標轉弱（{'、'.join(names)}）"
                    elif pos.check_trailing(price):
                        reason = "滾動停利觸發"
                if reason:
                    process_exit(state, pos, price, reason)
                    exited.append(pos)
            longs = [p for p in longs if p not in exited]

            if len(longs) < CONFIG["max_positions"] and trend in ("bull", "neutral"):
                sig = check_long_signals(df_30m)
                triggered = [k for k, v in sig.items() if v is True]
                print(f"  多頭訊號：{', '.join(triggered) if triggered else '無'} ({sig['count']}/3)")
                if sig["entry"]:
                    cap = state["capital_long"] * CONFIG["trade_ratio"]
                    if cap > 0:
                        pos = Position("long", price, cap, datetime.now(), sig)
                        longs.append(pos)
                        state["capital_long"] -= cap
                        db_log_trade("long", "ENTRY", price, 0,
                                     f"訊號：{', '.join(triggered)}", sig, pos.pos_id)
                        print(f"\n  📈 多單進場 #{pos.pos_id[-6:]}  "
                              f"資金 {cap:,.0f}  訊號：{', '.join(triggered)}")
            else:
                if trend == "bear":
                    print(f"  空頭趨勢，多單暫停")
                elif len(longs) >= CONFIG["max_positions"]:
                    print(f"  多單已滿 {CONFIG['max_positions']} 筆，等待出場")

            # ── 空單管理 ──────────────────────────────────────────────────────
            weak_short = check_short_exit(df_30m)
            exited = []
            for pos in shorts:
                pos.update_extreme(price)
                reason = None
                if pos.should_stop_loss(price):
                    reason = "停損"
                else:
                    if pos.trailing_active and weak_short["any"]:
                        names  = [k for k, v in weak_short.items() if v and k != "any"]
                        reason = f"指標轉強（{'、'.join(names)}）"
                    elif pos.check_trailing(price):
                        reason = "滾動停利觸發"
                if reason:
                    process_exit(state, pos, price, reason)
                    exited.append(pos)
            shorts = [p for p in shorts if p not in exited]

            if len(shorts) < CONFIG["max_positions"] and trend in ("bear", "neutral"):
                sig = check_short_signals(df_30m)
                triggered = [k for k, v in sig.items() if v is True]
                print(f"  空頭訊號：{', '.join(triggered) if triggered else '無'} ({sig['count']}/3)")
                if sig["entry"]:
                    cap = state["capital_short"] * CONFIG["trade_ratio"]
                    if cap > 0:
                        pos = Position("short", price, cap, datetime.now(), sig)
                        shorts.append(pos)
                        state["capital_short"] -= cap
                        db_log_trade("short", "ENTRY", price, 0,
                                     f"訊號：{', '.join(triggered)}", sig, pos.pos_id)
                        print(f"\n  📉 空單進場 #{pos.pos_id[-6:]}  "
                              f"資金 {cap:,.0f}  訊號：{', '.join(triggered)}")
            else:
                if trend == "bull":
                    print(f"  多頭趨勢，空單暫停")
                elif len(shorts) >= CONFIG["max_positions"]:
                    print(f"  空單已滿 {CONFIG['max_positions']} 筆，等待出場")

            state["positions_long"]  = [p.to_dict() for p in longs]
            state["positions_short"] = [p.to_dict() for p in shorts]
            save_state(state)

        except KeyboardInterrupt:
            print("\n⏹ 手動停止")
            break
        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(CONFIG["check_interval_sec"])

if __name__ == "__main__":
    run()
