"""
ETH 多空雙向模擬交易機器人 v4
- 單次趨勢只進場一次（MACD 交叉或 BB 觸軌反轉視為一次趨勢）
- 訊號反轉後重置，才能偵測下一次趨勢
- 成交量確認：反轉當根量 > 前20根均量 1.2 倍
- 多空各自最多 4 筆，每筆使用該方向資金 1/4
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
    "trade_ratio":   0.25,
    "max_positions": 4,
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
    "vol_ma_period":   20,   # 成交量均線週期
    "vol_multiplier":  1.2,  # 成交量需大於均量的倍數
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
            cur.execute("ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS position_id TEXT;")
        conn.commit()
        print("[DB] 資料表初始化完成")
    except Exception as e:
        print(f"[DB] 初始化失敗：{e}")
    finally:
        conn.close()

def _serialize(obj):
    if isinstance(obj, (bool, np.bool_)):   return bool(obj)
    if isinstance(obj, np.integer):          return int(obj)
    if isinstance(obj, np.floating):         return float(obj)
    if isinstance(obj, dict):                return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):       return [_serialize(i) for i in obj]
    return obj

def db_get(key, default=None):
    conn = get_db_conn()
    if not conn: return default
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
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO state (key, value, updated_at) VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """, (key, json.dumps(_serialize(value))))
        conn.commit()
    except Exception as e:
        print(f"[DB] 寫入失敗 {key}：{e}")
    finally:
        conn.close()

def db_log_trade(side, action, price, pnl, reason, signals, position_id=""):
    conn = get_db_conn()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trade_log (time, side, action, price, pnl_pct, reason, signals, position_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (datetime.now(), side, action, float(price),
                  round(float(pnl) * 100, 2), reason,
                  json.dumps(_serialize(signals)), position_id))
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
    avg_g = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    avg_l = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_g / avg_l.replace(0, np.nan)))

def calc_stoch_rsi(close, period=14, sk=3, sd=3):
    rsi = calc_rsi(close, period)
    mn, mx = rsi.rolling(period).min(), rsi.rolling(period).max()
    k = ((rsi - mn) / (mx - mn).replace(0, np.nan) * 100).rolling(sk).mean()
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
    df["vol_ma"] = df["volume"].rolling(c["vol_ma_period"]).mean()
    return df

# ── 趨勢判斷 ─────────────────────────────────────────────────────────────────

def check_trend(df_4h, df_1d):
    if df_4h.empty or df_1d.empty: return "neutral"
    df_4h = add_indicators(df_4h)
    df_1d = add_indicators(df_1d)
    m4, h4 = float(df_4h["macd"].iloc[-1]), float(df_4h["macd_hist"].iloc[-1])
    m1d    = float(df_1d["macd"].iloc[-1])
    if m4 > 0 and h4 > 0 and m1d > 0: return "bull"
    if m4 < 0 and h4 < 0 and m1d < 0: return "bear"
    return "neutral"

# ── 成交量確認 ────────────────────────────────────────────────────────────────

def volume_confirmed(df) -> bool:
    """當根收盤量 > 前N根均量 × 倍數"""
    row = df.iloc[-1]
    if pd.isna(row["vol_ma"]) or row["vol_ma"] == 0:
        return True  # 均量尚未建立，不過濾
    confirmed = bool(row["volume"] > row["vol_ma"] * CONFIG["vol_multiplier"])
    return confirmed

# ── 趨勢訊號狀態機 ────────────────────────────────────────────────────────────
#
# 每個訊號（MACD 金叉/死叉、BB 觸下軌/上軌）都有自己的狀態：
#   "idle"     : 等待訊號出現
#   "fired"    : 訊號已觸發並進場，等待反向訊號重置
#
# 重置條件：
#   MACD 多頭（金叉）→ 死叉出現後重置
#   MACD 空頭（死叉）→ 金叉出現後重置
#   BB 多頭（觸下軌）→ 價格回到中線以上後重置
#   BB 空頭（觸上軌）→ 價格回到中線以下後重置
#
# 任兩個訊號同時 fire → 進場（加上成交量確認）

def update_signal_states(df, states: dict) -> dict:
    """
    更新所有訊號狀態，回傳新狀態與本次可用的進場訊號
    states 格式：
    {
        "macd_long":  "idle" | "fired",
        "macd_short": "idle" | "fired",
        "bb_long":    "idle" | "fired",
        "bb_short":   "idle" | "fired",
        "stoch_long": "idle" | "fired",
        "stoch_short":"idle" | "fired",
    }
    """
    r, p = df.iloc[-1], df.iloc[-2]
    new_states  = dict(states)
    fire_long   = {}   # 本根 K 棒新觸發的多頭訊號
    fire_short  = {}   # 本根 K 棒新觸發的空頭訊號

    # ── MACD ────────────────────────────────────────────────
    macd_golden = bool(r["macd"] > r["macd_signal"] and p["macd"] <= p["macd_signal"])  # 金叉
    macd_dead   = bool(r["macd"] < r["macd_signal"] and p["macd"] >= p["macd_signal"])  # 死叉

    # 多頭：金叉觸發；死叉出現後重置
    if new_states["macd_long"] == "fired" and macd_dead:
        new_states["macd_long"] = "idle"
        print("  🔄 MACD 多頭訊號重置（死叉）")
    if new_states["macd_long"] == "idle" and macd_golden:
        new_states["macd_long"] = "fired"
        fire_long["macd"] = True

    # 空頭：死叉觸發；金叉出現後重置
    if new_states["macd_short"] == "fired" and macd_golden:
        new_states["macd_short"] = "idle"
        print("  🔄 MACD 空頭訊號重置（金叉）")
    if new_states["macd_short"] == "idle" and macd_dead:
        new_states["macd_short"] = "fired"
        fire_short["macd"] = True

    # ── BB 線 ────────────────────────────────────────────────
    bb_touch_lower = bool(p["close"] <= p["bb_lower"] * 1.005 and
                          r["close"] > r["bb_lower"] and r["close"] > p["close"])
    bb_touch_upper = bool(p["close"] >= p["bb_upper"] * 0.995 and
                          r["close"] < r["bb_upper"] and r["close"] < p["close"])
    price_above_mid = bool(r["close"] > r["bb_middle"])
    price_below_mid = bool(r["close"] < r["bb_middle"])

    # 多頭：觸下軌反彈觸發；價格回到中線以上後重置
    if new_states["bb_long"] == "fired" and price_above_mid:
        new_states["bb_long"] = "idle"
        print("  🔄 BB 多頭訊號重置（價格回中線以上）")
    if new_states["bb_long"] == "idle" and bb_touch_lower:
        new_states["bb_long"] = "fired"
        fire_long["bb"] = True

    # 空頭：觸上軌反轉觸發；價格跌回中線以下後重置
    if new_states["bb_short"] == "fired" and price_below_mid:
        new_states["bb_short"] = "idle"
        print("  🔄 BB 空頭訊號重置（價格跌回中線以下）")
    if new_states["bb_short"] == "idle" and bb_touch_upper:
        new_states["bb_short"] = "fired"
        fire_short["bb"] = True

    # ── Stoch RSI ────────────────────────────────────────────
    stoch_golden = bool(p["stoch_k"] < 20 and r["stoch_k"] > r["stoch_d"] and
                        p["stoch_k"] <= p["stoch_d"])
    stoch_dead   = bool(p["stoch_k"] > 80 and r["stoch_k"] < r["stoch_d"] and
                        p["stoch_k"] >= p["stoch_d"])
    stoch_mid_up   = bool(r["stoch_k"] > 50)
    stoch_mid_down = bool(r["stoch_k"] < 50)

    # 多頭：從超賣區上穿觸發；Stoch K 回到 50 以下後重置
    if new_states["stoch_long"] == "fired" and stoch_mid_down:
        new_states["stoch_long"] = "idle"
        print("  🔄 Stoch RSI 多頭訊號重置（K 線跌回 50 以下）")
    if new_states["stoch_long"] == "idle" and stoch_golden:
        new_states["stoch_long"] = "fired"
        fire_long["stoch_rsi"] = True

    # 空頭：從超買區下穿觸發；Stoch K 回到 50 以上後重置
    if new_states["stoch_short"] == "fired" and stoch_mid_up:
        new_states["stoch_short"] = "idle"
        print("  🔄 Stoch RSI 空頭訊號重置（K 線升回 50 以上）")
    if new_states["stoch_short"] == "idle" and stoch_dead:
        new_states["stoch_short"] = "fired"
        fire_short["stoch_rsi"] = True

    return new_states, fire_long, fire_short

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
        return ((price - self.entry_price) / self.entry_price if self.side == "long"
                else (self.entry_price - price) / self.entry_price)

    def should_stop_loss(self, price):
        return self.pnl_pct(price) <= -CONFIG["stop_loss_pct"]

    def check_trailing(self, price):
        price = float(price)
        if self.pnl_pct(price) >= CONFIG["trail_start_pct"]:
            self.trailing_active = True
        if self.trailing_active:
            retrace = ((self.extreme_price - price) / self.extreme_price if self.side == "long"
                       else (price - self.extreme_price) / self.extreme_price)
            return retrace >= CONFIG["trail_drop_pct"]
        return False

    def to_dict(self):
        return {
            "pos_id": self.pos_id, "side": self.side,
            "entry_price": self.entry_price, "capital": self.capital,
            "quantity": self.quantity, "entry_time": self.entry_time.isoformat(),
            "extreme_price": self.extreme_price,
            "trailing_active": bool(self.trailing_active),
            "entry_signals": _serialize(self.entry_signals),
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

DEFAULT_SIGNAL_STATES = {
    "macd_long": "idle", "macd_short": "idle",
    "bb_long":   "idle", "bb_short":   "idle",
    "stoch_long":"idle", "stoch_short":"idle",
}

def load_state():
    data = db_get("bot_state")
    if data:
        # 補上新欄位（舊版相容）
        if "signal_states" not in data:
            data["signal_states"] = DEFAULT_SIGNAL_STATES.copy()
        return data
    if os.path.exists("state.json"):
        with open("state.json") as f:
            data = json.load(f)
            if "signal_states" not in data:
                data["signal_states"] = DEFAULT_SIGNAL_STATES.copy()
            return data
    return {
        "capital_long":   CONFIG["initial_capital_long"],
        "capital_short":  CONFIG["initial_capital_short"],
        "positions_long":  [], "positions_short": [],
        "signal_states":  DEFAULT_SIGNAL_STATES.copy(),
        "total_trades": 0, "win_trades": 0,  "total_pnl": 0.0,
        "long_trades":  0, "long_wins":  0,  "long_pnl":  0.0,
        "short_trades": 0, "short_wins": 0,  "short_pnl": 0.0,
    }

def save_state(state):
    db_set("bot_state", state)
    with open("state.json", "w") as f:
        json.dump(_serialize(state), f, ensure_ascii=False, indent=2)

# ── 顯示狀態 ─────────────────────────────────────────────────────────────────

def print_status(state, price, trend, longs, shorts, sig_states):
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = state["total_trades"]
    wr    = state["win_trades"] / total * 100 if total > 0 else 0

    def ss(key): return "🟢" if sig_states.get(key) == "fired" else "⚪"

    print(f"\n{'='*65}")
    print(f"  ETH 多空模擬交易 v4  |  {now}")
    print(f"{'='*65}")
    print(f"  價格 {price:,.0f} {CONFIG['quote_currency']}  |  "
          f"趨勢 {'🟢多頭' if trend=='bull' else '🔴空頭' if trend=='bear' else '⚪中性'}  |  "
          f"累計 {total}筆 勝率{wr:.1f}% 損益{state['total_pnl']:+.2f}")
    print(f"  訊號狀態  多頭：MACD{ss('macd_long')} BB{ss('bb_long')} Stoch{ss('stoch_long')}  "
          f"空頭：MACD{ss('macd_short')} BB{ss('bb_short')} Stoch{ss('stoch_short')}")

    print(f"\n  【做多】資金 {state['capital_long']:,.2f}  持倉 {len(longs)}/{CONFIG['max_positions']}  損益 {state['long_pnl']:+.2f}")
    for i, pos in enumerate(longs, 1):
        pnl = pos.pnl_pct(price) * 100
        print(f"    #{i} 進場 {pos.entry_price:,.0f}  資金 {pos.capital:,.0f}  "
              f"損益 {pnl:+.2f}%  滾動{'✅' if pos.trailing_active else '⏳'}")
    if not longs: print(f"    空倉，等待多頭訊號")

    print(f"\n  【做空】資金 {state['capital_short']:,.2f}  持倉 {len(shorts)}/{CONFIG['max_positions']}  損益 {state['short_pnl']:+.2f}")
    for i, pos in enumerate(shorts, 1):
        pnl = pos.pnl_pct(price) * 100
        print(f"    #{i} 進場 {pos.entry_price:,.0f}  資金 {pos.capital:,.0f}  "
              f"損益 {pnl:+.2f}%  滾動{'✅' if pos.trailing_active else '⏳'}")
    if not shorts: print(f"    空倉，等待空頭訊號")
    print(f"{'='*65}")

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
        state["win_trades"]  += 1
        state[f"{sk}_wins"]  += 1
    icon  = "📤" if sk == "long" else "📥"
    label = "多單" if sk == "long" else "空單"
    print(f"\n  {icon} {label}出場 #{pos.pos_id[-6:]}  {reason}")
    print(f"     {pos.entry_price:,.0f}→{price:,.0f}  {pnl*100:+.2f}% ({pnl_usdt:+.2f} USDT)")
    db_log_trade(sk, "EXIT", price, pnl, reason, {}, pos.pos_id)

# ── 主循環 ───────────────────────────────────────────────────────────────────

def run():
    print("🚀 ETH 多空雙向模擬交易機器人 v4 啟動")
    print(f"   進場：單次趨勢只進一次，訊號反轉後重置")
    print(f"   成交量確認：需大於 {CONFIG['vol_ma_period']} 根均量 × {CONFIG['vol_multiplier']}")
    init_db()

    state  = load_state()

    # 舊版相容
    if "position_long" in state and "positions_long" not in state:
        state["positions_long"]  = [state.pop("position_long")]  if state.get("position_long")  else []
        state["positions_short"] = [state.pop("position_short")] if state.get("position_short") else []

    longs      = [Position.from_dict(d) for d in state.get("positions_long",  [])]
    shorts     = [Position.from_dict(d) for d in state.get("positions_short", [])]
    sig_states = state.get("signal_states", DEFAULT_SIGNAL_STATES.copy())

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

            # 更新訊號狀態機
            sig_states, fire_long, fire_short = update_signal_states(df_30m, sig_states)
            state["signal_states"] = sig_states

            print_status(state, price, trend, longs, shorts, sig_states)

            vol_ok = volume_confirmed(df_30m)
            if not vol_ok:
                print(f"  ⚠ 成交量不足（{df_30m['volume'].iloc[-1]:,.0f} < 均量{df_30m['vol_ma'].iloc[-1]:,.0f} × {CONFIG['vol_multiplier']}），本根不進場")

            # ── 多單管理 ──────────────────────────────────────────────────────
            weak_long = check_long_exit(df_30m)
            exited = []
            for pos in longs:
                pos.update_extreme(price)
                reason = None
                if pos.should_stop_loss(price):
                    reason = "停損"
                elif pos.trailing_active and weak_long["any"]:
                    names  = [k for k, v in weak_long.items() if v and k != "any"]
                    reason = f"指標轉弱（{'、'.join(names)}）"
                elif pos.check_trailing(price):
                    reason = "滾動停利觸發"
                if reason:
                    process_exit(state, pos, price, reason)
                    exited.append(pos)
            longs = [p for p in longs if p not in exited]

            # 多頭進場：有新訊號 + 成交量確認 + 趨勢允許 + 未滿倉
            if fire_long and len(fire_long) >= 1 and vol_ok and \
               len(longs) < CONFIG["max_positions"] and trend in ("bull", "neutral"):
                # 需要至少兩個訊號（含已 fired 的舊訊號）
                total_long_fired = sum(1 for k in ["macd_long","bb_long","stoch_long"]
                                       if sig_states.get(k) == "fired")
                if total_long_fired >= 2:
                    cap = state["capital_long"] * CONFIG["trade_ratio"]
                    if cap > 0:
                        triggered = list(fire_long.keys())
                        pos = Position("long", price, cap, datetime.now(), fire_long)
                        longs.append(pos)
                        state["capital_long"] -= cap
                        db_log_trade("long", "ENTRY", price, 0,
                                     f"新訊號：{', '.join(triggered)}", fire_long, pos.pos_id)
                        print(f"\n  📈 多單進場 #{pos.pos_id[-6:]}  資金 {cap:,.0f}  "
                              f"新訊號：{', '.join(triggered)}  已觸發 {total_long_fired}/3")
                else:
                    print(f"  多頭新訊號 {list(fire_long.keys())}，但累計觸發 {total_long_fired}/3，等待更多確認")
            elif trend == "bear":
                print(f"  空頭趨勢，多單暫停")

            # ── 空單管理 ──────────────────────────────────────────────────────
            weak_short = check_short_exit(df_30m)
            exited = []
            for pos in shorts:
                pos.update_extreme(price)
                reason = None
                if pos.should_stop_loss(price):
                    reason = "停損"
                elif pos.trailing_active and weak_short["any"]:
                    names  = [k for k, v in weak_short.items() if v and k != "any"]
                    reason = f"指標轉強（{'、'.join(names)}）"
                elif pos.check_trailing(price):
                    reason = "滾動停利觸發"
                if reason:
                    process_exit(state, pos, price, reason)
                    exited.append(pos)
            shorts = [p for p in shorts if p not in exited]

            # 空頭進場：有新訊號 + 成交量確認 + 趨勢允許 + 未滿倉
            if fire_short and len(fire_short) >= 1 and vol_ok and \
               len(shorts) < CONFIG["max_positions"] and trend in ("bear", "neutral"):
                total_short_fired = sum(1 for k in ["macd_short","bb_short","stoch_short"]
                                        if sig_states.get(k) == "fired")
                if total_short_fired >= 2:
                    cap = state["capital_short"] * CONFIG["trade_ratio"]
                    if cap > 0:
                        triggered = list(fire_short.keys())
                        pos = Position("short", price, cap, datetime.now(), fire_short)
                        shorts.append(pos)
                        state["capital_short"] -= cap
                        db_log_trade("short", "ENTRY", price, 0,
                                     f"新訊號：{', '.join(triggered)}", fire_short, pos.pos_id)
                        print(f"\n  📉 空單進場 #{pos.pos_id[-6:]}  資金 {cap:,.0f}  "
                              f"新訊號：{', '.join(triggered)}  已觸發 {total_short_fired}/3")
                else:
                    print(f"  空頭新訊號 {list(fire_short.keys())}，但累計觸發 {total_short_fired}/3，等待更多確認")
            elif trend == "bull":
                print(f"  多頭趨勢，空單暫停")

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
