"""
ETH 多空雙向模擬交易機器人
使用 MAX 交易所公開 API 取得 K 線資料
指標：Stoch RSI、MACD、BB 線
多單：任兩個指標出現多頭訊號 → 進場做多
空單：任兩個指標出現空頭訊號 → 進場做空
多空獨立計算，可同時持倉
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime

# ── 設定區 ──────────────────────────────────────────────────────────────────

CONFIG = {
    # 資金設定（多空各自使用）
    "initial_capital_long":  1000,   # 做多模擬資金 USDT
    "initial_capital_short": 1000,   # 做空模擬資金 USDT
    "trade_ratio": 1.0,              # 每次使用資金比例（1.0 = 全倉）

    # 交易對
    "symbol": "ethtwd",
    "quote_currency": "TWD",

    # K 線週期
    "main_period":       "30m",
    "trend_period_4h":   "4h",
    "trend_period_1d":   "1d",

    # 風險管理（多空共用）
    "stop_loss_pct":   0.10,   # 停損 10%
    "trail_start_pct": 0.03,   # 獲利 3% 後啟動滾動停利
    "trail_drop_pct":  0.015,  # 從極值回落 1.5% 出場

    # 指標參數
    "stoch_rsi_period":   14,
    "stoch_rsi_smooth_k":  3,
    "stoch_rsi_smooth_d":  3,
    "macd_fast":   12,
    "macd_slow":   26,
    "macd_signal":  9,
    "bb_period":   20,
    "bb_std":       2.0,

    # 執行設定
    "check_interval_sec": 60,
    "log_file":   "trade_log.json",
    "state_file": "state.json",
}

MAX_API_BASE = "https://max-api.maicoin.com/api/v2"

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

# ── 技術指標計算 ─────────────────────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
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
    d = k.rolling(smooth_d).mean()
    return k, d

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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    cfg = CONFIG
    k, d = calc_stoch_rsi(df["close"], cfg["stoch_rsi_period"],
                           cfg["stoch_rsi_smooth_k"], cfg["stoch_rsi_smooth_d"])
    df["stoch_k"] = k
    df["stoch_d"] = d
    macd, sig, hist = calc_macd(df["close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = hist
    upper, mid, lower = calc_bb(df["close"], cfg["bb_period"], cfg["bb_std"])
    df["bb_upper"] = upper
    df["bb_middle"] = mid
    df["bb_lower"] = lower
    return df

# ── 趨勢判斷 ─────────────────────────────────────────────────────────────────

def check_trend(df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> str:
    """
    bull  : 4H MACD > 0 且柱狀圖 > 0，日K MACD > 0
    bear  : 4H MACD < 0 且柱狀圖 < 0，日K MACD < 0
    neutral : 其他
    """
    if df_4h.empty or df_1d.empty:
        return "neutral"
    df_4h = add_indicators(df_4h)
    df_1d = add_indicators(df_1d)
    m4  = df_4h["macd"].iloc[-1]
    h4  = df_4h["macd_hist"].iloc[-1]
    m1d = df_1d["macd"].iloc[-1]
    if m4 > 0 and h4 > 0 and m1d > 0:
        return "bull"
    if m4 < 0 and h4 < 0 and m1d < 0:
        return "bear"
    return "neutral"

# ── 進場訊號 ─────────────────────────────────────────────────────────────────

def check_long_signals(df: pd.DataFrame) -> dict:
    """多頭進場訊號（任兩個滿足）"""
    row  = df.iloc[-1]
    prev = df.iloc[-2]

    # Stoch RSI：從超賣區（<20）K 線上穿 D 線
    s_stoch = (prev["stoch_k"] < 20 and
               row["stoch_k"] > row["stoch_d"] and
               prev["stoch_k"] <= prev["stoch_d"])

    # MACD：金叉且柱狀圖轉正
    s_macd = (row["macd"] > row["macd_signal"] and
              prev["macd"] <= prev["macd_signal"] and
              row["macd_hist"] > 0)

    # BB：前根觸及下軌，當根收回下軌上方
    s_bb = (prev["close"] <= prev["bb_lower"] * 1.005 and
            row["close"] > row["bb_lower"] and
            row["close"] > prev["close"])

    count = sum([s_stoch, s_macd, s_bb])
    return {"stoch_rsi": s_stoch, "macd": s_macd, "bb": s_bb,
            "count": count, "entry": count >= 2}


def check_short_signals(df: pd.DataFrame) -> dict:
    """空頭進場訊號（與多頭完全相反，任兩個滿足）"""
    row  = df.iloc[-1]
    prev = df.iloc[-2]

    # Stoch RSI：從超買區（>80）K 線下穿 D 線
    s_stoch = (prev["stoch_k"] > 80 and
               row["stoch_k"] < row["stoch_d"] and
               prev["stoch_k"] >= prev["stoch_d"])

    # MACD：死叉且柱狀圖轉負
    s_macd = (row["macd"] < row["macd_signal"] and
              prev["macd"] >= prev["macd_signal"] and
              row["macd_hist"] < 0)

    # BB：前根觸及上軌，當根收回上軌下方
    s_bb = (prev["close"] >= prev["bb_upper"] * 0.995 and
            row["close"] < row["bb_upper"] and
            row["close"] < prev["close"])

    count = sum([s_stoch, s_macd, s_bb])
    return {"stoch_rsi": s_stoch, "macd": s_macd, "bb": s_bb,
            "count": count, "entry": count >= 2}

# ── 出場訊號（指標加速出場）────────────────────────────────────────────────────

def check_long_exit_signals(df: pd.DataFrame) -> dict:
    """多單：指標轉弱 → 加速出場"""
    row  = df.iloc[-1]
    prev = df.iloc[-2]
    weak = {
        "stoch_rsi": (prev["stoch_k"] > 80 and row["stoch_k"] < row["stoch_d"]),
        "macd":      ((row["macd"] < row["macd_signal"] and prev["macd"] >= prev["macd_signal"]) or
                      (row["macd_hist"] < 0 and prev["macd_hist"] >= 0)),
        "bb":        (prev["close"] >= prev["bb_upper"] * 0.995 and row["close"] < row["bb_middle"]),
    }
    weak["any"] = any(weak.values())
    return weak


def check_short_exit_signals(df: pd.DataFrame) -> dict:
    """空單：指標轉強 → 加速出場（方向與多單相反）"""
    row  = df.iloc[-1]
    prev = df.iloc[-2]
    weak = {
        # Stoch RSI 從超賣區向上穿越（空單反轉訊號）
        "stoch_rsi": (prev["stoch_k"] < 20 and row["stoch_k"] > row["stoch_d"]),
        # MACD 金叉或柱狀圖轉正
        "macd":      ((row["macd"] > row["macd_signal"] and prev["macd"] <= prev["macd_signal"]) or
                      (row["macd_hist"] > 0 and prev["macd_hist"] <= 0)),
        # 價格從下軌附近反彈回中軌以上
        "bb":        (prev["close"] <= prev["bb_lower"] * 1.005 and row["close"] > row["bb_middle"]),
    }
    weak["any"] = any(weak.values())
    return weak

# ── 持倉類別（多空通用）──────────────────────────────────────────────────────

class Position:
    def __init__(self, side: str, entry_price: float, capital: float,
                 entry_time: datetime, signals: dict):
        self.side         = side          # "long" 或 "short"
        self.entry_price  = entry_price
        self.capital      = capital
        self.entry_time   = entry_time
        self.entry_signals = signals
        self.quantity     = capital / entry_price
        self.trailing_active = False
        # 多單追蹤最高價；空單追蹤最低價
        self.extreme_price = entry_price

    def update_extreme(self, price: float):
        if self.side == "long":
            if price > self.extreme_price:
                self.extreme_price = price
        else:
            if price < self.extreme_price:
                self.extreme_price = price

    def current_pnl_pct(self, price: float) -> float:
        if self.side == "long":
            return (price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - price) / self.entry_price

    def should_stop_loss(self, price: float) -> bool:
        return self.current_pnl_pct(price) <= -CONFIG["stop_loss_pct"]

    def check_trailing(self, price: float) -> bool:
        pnl = self.current_pnl_pct(price)
        if pnl >= CONFIG["trail_start_pct"]:
            self.trailing_active = True
        if self.trailing_active:
            if self.side == "long":
                retracement = (self.extreme_price - price) / self.extreme_price
            else:
                retracement = (price - self.extreme_price) / self.extreme_price
            return retracement >= CONFIG["trail_drop_pct"]
        return False

    def to_dict(self) -> dict:
        return {
            "side":          self.side,
            "entry_price":   self.entry_price,
            "capital":       self.capital,
            "quantity":      self.quantity,
            "entry_time":    self.entry_time.isoformat(),
            "extreme_price": self.extreme_price,
            "trailing_active": self.trailing_active,
            "entry_signals": self.entry_signals,
        }

    @classmethod
    def from_dict(cls, d: dict):
        p = cls(d["side"], d["entry_price"], d["capital"],
                datetime.fromisoformat(d["entry_time"]), d.get("entry_signals", {}))
        p.quantity        = d["quantity"]
        p.extreme_price   = d["extreme_price"]
        p.trailing_active = d["trailing_active"]
        return p

# ── 狀態與紀錄 ───────────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(CONFIG["state_file"]):
        with open(CONFIG["state_file"], "r") as f:
            return json.load(f)
    return {
        "capital_long":  CONFIG["initial_capital_long"],
        "capital_short": CONFIG["initial_capital_short"],
        "position_long":  None,
        "position_short": None,
        "total_trades": 0,
        "win_trades":   0,
        "total_pnl":    0.0,
        # 多空分開統計
        "long_trades":  0, "long_wins":  0, "long_pnl":  0.0,
        "short_trades": 0, "short_wins": 0, "short_pnl": 0.0,
    }

def save_state(state: dict):
    with open(CONFIG["state_file"], "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def log_trade(side: str, action: str, price: float, pnl: float, reason: str, signals: dict):
    entry = {
        "time":    datetime.now().isoformat(),
        "side":    side,
        "action":  action,
        "price":   price,
        "pnl_pct": round(pnl * 100, 2),
        "reason":  reason,
        "signals": signals,
    }
    logs = []
    if os.path.exists(CONFIG["log_file"]):
        with open(CONFIG["log_file"], "r") as f:
            logs = json.load(f)
    logs.append(entry)
    with open(CONFIG["log_file"], "w") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

# ── 顯示狀態 ─────────────────────────────────────────────────────────────────

def print_status(state: dict, price: float, trend: str,
                 pos_long: Position, pos_short: Position):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = state["total_trades"]
    wr = state["win_trades"] / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  ETH 多空模擬交易機器人  |  {now}")
    print(f"{'='*60}")
    print(f"  當前價格   : {price:,.0f} {CONFIG['quote_currency']}")
    print(f"  大趨勢方向 : {'🟢 多頭' if trend=='bull' else '🔴 空頭' if trend=='bear' else '⚪ 中性'}")
    print(f"  累計交易   : {total} 筆  |  勝率 {wr:.1f}%  |  累計損益 {state['total_pnl']:+.2f} USDT")

    # 多單狀態
    print(f"\n  【做多】資金 {state['capital_long']:,.2f} USDT  |  {state['long_trades']}筆 {state['long_pnl']:+.2f} USDT")
    if pos_long:
        pnl = pos_long.current_pnl_pct(price) * 100
        print(f"    ▶ 持倉中  進場 {pos_long.entry_price:,.0f}  |  損益 {pnl:+.2f}%  |  "
              f"極值 {pos_long.extreme_price:,.0f}  |  滾動停利 {'✅' if pos_long.trailing_active else '⏳'}")
    else:
        print(f"    ▶ 空倉，等待多頭訊號")

    # 空單狀態
    print(f"\n  【做空】資金 {state['capital_short']:,.2f} USDT  |  {state['short_trades']}筆 {state['short_pnl']:+.2f} USDT")
    if pos_short:
        pnl = pos_short.current_pnl_pct(price) * 100
        print(f"    ▶ 持倉中  進場 {pos_short.entry_price:,.0f}  |  損益 {pnl:+.2f}%  |  "
              f"極值 {pos_short.extreme_price:,.0f}  |  滾動停利 {'✅' if pos_short.trailing_active else '⏳'}")
    else:
        print(f"    ▶ 空倉，等待空頭訊號")

    print(f"{'='*60}")

# ── 出場處理（多空共用）──────────────────────────────────────────────────────

def process_exit(state: dict, pos: Position, price: float,
                 exit_reason: str) -> None:
    pnl = pos.current_pnl_pct(price)
    pnl_usdt = pnl * pos.capital
    side_key = pos.side  # "long" or "short"

    state[f"capital_{side_key}"] += pos.capital + pnl_usdt
    state[f"{side_key}_trades"]  += 1
    state[f"{side_key}_pnl"]     += pnl_usdt
    state["total_trades"] += 1
    state["total_pnl"]    += pnl_usdt
    if pnl > 0:
        state["win_trades"]         += 1
        state[f"{side_key}_wins"]   += 1

    icon = "📤" if pos.side == "long" else "📥"
    label = "多單" if pos.side == "long" else "空單"
    print(f"\n  {icon} {label}出場！原因：{exit_reason}")
    print(f"     進場 {pos.entry_price:,.0f} → 出場 {price:,.0f}  "
          f"損益 {pnl*100:+.2f}% ({pnl_usdt:+.2f} USDT)")

    log_trade(pos.side, "EXIT", price, pnl, exit_reason, {})

# ── 主循環 ───────────────────────────────────────────────────────────────────

def run():
    print("🚀 ETH 多空雙向模擬交易機器人啟動")
    print(f"   做多資金：{CONFIG['initial_capital_long']} USDT  |  做空資金：{CONFIG['initial_capital_short']} USDT")
    print(f"   停損 {CONFIG['stop_loss_pct']*100:.0f}%  |  滾動停利 {CONFIG['trail_start_pct']*100:.0f}% 啟動，回落 {CONFIG['trail_drop_pct']*100:.1f}% 出場\n")

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

            price  = df_30m["close"].iloc[-1]
            df_30m = add_indicators(df_30m)
            trend  = check_trend(df_4h, df_1d)

            print_status(state, price, trend, pos_long, pos_short)

            # ══ 多單管理 ══════════════════════════════════════════════════════
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
                # 趨勢為多頭或中性時才找多單進場
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
                        print(f"\n  📈 多單進場！價格 {price:,.0f}  |  訊號：{', '.join(triggered)}")
                else:
                    print(f"  空頭趨勢中，多單暫停")

            # ══ 空單管理 ══════════════════════════════════════════════════════
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
                # 趨勢為空頭或中性時才找空單進場
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
                        print(f"\n  📉 空單進場！價格 {price:,.0f}  |  訊號：{', '.join(triggered)}")
                else:
                    print(f"  多頭趨勢中，空單暫停")

            # 同步儲存
            if pos_long:
                state["position_long"]  = pos_long.to_dict()
            if pos_short:
                state["position_short"] = pos_short.to_dict()
            save_state(state)

        except KeyboardInterrupt:
            print("\n\n⏹ 手動停止機器人")
            break
        except Exception as e:
            print(f"[ERROR] 發生錯誤：{e}")

        time.sleep(CONFIG["check_interval_sec"])


if __name__ == "__main__":
    run()
