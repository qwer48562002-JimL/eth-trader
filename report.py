"""
績效報告產生器（支援多空分開統計）
執行方式：python report.py
"""

import json, os
from datetime import datetime

LOG_FILE   = "trade_log.json"
STATE_FILE = "state.json"
INITIAL_CAPITAL = 2000  # 多空合計

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def calc_stats(trades):
    if not trades:
        return None
    total  = len(trades)
    wins   = sum(1 for t in trades if t["pnl_pct"] > 0)
    losses = total - wins
    pnls   = [t["pnl_pct"] for t in trades]
    avg_win  = sum(p for p in pnls if p > 0) / wins   if wins   else 0
    avg_loss = sum(p for p in pnls if p <= 0) / losses if losses else 0
    return dict(total=total, wins=wins, losses=losses,
                win_rate=wins/total*100,
                avg_win=avg_win, avg_loss=avg_loss,
                ratio=abs(avg_win/avg_loss) if avg_loss else float("inf"))

def pair_trades(logs, side):
    trades, entry = [], None
    for log in logs:
        if log.get("side") != side:
            continue
        if log["action"] == "ENTRY":
            entry = log
        elif log["action"] == "EXIT" and entry:
            trades.append({
                "entry_time":  entry["time"],
                "exit_time":   log["time"],
                "entry_price": entry["price"],
                "exit_price":  log["price"],
                "pnl_pct":     log["pnl_pct"],
                "reason":      log["reason"],
            })
            entry = None
    return trades

def print_section(label, trades, total_pnl, initial):
    s = calc_stats(trades)
    roi = total_pnl / (initial / 2) * 100
    print(f"\n  ── {label} ──")
    if not s:
        print("    尚無完整交易紀錄")
        return
    print(f"    交易次數  : {s['total']}  |  勝率 {s['win_rate']:.1f}%  ({s['wins']}勝 {s['losses']}敗)")
    print(f"    累計損益  : {total_pnl:+.2f} USDT  ({roi:+.2f}%)")
    print(f"    平均獲利  : +{s['avg_win']:.2f}%  |  平均虧損 {s['avg_loss']:.2f}%  |  盈虧比 {s['ratio']:.2f}")

    reasons = {}
    for t in trades:
        r = t["reason"].split("（")[0]
        reasons[r] = reasons.get(r, 0) + 1
    print(f"    出場原因  : " + "  ".join(f"{r}×{c}" for r, c in sorted(reasons.items(), key=lambda x: -x[1])))

    print(f"\n    最近 5 筆：")
    for t in trades[-5:]:
        et = t["entry_time"][:16].replace("T"," ")
        arrow = "📈" if t["pnl_pct"] > 0 else "📉"
        print(f"    {arrow} {et}  進 {t['entry_price']:>9,.0f} → 出 {t['exit_price']:>9,.0f}  {t['pnl_pct']:>+7.2f}%  {t['reason']}")

def generate_report():
    logs  = load_json(LOG_FILE)  or []
    state = load_json(STATE_FILE) or {}

    long_trades  = pair_trades(logs, "long")
    short_trades = pair_trades(logs, "short")

    print("\n" + "="*60)
    print("  📊 ETH 多空模擬交易績效報告")
    print("="*60)
    print(f"  初始資金    : {INITIAL_CAPITAL:,.2f} USDT（多空各 {INITIAL_CAPITAL//2}）")
    print(f"  目前多單資金: {state.get('capital_long', '-'):,.2f} USDT")
    print(f"  目前空單資金: {state.get('capital_short', '-'):,.2f} USDT")
    total_pnl = state.get("total_pnl", 0)
    roi = total_pnl / INITIAL_CAPITAL * 100
    print(f"  總損益      : {total_pnl:+.2f} USDT  ({roi:+.2f}%)")

    print_section("做多績效", long_trades,  state.get("long_pnl", 0),  INITIAL_CAPITAL)
    print_section("做空績效", short_trades, state.get("short_pnl", 0), INITIAL_CAPITAL)
    print("\n" + "="*60)

if __name__ == "__main__":
    generate_report()
