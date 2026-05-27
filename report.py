"""
績效報告產生器（從 PostgreSQL 讀取）
執行方式：python report.py
"""

import os, json
from datetime import datetime

try:
    import psycopg2
    HAS_PG = True
except ImportError:
    HAS_PG = False

INITIAL_CAPITAL = 2000

def get_conn():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url or not HAS_PG:
        return None
    try:
        return psycopg2.connect(db_url, sslmode="require")
    except:
        return None

def load_from_db():
    conn = get_conn()
    if not conn:
        return None, None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM state WHERE key = 'bot_state'")
            row = cur.fetchone()
            state = json.loads(row[0]) if row else {}
            cur.execute("""
                SELECT side, action, price, pnl_pct, reason, time
                FROM trade_log ORDER BY time ASC
            """)
            logs = [{"side": r[0], "action": r[1], "price": r[2],
                     "pnl_pct": r[3], "reason": r[4],
                     "time": r[5].isoformat()} for r in cur.fetchall()]
        return state, logs
    except Exception as e:
        print(f"[DB] 讀取失敗：{e}")
        return None, None
    finally:
        conn.close()

def load_from_file():
    state, logs = {}, []
    if os.path.exists("state.json"):
        with open("state.json") as f:
            state = json.load(f)
    if os.path.exists("trade_log.json"):
        with open("trade_log.json") as f:
            logs = json.load(f)
    return state, logs

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

def print_section(label, trades, total_pnl):
    if not trades:
        print(f"\n  ── {label} ──")
        print(f"    尚無完整交易紀錄")
        return
    total  = len(trades)
    wins   = sum(1 for t in trades if t["pnl_pct"] > 0)
    losses = total - wins
    wr     = wins / total * 100
    pnls   = [t["pnl_pct"] for t in trades]
    avg_win  = sum(p for p in pnls if p > 0) / wins   if wins   else 0
    avg_loss = sum(p for p in pnls if p <= 0) / losses if losses else 0
    ratio    = abs(avg_win / avg_loss) if avg_loss else float("inf")
    roi      = total_pnl / (INITIAL_CAPITAL / 2) * 100

    print(f"\n  ── {label} ──")
    print(f"    交易次數 : {total}  勝率 {wr:.1f}%  ({wins}勝 {losses}敗)")
    print(f"    累計損益 : {total_pnl:+.2f} USDT  ({roi:+.2f}%)")
    print(f"    均獲利   : +{avg_win:.2f}%  均虧損 {avg_loss:.2f}%  盈虧比 {ratio:.2f}")

    reasons = {}
    for t in trades:
        r = t["reason"].split("（")[0]
        reasons[r] = reasons.get(r, 0) + 1
    print(f"    出場原因 : " + "  ".join(f"{r}×{c}" for r, c in sorted(reasons.items(), key=lambda x: -x[1])))

    print(f"\n    最近 5 筆：")
    for t in trades[-5:]:
        et    = t["entry_time"][:16].replace("T", " ")
        arrow = "📈" if t["pnl_pct"] > 0 else "📉"
        print(f"    {arrow} {et}  {t['entry_price']:>9,.0f}→{t['exit_price']:>9,.0f}  {t['pnl_pct']:>+7.2f}%  {t['reason']}")

def generate_report():
    state, logs = load_from_db()
    if state is None:
        print("[INFO] 從本機檔案讀取...")
        state, logs = load_from_file()

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

    print_section("做多績效", long_trades,  state.get("long_pnl",  0))
    print_section("做空績效", short_trades, state.get("short_pnl", 0))
    print("\n" + "="*60)

if __name__ == "__main__":
    generate_report()
