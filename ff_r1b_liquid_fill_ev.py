"""
R1b — realistic-fill EV on the LIQUID subset (CSFF practical review addendum,
Fabien sign-off 2026-07-07). Spec: liquid-list tickers, FF>20, entries
2024-25, straddle calendar (4 legs). Fill model = cross 50% of quoted spread
per leg: entry 4 legs; exit 2 back legs (expiry exits) or 4 legs
(chain_roll, exit-date quotes, entry-width fallback if missing). Widths from
Dolt option_chain bid/ask (EOD — likely wider than 3:45 PM touch =
conservative). Zero-bid legs kept (width=ask) + fraction reported.
KEEP bar: liquid FF>20 realistic mean EV >= +10%.
"""
import pandas as pd, pymysql, sys
from collections import defaultdict

LIQ = set("""AAPL MSFT NVDA AMZN META GOOGL GOOG TSLA AMD NFLX INTC BA JPM BAC C WFC GS XOM CVX
PFE MRK KO PEP DIS PLTR SOFI F GM T VZ CSCO MU AVGO QCOM ORCL CRM UBER SNAP COIN MARA RIOT
AAL DAL UAL CCL NCLH RCL FCX GME AMC BABA NIO SHOP PYPL SQ ROKU ZM DKNG LUV WMT TGT HD LOW
SBUX MCD NKE SPY DIA XLK XLF XLE XLV XLI XLP XLU XLB XLRE XLC XLY GDX SMH KRE XBI EWZ FXI""".split())

T = pd.read_csv("ff_all_trades.csv")
T = T[T.hold_mid_ret.notna() & T.ticker.isin(LIQ) & (T.entry_ff > 20)
      & (T.entry_date >= "2024-01-01") & (T.entry_date < "2026-01-01")].copy()
print(f"liquid FF>20 2024-25 trades with exits: {len(T)}")

# quote needs: (date, ticker, exp, strike) -> both call/put legs
need = defaultdict(set)   # date -> {(ticker, exp, strike)}
for _, t in T.iterrows():
    k_f, k_b = (t.ticker, t.front_exp, t.strike), (t.ticker, t.back_exp, t.strike)
    need[t.entry_date].update([k_f, k_b])
    if t.exit_reason == "expiry":
        need[t.exit_date].add(k_b)
    elif pd.notna(t.exit_date):
        need[t.exit_date].update([k_f, k_b])

conn = pymysql.connect(host="127.0.0.1", port=3307, user="root", password="", database="options")
cur = conn.cursor()
Q = {}  # (date, ticker, exp, strike, cp) -> (bid, ask)
for i, (d, keys) in enumerate(sorted(need.items())):
    tks = sorted({k[0] for k in keys})
    cur.execute(
        "SELECT act_symbol, expiration, strike, call_put, bid, ask FROM option_chain "
        f"WHERE date=%s AND act_symbol IN ({','.join(['%s']*len(tks))})", [d] + tks)
    want = {(k[0], k[1], float(k[2])) for k in keys}
    for sym, exp, stk, cp, bid, ask in cur.fetchall():
        kk = (sym, str(exp), float(stk))
        if kk in want:
            Q[(d, sym, str(exp), float(stk), cp[0].upper())] = (float(bid or 0), float(ask or 0))
    if i % 100 == 0: print(f"  quotes: {i}/{len(need)} dates", file=sys.stderr)
conn.close()
print(f"quote rows kept: {len(Q)}")

def legs_width(d, tk, exp, stk):
    out, zb = 0.0, 0
    for cp in ("C", "P"):
        q = Q.get((d, tk, str(exp), float(stk), cp))
        if q is None: return None, 0
        b, a = q
        if a <= 0: return None, 0
        w = max(a - b, 0.0)
        out += w; zb += (b <= 0)
    return out, zb

res = []
miss = 0
for _, t in T.iterrows():
    we, zb1 = legs_width(t.entry_date, t.ticker, t.front_exp, t.strike)
    wb, zb2 = legs_width(t.entry_date, t.ticker, t.back_exp, t.strike)
    if we is None or wb is None: miss += 1; continue
    entry_cross = 0.5 * (we + wb)
    if t.exit_reason == "expiry":
        wx, zb3 = legs_width(t.exit_date, t.ticker, t.back_exp, t.strike)
        exit_cross = 0.5 * wx if wx is not None else 0.5 * wb  # fallback: entry back width
    else:
        wxf, z1 = legs_width(t.exit_date, t.ticker, t.front_exp, t.strike) if pd.notna(t.exit_date) else (None, 0)
        wxb, z2 = legs_width(t.exit_date, t.ticker, t.back_exp, t.strike) if pd.notna(t.exit_date) else (None, 0)
        exit_cross = 0.5 * ((wxf if wxf is not None else we) + (wxb if wxb is not None else wb))
        zb3 = z1 + z2
    cost_pct = (entry_cross + exit_cross) / t.entry_debit * 100 if t.entry_debit > 0 else None
    if cost_pct is None: miss += 1; continue
    res.append({"ticker": t.ticker, "year": t.entry_date[:4], "entry_ff": t.entry_ff,
                "mid_ret": t.hold_mid_ret, "real_ret": t.hold_mid_ret - cost_pct,
                "cost_pct": cost_pct, "debit": t.entry_debit,
                "zero_bid": zb1 + zb2 + zb3, "exit_reason": t.exit_reason})
R = pd.DataFrame(res)
R.to_csv("ff_r1b_results.csv", index=False)
print(f"\npriced: {len(R)} | missing quotes: {miss} ({miss/(miss+len(R)):.0%})")
print(f"any zero-bid leg: {(R.zero_bid>0).mean():.0%} | median spread cost {R.cost_pct.median():.1f}% of debit")
print(f"\n{'group':22s} {'n':>5s} {'midEV':>7s} {'realEV':>7s} {'med real':>8s} {'win':>4s}")
def row(tag, s):
    if len(s): print(f"{tag:22s} {len(s):5d} {s.mid_ret.mean():+7.1f} {s.real_ret.mean():+7.1f} {s.real_ret.median():+8.1f} {(s.real_ret>0).mean():4.0%}")
row("ALL liquid FF>20", R)
for y in ("2024", "2025"): row(f"  {y}", R[R.year == y])
row("  FF 20-25", R[R.entry_ff < 25]); row("  FF >25", R[R.entry_ff >= 25])
row("  debit >= $1", R[R.debit >= 1.0]); row("  debit >= $2", R[R.debit >= 2.0])
ev = R.real_ret.mean()
print(f"\nKEEP bar: realistic mean EV {ev:+.1f}% vs +10% -> {'PASS' if ev >= 10 else 'FAIL'}")
