"""
Post-analysis for ff_etf_calendar_qc.py (pre-registered 2026-07-08).
Input: ff_etf_calendar_trades.csv downloaded from QC ObjectStore.
Reconstructs per-trade EV across the fill spectrum k in {0, .25, .5, 1.0}
(cost = k * full leg width, per leg, per side) + solves k* (breakeven).
Commission $0.65/contract/side x 2 legs x 2 sides = $2.60/trade.
Decision rule (frozen): KILL if EV(k=0.25) < 0. KEEP requires live BAG
combo k_measured <= 0.8*k_star AND EV(k_measured) >= +10%.
Usage: python ff_qc_post_analysis.py ff_etf_calendar_trades.csv
"""
import sys
import pandas as pd

COMM = 2.60 / 100.0  # per share terms

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "ff_etf_calendar_trades.csv")
df = df[df.exit_reason != "end_of_data"].copy()
for c in ["e_f_bid", "e_f_ask", "e_b_bid", "e_b_ask",
          "x_f_bid", "x_f_ask", "x_b_bid", "x_b_ask", "mid_debit"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

we = (df.e_f_ask - df.e_f_bid) + (df.e_b_ask - df.e_b_bid)
wx = ((df.x_f_ask - df.x_f_bid) + (df.x_b_ask - df.x_b_bid)).fillna(we)
exit_mid = (0.5 * (df.x_b_bid + df.x_b_ask) - 0.5 * (df.x_f_bid + df.x_f_ask))
ok = exit_mid.notna() & (df.mid_debit > 0)
df, we, wx, exit_mid = df[ok], we[ok], wx[ok], exit_mid[ok]
print(f"trades: {len(df)} (dropped {int((~ok).sum())} without exit quotes)")

def ev(k):
    pnl = exit_mid - df.mid_debit - k * (we + wx) - COMM
    r = pnl / df.mid_debit * 100
    return r

print(f"\n{'k':>6s} {'meanEV%':>8s} {'medEV%':>7s} {'win':>5s}")
for k in (0.0, 0.25, 0.5, 1.0):
    r = ev(k)
    print(f"{k:6.2f} {r.mean():+8.1f} {r.median():+7.1f} {(r > 0).mean():5.0%}")

lo, hi = 0.0, 2.0
for _ in range(40):
    mid = 0.5 * (lo + hi)
    if ev(mid).mean() > 0:
        lo = mid
    else:
        hi = mid
kstar = 0.5 * (lo + hi)
print(f"\nk* (breakeven crossing fraction, mean EV): {kstar:.3f}")
print("per ticker (k=0.25):")
r25 = ev(0.25)
print(df.assign(r=r25).groupby("ticker").agg(
    n=("r", "size"), meanEV=("r", "mean"), win=("r", lambda x: (x > 0).mean())
).round(1).to_string())
r = ev(0.25)
print(f"\nKILL check: EV(k=0.25) = {r.mean():+.1f}% -> "
      f"{'ALIVE, awaiting live k' if r.mean() > 0 else 'KILL'}")
