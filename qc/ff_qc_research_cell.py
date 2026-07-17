# ============================================================================
# FF ETF Calendar — all-in-one QC RESEARCH NOTEBOOK cell (2026-07-08)
# Paste into a Research notebook in the SAME QC organization that ran
# ff_etf_calendar_qc.py. Reads the trade CSV from the Object Store and runs
# the pre-registered post-analysis (k-spectrum, k*, per-ticker, kill check).
# ============================================================================
import io
import pandas as pd

qb = QuantBook()
KEY = "ff_etf_calendar_trades.csv"
if not qb.ObjectStore.ContainsKey(KEY):
    raise RuntimeError(f"'{KEY}' not in Object Store — has the backtest finished?")
df = pd.read_csv(io.StringIO(qb.ObjectStore.Read(KEY)))
print(f"raw rows: {len(df)}")

df = df[df.exit_reason != "end_of_data"].copy()
num_cols = ["e_f_bid", "e_f_ask", "e_b_bid", "e_b_ask",
            "x_f_bid", "x_f_ask", "x_b_bid", "x_b_ask", "mid_debit", "ff"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

COMM = 2.60 / 100.0  # $0.65/contract/side x 2 legs x 2 sides, per-share terms
we = (df.e_f_ask - df.e_f_bid) + (df.e_b_ask - df.e_b_bid)          # entry leg-sum width
wx = ((df.x_f_ask - df.x_f_bid) + (df.x_b_ask - df.x_b_bid)).fillna(we)  # exit (fallback entry)
exit_mid = 0.5 * (df.x_b_bid + df.x_b_ask) - 0.5 * (df.x_f_bid + df.x_f_ask)

ok = exit_mid.notna() & (df.mid_debit > 0)
df, we, wx, exit_mid = df[ok], we[ok], wx[ok], exit_mid[ok]
print(f"analyzed: {len(df)} trades (dropped {int((~ok).sum())} without exit quotes)")
print(f"exit reasons: {df.exit_reason.value_counts().to_dict()}")
print(f"median entry leg-sum width: ${we.median():.3f} on median debit ${df.mid_debit.median():.2f} "
      f"({(we / df.mid_debit).median():.0%} of debit)")

def ev(k):
    """EV% of debit at crossing fraction k (cost = k x full leg width, per leg, per side)."""
    pnl = exit_mid - df.mid_debit - k * (we + wx) - COMM
    return pnl / df.mid_debit * 100

print(f"\n{'k':>6} {'meanEV%':>8} {'medEV%':>7} {'win':>5}   (k=0 mid | 0.5 = market order)")
for k in (0.0, 0.25, 0.5, 1.0):
    r = ev(k)
    print(f"{k:6.2f} {r.mean():+8.1f} {r.median():+7.1f} {(r > 0).mean():5.0%}")

lo, hi = 0.0, 2.0
for _ in range(40):
    m = 0.5 * (lo + hi)
    lo, hi = (m, hi) if ev(m).mean() > 0 else (lo, m)
kstar = 0.5 * (lo + hi)
print(f"\nk* (breakeven crossing fraction, mean EV): {kstar:.3f}")

print("\nper ticker (k=0.25):")
print(df.assign(r=ev(0.25)).groupby("ticker")
        .agg(n=("r", "size"), meanEV=("r", "mean"), medEV=("r", "median"),
             win=("r", lambda x: (x > 0).mean()))
        .round(1).sort_values("n", ascending=False).to_string())

print("\nby year (k=0.25):")
print(df.assign(r=ev(0.25), y=df.entry_date.str[:4]).groupby("y")
        .agg(n=("r", "size"), meanEV=("r", "mean"), win=("r", lambda x: (x > 0).mean()))
        .round(1).to_string())

r25 = ev(0.25).mean()
print("\n" + "=" * 60)
print(f"PRE-REGISTERED KILL CHECK: EV(k=0.25) = {r25:+.1f}%")
print("VERDICT:", "ALIVE — awaiting live BAG combo k from R0 snapshots "
      f"(keep needs k_measured <= {0.8 * kstar:.2f} and EV(k_measured) >= +10%)"
      if r25 > 0 else "KILL — no combo market beats quarter leg-sum systematically")
print("=" * 60)
