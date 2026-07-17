#!/usr/bin/env python3
"""
ff_exit_research.py — empirically derive exit thresholds from the opex trade dataset.

Loads opex_trades.csv (produced by ff_opex_signal.py) and answers:

  A. TP RESEARCH
     For each trade, compute the theoretical maximum calendar value at expiry
     (spot pins strike, back IV converges to forward vol level):
         max_theoretical = back_straddle_entry × sqrt(t_fwd / t_back)
     Then: what fraction of that theoretical max was actually realised AT EXPIRY?
     Tests TP thresholds [40%, 50%, 60%, 70%, 80%, 90%, 100%]:
       "If we had set TP at X% of max_profit, would we have exited before expiry?"
     NOTE: the expiry value is a LOWER BOUND on the intra-period peak — the calendar
     typically peaks a few days before expiry. So "hit at expiry" understates
     actual hit rate during the holding period. We flag this explicitly.

  B. FF-DROP RESEARCH (cross-sectional only — monthly data limitation)
     With only monthly snapshots, we cannot track the intra-period FF path.
     What we CAN measure: does entry FF level predict how much of the theoretical
     max is captured? Do high-FF trades do better or worse on the TP metric?
     This gives intuition for whether FF level should modulate the TP target.

  C. WHAT IS NOT ANSWERABLE HERE
     Intra-period exit timing (when during the month to exit) requires daily data
     (Dolt daily coverage starts ~2024). That is a separate study.

Outputs: printed tables + opex_exit_research.txt for offline review.
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent / "opex_trades.csv"
IB_COMM = 0.052


def load_trades() -> list[dict]:
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found — run ff_opex_signal.py first.")
        sys.exit(1)
    trades = []
    with open(DATA_PATH) as f:
        for row in csv.DictReader(f):
            t = {}
            for k, v in row.items():
                if v == "" or v == "None":
                    t[k] = None
                else:
                    try:
                        t[k] = float(v)
                    except ValueError:
                        t[k] = v
            trades.append(t)
    return trades


def enrich(trades: list[dict]) -> list[dict]:
    """Add computed fields: max_theoretical, max_profit, achieved_frac."""
    out = []
    for t in trades:
        bm = t.get("back_straddle")
        em = t.get("entry_mid")
        xm = t.get("exit_mid")
        tf = t.get("t_fwd")
        tb = t.get("t_back")
        if None in (bm, em, xm, tf, tb) or tb <= 0 or tf <= 0:
            continue
        max_theo = bm * math.sqrt(tf / tb)
        max_profit = max_theo - em
        if max_profit <= 0:
            continue
        # fraction of theoretical max profit realised at expiry
        actual_profit = xm - em
        achieved_frac = actual_profit / max_profit
        t = dict(t)
        t["max_theo"] = max_theo
        t["max_profit"] = max_profit
        t["achieved_frac"] = achieved_frac
        t["actual_profit"] = actual_profit
        out.append(t)
    return out


def section(title: str, width: int = 70):
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")


def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} raw trades from {DATA_PATH.name}")
    trades = enrich(trades)
    print(f"Enriched: {len(trades)} trades with valid TP components\n")

    ff_trades = [t for t in trades if t.get("ff_pct") is not None]
    print(f"Trades with FF signal: {len(ff_trades)}")

    # ── A. THEORETICAL MAX ANALYSIS ──────────────────────────────────────────
    section("A. THEORETICAL MAX CALENDAR VALUE vs ACTUAL EXIT (mid, at expiry)")
    print("What fraction of max_theoretical was realised at expiry?")
    print("  max_theoretical = back_straddle_entry × √(t_fwd / t_back)")
    print("  achieved_frac   = (exit_mid − entry_debit) / (max_theo − entry_debit)")
    print("  NOTE: expiry value is a LOWER BOUND on the intra-period peak.\n")

    af = np.array([t["achieved_frac"] for t in trades])
    pcts = [10, 25, 50, 75, 90, 95]
    print(f"  n = {len(af)}")
    print(f"  mean  = {af.mean():+.3f}  ({af.mean()*100:+.1f}% of theoretical max profit)")
    print(f"  median= {np.median(af):+.3f}  ({np.median(af)*100:+.1f}%)")
    print(f"  stdev = {af.std():.3f}")
    print(f"\n  Percentile distribution of achieved_frac:")
    for p in pcts:
        v = np.percentile(af, p)
        print(f"    p{p:>3}: {v:+.3f}  ({v*100:+.1f}%)")
    print(f"\n  Fraction of trades that ended ABOVE theoretical max (achieved > 1.0): "
          f"{(af > 1.0).mean()*100:.1f}%")
    print(f"  Fraction that ended negative (achieved < 0): "
          f"{(af < 0).mean()*100:.1f}%")

    # ── B. TP THRESHOLD STUDY ─────────────────────────────────────────────────
    section("B. TP THRESHOLD STUDY")
    print("If TP set at X% of max_profit, what fraction of trades hits it AT EXPIRY?")
    print("(Actual hit rate during holding period is higher — this is a lower bound.)\n")
    print(f"{'TP%':>6}{'hit_rate':>10}{'n_hit':>8}{'med_ret_hit%':>15}{'med_ret_miss%':>16}")

    tp_targets = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    for tp in tp_targets:
        hit = [t for t in trades if t["achieved_frac"] >= tp]
        miss = [t for t in trades if t["achieved_frac"] < tp]
        hit_rets = np.array([t["mid_ret"] for t in hit]) if hit else np.array([])
        miss_rets = np.array([t["mid_ret"] for t in miss]) if miss else np.array([])
        hit_rate = len(hit) / len(trades) * 100
        med_hit  = np.median(hit_rets) if len(hit_rets) else float("nan")
        med_miss = np.median(miss_rets) if len(miss_rets) else float("nan")
        print(f"{tp*100:>5.0f}%{hit_rate:>9.1f}%{len(hit):>8}{med_hit:>+14.1f}%{med_miss:>+15.1f}%")

    print("\n  Interpretation: 'hit_rate' = fraction of trades where the calendar was")
    print("  worth ≥ entry_debit + TP% × max_profit at EXPIRY (lower bound on when")
    print("  it would have been reachable during the holding period).")

    # ── C. TP HIT RATE BY FF QUINTILE ────────────────────────────────────────
    section("C. TP HIT RATE BY FF QUINTILE (does FF predict TP capture?)")
    print("Testing TP=80% of max_profit. Does entry FF predict whether TP is reached?\n")

    if len(ff_trades) >= 10:
        ff_vals = np.array([t["ff_pct"] for t in ff_trades])
        q_edges = np.percentile(ff_vals, [0, 20, 40, 60, 80, 100])
        print(f"{'FF range':>18}{'n':>6}{'hit_rate_80%':>14}{'med_achieved%':>15}{'med_ret%':>10}")
        for qi in range(5):
            lo, hi = q_edges[qi], q_edges[qi+1]
            bucket = [t for t in ff_trades if lo <= t["ff_pct"] <= hi]
            if not bucket:
                continue
            hit80 = [t for t in bucket if t["achieved_frac"] >= 0.80]
            af_b = np.array([t["achieved_frac"] for t in bucket])
            rets_b = np.array([t["mid_ret"] for t in bucket])
            print(f"{lo:>7.1f}–{hi:<7.1f}%{len(bucket):>6}"
                  f"{len(hit80)/len(bucket)*100:>13.1f}%"
                  f"{np.median(af_b)*100:>+14.1f}%"
                  f"{np.median(rets_b):>+9.1f}%")
    else:
        print("  Too few FF records for quintile split.")

    # ── D. ACHIEVED FRACTION BY YEAR ─────────────────────────────────────────
    section("D. ACHIEVED FRACTION BY YEAR (regime dependence)")
    print(f"{'year':>6}{'n':>6}{'med_achieved%':>15}{'p25_achieved%':>15}{'p75_achieved%':>15}{'hit80%':>10}")
    years = sorted({int(t["year"]) for t in trades})
    for y in years:
        yr = [t for t in trades if int(t["year"]) == y]
        if not yr:
            continue
        af_y = np.array([t["achieved_frac"] for t in yr])
        hit80 = sum(1 for t in yr if t["achieved_frac"] >= 0.80)
        print(f"{y:>6}{len(yr):>6}"
              f"{np.median(af_y)*100:>+14.1f}%"
              f"{np.percentile(af_y,25)*100:>+14.1f}%"
              f"{np.percentile(af_y,75)*100:>+14.1f}%"
              f"{hit80/len(yr)*100:>9.1f}%")

    # ── E. RELATIONSHIP: FF vs ACHIEVED_FRAC ─────────────────────────────────
    section("E. CORRELATION: entry FF vs achieved_frac")
    if len(ff_trades) >= 10:
        ff_arr = np.array([t["ff_pct"] for t in ff_trades])
        af_arr = np.array([t["achieved_frac"] for t in ff_trades])
        corr = np.corrcoef(ff_arr, af_arr)[0, 1]
        print(f"  Pearson r(FF, achieved_frac) = {corr:+.3f}")
        print(f"  n = {len(ff_trades)}")
        if abs(corr) < 0.1:
            print("  → Weak/no linear relationship: FF level does not predict TP capture.")
        elif corr > 0:
            print("  → Positive: higher FF entry → more of theoretical max captured.")
        else:
            print("  → Negative: higher FF entry → less of theoretical max captured.")

    # ── F. WHAT THIS CANNOT ANSWER ────────────────────────────────────────────
    section("F. LIMITATIONS — what requires daily data (post-2024)")
    print("""
  1. INTRA-PERIOD FF PATH: We have no FF snapshots between entry and expiry.
     Cannot answer: "at what FF level should we exit early?"
     Requires: daily Dolt data (available 2024+). Study: for each live trade,
     track daily FF, find when FF first drops to threshold X, record calendar
     value at that moment vs at expiry.

  2. INTRA-PERIOD CALENDAR VALUE PATH: The calendar peaks BEFORE expiry
     (short gamma front accelerating into expiry). The TP hit rates above
     (section B) are lower bounds. True TP hit rates require daily pricing.
     Requires: same daily dataset.

  3. FF-DROP EXIT THRESHOLDS: Cannot be derived here. Need daily study to
     find: when FF drops by X points / to Y%, does the calendar's value
     at that moment exceed what it would be at expiry?

  ACTION: once the corrected-formula daily backtest is built (post-2024),
  run ff_exit_daily_research.py on that data to derive the FF-drop and
  intra-period TP thresholds empirically.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
