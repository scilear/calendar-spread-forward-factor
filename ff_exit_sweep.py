#!/usr/bin/env python3
"""
ff_exit_sweep.py — exit strategy optimization on ff_all_trades.csv + ff_daily_paths.csv.

Simulates multiple exit rules on recorded daily paths and compares vs hold-to-chain-roll:
  - Baseline : hold until chain rolls (last observable quote)
  - TP exits : take profit at X% of max_profit achieved_frac (40/50/60/70/80%)
  - FF drop  : exit when current_ff drops below absolute threshold (0, 8, 12%)
  - FF rel   : exit when current_ff < entry_ff × ratio (0.5, 0.75, 1.0)
  - TimeCut  : exit after N path points (1, 2, 3)

All returns = (cal_val - entry_debit - IB_COMM) / entry_debit × 100  (like hold_mid_ret).
Trades with 0 path points are EXCLUDED from strategy-dependent analyses (no intra-trade data).
They ARE included in baseline (their hold_mid_ret is None = not counted).

Breakdowns:
  - All trades
  - FF entry tier: >0%, >8%, >12%, >16%, >20%
  - Year
  - Entry FF quintile: p0–p20 / p20–p40 / p40–p60 / p60–p80 / p80–p100

Usage:
  python ff_exit_sweep.py
  python ff_exit_sweep.py --ff-min 16    # filter to FF>16% entries only
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

IB_COMM = 0.052
OUT_DIR  = Path(__file__).parent

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(ff_min: float = None):
    trades_path = OUT_DIR / "ff_all_trades.csv"
    paths_path  = OUT_DIR / "ff_daily_paths.csv"

    trades = pd.read_csv(trades_path, parse_dates=["entry_date", "exit_date"])
    paths  = pd.read_csv(paths_path,  parse_dates=["obs_date"])

    print(f"Loaded {len(trades):,} trades, {len(paths):,} path rows")

    if ff_min is not None:
        trades = trades[trades["entry_ff"] > ff_min].copy()
        paths  = paths[paths["trade_id"].isin(trades["trade_id"])].copy()
        print(f"After FF>{ff_min}% filter: {len(trades):,} trades")

    # Attach entry metadata to paths for easy filtering
    meta = trades[["trade_id", "entry_ff", "entry_debit", "max_profit",
                   "entry_date", "ticker"]].copy()
    paths = paths.merge(meta, on="trade_id", how="left")

    return trades, paths


# ── Return calculation ─────────────────────────────────────────────────────────

def path_ret(cal_val, entry_debit):
    return (cal_val - entry_debit - IB_COMM) / entry_debit * 100


# ── Exit strategy simulation ───────────────────────────────────────────────────

def simulate_exits(trades: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade with path data, compute the realized return under each exit rule.
    Returns a DataFrame with one row per trade × strategy.
    """
    # Group paths by trade_id, sorted by obs_date
    trade_paths = {}
    for tid, grp in paths.sort_values("obs_date").groupby("trade_id"):
        trade_paths[tid] = grp.reset_index(drop=True)

    records = []

    for _, tr in trades.iterrows():
        tid   = tr["trade_id"]
        ed    = tr["entry_debit"]
        mp    = tr["max_profit"]
        eff   = tr["entry_ff"] if not math.isnan(tr["entry_ff"]) else None
        year  = str(tr["entry_date"].year)

        # Baseline: hold to chain roll
        baseline_ret = tr["hold_mid_ret"]  # may be None

        base_rec = {
            "trade_id":   tid,
            "year":       year,
            "entry_ff":   eff,
            "entry_debit": ed,
            "strategy":   "baseline",
            "realized_ret": baseline_ret,
            "has_path":   tid in trade_paths,
        }
        records.append(base_rec)

        if tid not in trade_paths:
            continue

        pts = trade_paths[tid]
        n   = len(pts)

        # ─ TP exits ───────────────────────────────────────────────────────────
        for tp_frac in [0.40, 0.50, 0.60, 0.70, 0.80]:
            ret_out = baseline_ret  # default: hold if TP never hit
            if mp is not None and mp > 0 and not math.isnan(mp):
                for _, p in pts.iterrows():
                    af = p["achieved_frac"]
                    if af is not None and not math.isnan(af) and af >= tp_frac:
                        ret_out = path_ret(p["cal_val"], ed)
                        break
            records.append({
                "trade_id": tid, "year": year, "entry_ff": eff, "entry_debit": ed,
                "strategy": f"TP{int(tp_frac*100)}",
                "realized_ret": ret_out, "has_path": True,
            })

        # ─ FF absolute drop exits ─────────────────────────────────────────────
        for ff_thr in [0.0, 8.0, 12.0]:
            ret_out = baseline_ret
            for _, p in pts.iterrows():
                cff = p["current_ff"]
                if cff is not None and not math.isnan(cff) and cff < ff_thr:
                    ret_out = path_ret(p["cal_val"], ed)
                    break
            records.append({
                "trade_id": tid, "year": year, "entry_ff": eff, "entry_debit": ed,
                "strategy": f"FF_drop_{ff_thr:.0f}",
                "realized_ret": ret_out, "has_path": True,
            })

        # ─ FF relative drop exits ─────────────────────────────────────────────
        for ratio in [0.5, 0.75, 1.0]:
            ret_out = baseline_ret
            if eff is not None and not math.isnan(eff) and eff > 0:
                thr = eff * ratio
                for _, p in pts.iterrows():
                    cff = p["current_ff"]
                    if cff is not None and not math.isnan(cff) and cff < thr:
                        ret_out = path_ret(p["cal_val"], ed)
                        break
            records.append({
                "trade_id": tid, "year": year, "entry_ff": eff, "entry_debit": ed,
                "strategy": f"FF_rel_{ratio:.2f}x",
                "realized_ret": ret_out, "has_path": True,
            })

        # ─ Time-based cuts ────────────────────────────────────────────────────
        for npt in [1, 2, 3]:
            if n >= npt:
                ret_out = path_ret(pts.iloc[npt-1]["cal_val"], ed)
            else:
                ret_out = baseline_ret
            records.append({
                "trade_id": tid, "year": year, "entry_ff": eff, "entry_debit": ed,
                "strategy": f"TimeCut_{npt}",
                "realized_ret": ret_out, "has_path": True,
            })

    return pd.DataFrame(records)


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics(rets: pd.Series) -> dict:
    r = rets.dropna()
    if len(r) == 0:
        return {"n": 0, "mean": None, "median": None, "win_rate": None,
                "p25": None, "p75": None}
    return {
        "n":        len(r),
        "mean":     round(r.mean(), 2),
        "median":   round(r.median(), 2),
        "win_rate": round((r > 0).mean() * 100, 1),
        "p25":      round(r.quantile(0.25), 2),
        "p75":      round(r.quantile(0.75), 2),
    }


def summarise(sim: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    rows = []
    strategies = sim["strategy"].unique()
    groups = [None] if group_col is None else sim[group_col].dropna().unique()

    for g in sorted(groups, key=lambda x: (x is None, x)):
        sub = sim if g is None else sim[sim[group_col] == g]
        for strat in strategies:
            s = sub[sub["strategy"] == strat]
            m = metrics(s["realized_ret"])
            row = {"group": g or "ALL", "strategy": strat, **m}
            rows.append(row)
    return pd.DataFrame(rows)


# ── Print tables ───────────────────────────────────────────────────────────────

STRATEGY_ORDER = [
    "baseline",
    "TP40", "TP50", "TP60", "TP70", "TP80",
    "FF_drop_0", "FF_drop_8", "FF_drop_12",
    "FF_rel_0.50x", "FF_rel_0.75x", "FF_rel_1.00x",
    "TimeCut_1", "TimeCut_2", "TimeCut_3",
]


def print_table(df: pd.DataFrame, title: str):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")
    print(f"{'Strategy':<18}{'n':>7}{'mean%':>8}{'med%':>8}{'win%':>8}"
          f"{'p25%':>8}{'p75%':>8}")
    print(f"{'─'*70}")

    # sort by STRATEGY_ORDER
    strat_idx = {s: i for i, s in enumerate(STRATEGY_ORDER)}
    df_sorted = df.sort_values("strategy", key=lambda c: c.map(
        lambda x: strat_idx.get(x, 99)))

    for _, row in df_sorted.iterrows():
        if row["n"] == 0:
            continue
        flag = " ◄" if row["strategy"] != "baseline" and row["mean"] is not None \
               and row.get("_base_mean") is not None \
               and row["mean"] > row["_base_mean"] + 0.5 else ""
        print(f"{row['strategy']:<18}{row['n']:>7}{row['mean']:>8.1f}"
              f"{row['median']:>8.1f}{row['win_rate']:>8.1f}"
              f"{row['p25']:>8.1f}{row['p75']:>8.1f}{flag}")


def print_group_table(df: pd.DataFrame, title: str, group_col: str = "group"):
    groups = df[group_col].unique()
    strategies = [s for s in STRATEGY_ORDER if s in df["strategy"].values]

    print(f"\n{'═'*110}")
    print(f"  {title}")
    print(f"{'═'*110}")

    # Header
    hdr = f"{'Strategy':<18}"
    for g in sorted(groups, key=str):
        hdr += f" {'n':>5}{'mean%':>7}"
    print(hdr)
    print("─"*110)

    for strat in strategies:
        sub = df[df["strategy"] == strat]
        line = f"{strat:<18}"
        for g in sorted(groups, key=str):
            r = sub[sub[group_col] == g]
            if len(r) == 0 or r.iloc[0]["n"] == 0:
                line += f" {'':>5}{'':>7}"
            else:
                line += f" {r.iloc[0]['n']:>5}{r.iloc[0]['mean']:>7.1f}"
        print(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ff-min", type=float, default=None,
                        help="Minimum entry FF% filter (e.g. 16)")
    parser.add_argument("--save", action="store_true",
                        help="Save simulation results to ff_exit_sim.csv")
    args = parser.parse_args()

    trades, paths = load_data(args.ff_min)

    print("\nRunning exit strategy simulation ...")
    sim = simulate_exits(trades, paths)

    # ── 1. Overall summary ────────────────────────────────────────────────────
    overall = summarise(sim[sim["strategy"].isin(STRATEGY_ORDER)])
    print_table(overall, "ALL TRADES — exit strategy comparison (hold_mid_ret basis)")

    # ── 2. By FF entry tier ────────────────────────────────────────────────────
    def ff_tier(ff):
        if ff is None or math.isnan(ff):
            return "FF=None"
        if ff > 20: return "FF>20%"
        if ff > 16: return "FF>16%"
        if ff > 12: return "FF>12%"
        if ff >  8: return "FF>8%"
        if ff >  0: return "FF>0%"
        return "FF≤0%"

    sim["ff_tier"] = sim["entry_ff"].apply(ff_tier)

    for tier in ["FF>20%", "FF>16%", "FF>12%", "FF>8%", "FF>0%", "FF≤0%"]:
        sub = sim[sim["ff_tier"] == tier]
        if len(sub) == 0:
            continue
        s = summarise(sub[sub["strategy"].isin(STRATEGY_ORDER)])
        print_table(s, f"FF TIER: {tier}  ({len(sub[sub['strategy']=='baseline']):,} trades)")

    # ── 3. By year ────────────────────────────────────────────────────────────
    yr_sim = summarise(sim[sim["strategy"].isin(STRATEGY_ORDER)], group_col="year")
    print_group_table(yr_sim, "BY YEAR — mean return % (each column = one year)", "group")

    # ── 4. Focus table: baseline vs best TP vs best FF-drop, by FF tier ───────
    focus_strats = ["baseline", "TP50", "TP60", "TP70", "FF_drop_0", "FF_drop_8"]
    print(f"\n{'═'*100}")
    print("  FOCUS: baseline vs selected exits — mean return% by FF tier")
    print(f"{'═'*100}")
    header = f"{'FF tier':<12}"
    for s in focus_strats:
        header += f"  {s:>12}"
    print(header)
    print("─"*100)

    tier_order = ["FF>20%", "FF>16%", "FF>12%", "FF>8%", "FF>0%", "FF≤0%", "ALL"]
    for tier in tier_order:
        if tier == "ALL":
            sub = sim.copy()
        else:
            sub = sim[sim["ff_tier"] == tier]
        if len(sub) == 0:
            continue
        n = len(sub[sub["strategy"] == "baseline"])
        line = f"{tier:<12}"
        for s in focus_strats:
            s_rows = sub[sub["strategy"] == s]
            if len(s_rows) == 0:
                line += f"  {'N/A':>12}"
            else:
                m = s_rows["realized_ret"].dropna()
                line += f"  {m.mean():>11.1f}%" if len(m) > 0 else f"  {'—':>12}"
        print(line + f"  (n={n})")

    # ── 5. Histogram of baseline returns by FF tier ────────────────────────────
    print(f"\n{'═'*70}")
    print("  BASELINE RETURN DISTRIBUTION by FF tier")
    print(f"{'═'*70}")
    bins = [-200, -100, -50, -20, 0, 20, 50, 100, 200, 500]
    bin_labels = [f"<{bins[i+1]}" for i in range(len(bins)-1)]

    hdr = f"{'FF tier':<12}"
    for bl in bin_labels:
        hdr += f" {bl:>8}"
    hdr += f" {'mean%':>7}"
    print(hdr)
    print("─"*70)

    base_sim = sim[sim["strategy"] == "baseline"]
    for tier in ["FF>20%", "FF>16%", "FF>12%", "FF>8%", "FF>0%", "FF≤0%", "ALL"]:
        if tier == "ALL":
            sub = base_sim.copy()
        else:
            sub = base_sim[base_sim["ff_tier"] == tier]
        r = sub["realized_ret"].dropna()
        if len(r) == 0:
            continue
        counts, _ = np.histogram(r, bins=bins)
        pcts = counts / len(r) * 100
        line = f"{tier:<12}"
        for p in pcts:
            line += f" {p:>7.0f}%"
        line += f" {r.mean():>6.1f}%"
        print(line + f"  (n={len(r)})")

    # ── 6. Path coverage ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PATH COVERAGE (trades with ≥1 path point vs total)")
    print(f"{'═'*70}")
    base = sim[sim["strategy"] == "baseline"]
    total  = len(base)
    with_p = base["has_path"].sum()
    print(f"  Total trades:          {total:>7,}")
    print(f"  With ≥1 path point:    {with_p:>7,}  ({with_p/total*100:.1f}%)")
    print(f"  Zero path points:      {total-with_p:>7,}  ({(total-with_p)/total*100:.1f}%)")
    print(f"  Avg path rows / trade: {len(paths)/total:.2f}")

    # ── 7. Save ────────────────────────────────────────────────────────────────
    if args.save:
        out = OUT_DIR / "ff_exit_sim.csv"
        sim.to_csv(out, index=False)
        print(f"\nSaved simulation → {out.name}")


if __name__ == "__main__":
    main()
