#!/usr/bin/env python3
"""
ff_exit_daily_research.py — empirically derive intra-period exit thresholds.

Uses Mon/Wed/Fri daily Dolt coverage (2024-2025) to track calendar value and FF
path for every opex trade in opex_trades.csv (2024-2025 subset).

For each trade, simulates:
  TP exits   : exit first day cal_value >= entry_debit + X% * max_profit
               X in [40, 50, 60, 70, 80, 90, 100]
  FF absolute: exit first day current_FF < threshold
               threshold in [0, 3, 5, 8, 12]%
  FF relative: exit first day current_FF < entry_FF * ratio
               ratio in [0.25, 0.40, 0.50, 0.60]

Baseline: hold to front_exp (exit_mid from opex_trades.csv).

Compares P&L of each early-exit strategy vs baseline.
Outputs a per-trade CSV and printed summary tables.
"""
import csv
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ff_backtest import _fetch_option_prices, _get_straddle
from ff_universe_scan import conn, forward_vol, compute_ff
from ff_opex_signal import atm_iv_for_expiry, atm_strike_from_chain, spot_from_chain

DATA_DIR  = Path(__file__).parent
TRADES_CSV = DATA_DIR / "opex_trades.csv"
OUT_CSV    = DATA_DIR / "exit_daily_results.csv"

IB_COMM = 0.052

TEST_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","BAC","XOM",
    "WMT","HD","DIS","NFLX","INTC","AMD","CSCO","PFE","KO","MCD",
    "SPY","QQQ","IWM","XLF","XLE",
]

TP_THRESHOLDS    = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
FF_ABS_EXITS     = [0.0, 3.0, 5.0, 8.0, 12.0]
FF_REL_RATIOS    = [0.25, 0.40, 0.50, 0.60]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(v):
    if v is None or v in ("", "None"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def load_trades_24_25() -> list[dict]:
    trades = []
    with open(TRADES_CSV) as f:
        for row in csv.DictReader(f):
            t = {k: _parse(v) for k, v in row.items()}
            y = int(t["year"]) if t["year"] is not None else 0
            if y not in (2024, 2025):
                continue
            trades.append(t)
    return trades


def get_trading_dates_in_range(start: date, end: date, conn_) -> list[date]:
    """Return distinct dates in Dolt between start (exclusive) and end (inclusive)."""
    with conn_.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT date FROM option_chain "
            "WHERE act_symbol='SPY' AND date > %s AND date <= %s ORDER BY date",
            (start.isoformat(), end.isoformat()),
        )
        return [r[0] for r in cur.fetchall()]


def current_ff(price_data, ticker, front_exp, back_exp, strike, check_date: date):
    """FF computed on the current (check_date) observation, using the trade's expirations."""
    front_iv = atm_iv_for_expiry(price_data, ticker, front_exp, strike)
    back_iv  = atm_iv_for_expiry(price_data, ticker, back_exp,  strike)
    if front_iv is None or back_iv is None:
        return None
    f_dte = (front_exp - check_date).days
    b_dte = (back_exp  - check_date).days
    if f_dte <= 0 or b_dte <= f_dte:
        return None
    fwd = forward_vol(front_iv, f_dte / 365, back_iv, b_dte / 365)
    return compute_ff(front_iv, fwd)


def calendar_mid(price_data, ticker, front_exp, back_exp, strike, check_date: date):
    """
    Calendar mid value on check_date.
    On front_exp: back_mid - |spot - strike| (front expires, valued at intrinsic).
    Before front_exp: back_mid - front_mid (both quoted).
    """
    bm, _ = _get_straddle(price_data, ticker, back_exp, strike, "mid")
    if bm is None:
        return None
    if check_date == front_exp:
        # Use intrinsic for front leg
        s = spot_from_chain(price_data, ticker, back_exp, strike)
        if s is None:
            return None
        return bm - abs(s - strike)
    else:
        fm, _ = _get_straddle(price_data, ticker, front_exp, strike, "mid")
        if fm is None:
            return None
        return bm - fm


def section(title: str, width: int = 72):
    print(f"\n{'─'*width}\n  {title}\n{'─'*width}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    trades = load_trades_24_25()
    if not trades:
        print("No 2024-2025 trades found in opex_trades.csv — run ff_opex_signal.py first.")
        return 1

    print(f"Loaded {len(trades)} trades for 2024-2025")

    # ── Identify unique time windows ─────────────────────────────────────────
    windows = sorted({
        (t["entry_d"], t["front_exp"], t["back_exp"])
        for t in trades
    })
    print(f"Unique entry windows: {len(windows)}")
    for w in windows:
        print(f"  entry={w[0]}  front={w[1]}  back={w[2]}")

    opt_conn = conn("options", read_timeout=120)

    # ── Fetch daily dates and chains ─────────────────────────────────────────
    # For each window, get all available dates from entry_d (inclusive for entry
    # re-pricing) through front_exp (inclusive for expiry exit).
    all_dates_needed: set[date] = set()
    window_dates: dict[tuple, list[date]] = {}

    for entry_s, front_s, back_s in windows:
        entry_d = date.fromisoformat(entry_s)
        front_exp = date.fromisoformat(front_s)
        # entry date + daily dates between entry+1 and front_exp
        intra = get_trading_dates_in_range(entry_d, front_exp, opt_conn)
        all_dates = [entry_d] + intra   # entry_d first for re-pricing; intra for path
        window_dates[(entry_s, front_s, back_s)] = all_dates
        all_dates_needed.update(all_dates)

    all_dates_sorted = sorted(all_dates_needed)
    print(f"\nFetching {len(all_dates_sorted)} unique daily snapshots for {len(TEST_TICKERS)} tickers ...")

    chains: dict[date, dict] = {}
    for i, d in enumerate(all_dates_sorted):
        chains[d] = _fetch_option_prices(opt_conn, d, TEST_TICKERS)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(all_dates_sorted)} dates loaded")

    opt_conn.close()
    print(f"Chain data loaded.\n")

    # ── Build per-trade daily path ────────────────────────────────────────────
    results = []

    for entry_s, front_s, back_s in windows:
        entry_d    = date.fromisoformat(entry_s)
        front_exp  = date.fromisoformat(front_s)
        back_exp   = date.fromisoformat(back_s)
        path_dates = window_dates[(entry_s, front_s, back_s)]
        intra_dates = [d for d in path_dates if d > entry_d]  # monitoring dates

        edata = chains.get(entry_d, {})

        window_trades = [
            t for t in trades
            if t["entry_d"] == entry_s and t["front_exp"] == front_s
        ]

        for base_trade in window_trades:
            tk = base_trade["ticker"]

            # Re-derive entry strike from the entry-date chain (ATM by put-call parity)
            strike = atm_strike_from_chain(edata, tk, back_exp)
            if strike is None:
                continue

            # Validate entry pricing against stored values (should match)
            bm, _ = _get_straddle(edata, tk, back_exp, strike, "mid")
            fm, _ = _get_straddle(edata, tk, front_exp, strike, "mid")
            if bm is None or fm is None:
                continue
            entry_debit = bm - fm
            if entry_debit <= 0.05:
                continue

            # Theoretical max and profit
            t_front = float(base_trade["t_front"])
            t_back  = float(base_trade["t_back"])
            t_fwd   = float(base_trade["t_fwd"])
            if t_back <= 0 or t_fwd <= 0:
                continue
            max_theo   = bm * math.sqrt(t_fwd / t_back)
            max_profit = max_theo - entry_debit
            if max_profit <= 0:
                continue

            entry_ff = base_trade.get("ff_pct")  # may be None
            # Hold-to-expiry values from CSV
            hold_exit_mid = base_trade.get("exit_mid")
            hold_mid_ret  = base_trade.get("mid_ret")
            if hold_exit_mid is None or hold_mid_ret is None:
                continue

            # ── Build daily path ──────────────────────────────────────────
            # Exclude front_exp from TP trigger window — exit on front_exp IS
            # hold-to-expiry; including it makes early-exit vs hold comparison nonsensical.
            path = []
            path_expiry = None  # separate slot for front_exp day
            for check_d in intra_dates:
                cdata = chains.get(check_d, {})
                cal_v = calendar_mid(cdata, tk, front_exp, back_exp, strike, check_d)
                ff_v  = current_ff(cdata, tk, front_exp, back_exp, strike, check_d)
                if cal_v is not None:
                    entry = {
                        "date":    check_d,
                        "cal_val": cal_v,
                        "ff":      ff_v,
                        "achieved_frac": (cal_v - entry_debit) / max_profit,
                    }
                    if check_d == front_exp:
                        path_expiry = entry
                    else:
                        path.append(entry)

            if not path and path_expiry is None:
                continue

            # ── Apply exit strategies ─────────────────────────────────────
            # path       = intra-period checkpoints BEFORE front_exp (for early exits)
            # path_expiry= front_exp day value (= hold-to-expiry)
            strats = {}

            # TP exits
            for tp in TP_THRESHOLDS:
                tp_target = entry_debit + tp * max_profit
                hit = next((p for p in path if p["cal_val"] >= tp_target), None)
                strats[f"tp_{int(tp*100)}"] = hit  # dict or None

            # FF absolute exits (exit when FF drops BELOW threshold)
            for thresh in FF_ABS_EXITS:
                hit = next(
                    (p for p in path if p["ff"] is not None and p["ff"] < thresh),
                    None,
                )
                strats[f"ff_abs_{thresh:.0f}".replace("-","m")] = hit

            # FF relative exits (exit when FF drops below entry_ff * ratio)
            if entry_ff is not None and entry_ff > 0:
                for ratio in FF_REL_RATIOS:
                    rel_thresh = entry_ff * ratio
                    hit = next(
                        (p for p in path if p["ff"] is not None and p["ff"] < rel_thresh),
                        None,
                    )
                    strats[f"ff_rel_{int(ratio*100)}"] = hit
            else:
                for ratio in FF_REL_RATIOS:
                    strats[f"ff_rel_{int(ratio*100)}"] = None

            # Max intra-period calendar value (pre-expiry only = true early-exit peak)
            max_cal_val   = max((p["cal_val"] for p in path), default=None)
            max_achieved  = (max_cal_val - entry_debit) / max_profit if max_cal_val is not None else None
            # FF on first check date (should be close to entry_ff)
            first_ff = path[0]["ff"] if path else None
            # Flag: was this trade entered with a positive FF signal?
            ff_positive_entry = (entry_ff is not None and entry_ff > 0)

            record = {
                "entry_d":     entry_s,
                "front_exp":   front_s,
                "back_exp":    back_s,
                "ticker":      tk,
                "entry_debit": round(entry_debit, 4),
                "max_theo":    round(max_theo, 4),
                "max_profit":  round(max_profit, 4),
                "entry_ff":    entry_ff,
                "ff_positive_entry": int(ff_positive_entry),
                "first_check_ff": first_ff,
                "hold_exit_mid": hold_exit_mid,
                "hold_mid_ret":  hold_mid_ret,
                "n_checkpoints": len(path),
                "max_intra_cal": max_cal_val,
                "max_achieved_frac": max_achieved,
            }

            # Add per-strategy exit info
            for sname, hit in strats.items():
                if hit is not None:
                    exit_val = hit["cal_val"]
                    pnl      = (exit_val - entry_debit - IB_COMM) / entry_debit * 100
                    record[f"{sname}_exit_val"]  = round(exit_val, 4)
                    record[f"{sname}_mid_ret"]   = round(pnl, 4)
                    record[f"{sname}_days_held"] = (hit["date"] - entry_d).days
                    record[f"{sname}_exit_ff"]   = hit["ff"]
                    record[f"{sname}_beat_hold"] = int(pnl > hold_mid_ret)
                else:
                    record[f"{sname}_exit_val"]  = None
                    record[f"{sname}_mid_ret"]   = hold_mid_ret  # no trigger → held to expiry
                    record[f"{sname}_days_held"] = (front_exp - entry_d).days
                    record[f"{sname}_exit_ff"]   = None
                    record[f"{sname}_beat_hold"] = 0

            results.append(record)

    print(f"Processed {len(results)} trades with full daily path.")

    if not results:
        print("ERROR: no results — check Dolt data for 2024-2025 window dates.")
        return 1

    # ── Save CSV ──────────────────────────────────────────────────────────────
    fieldnames = list(results[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"Saved {len(results)} rows → {OUT_CSV.name}\n")

    # ── Summary tables ────────────────────────────────────────────────────────
    n = len(results)
    hold_rets = np.array([r["hold_mid_ret"] for r in results])

    ff_pos = [r for r in results if r["ff_positive_entry"]]
    n_ff_pos = len(ff_pos)

    # Entry FF distribution
    entry_ffs_all = [r["entry_ff"] for r in results if r["entry_ff"] is not None]
    if entry_ffs_all:
        ea = np.array(entry_ffs_all)
        section("ENTRY FF DISTRIBUTION (all 2024-2025 trades)")
        print(f"  n={len(ea)}  mean={ea.mean():+.1f}%  median={np.median(ea):+.1f}%  "
              f"p25={np.percentile(ea,25):+.1f}%  p75={np.percentile(ea,75):+.1f}%")
        print(f"  FF > 0%:  {(ea>0).sum()}/{len(ea)} trades  ({(ea>0).mean()*100:.0f}%)")
        print(f"  FF > 8%:  {(ea>8).sum()}/{len(ea)} trades  ({(ea>8).mean()*100:.0f}%)")
        print(f"  FF > 16%: {(ea>16).sum()}/{len(ea)} trades  ({(ea>16).mean()*100:.0f}%)")
        print(f"\n  NOTE: opex_trades.csv records ALL priced trades (no FF gate).")
        print(f"  Exit research is split: ALL trades + FF>0 sub-universe only.")

    section(f"BASELINE: Hold-to-expiry (n={n} all, n={n_ff_pos} with FF>0)")
    print(f"  ALL trades: median={np.median(hold_rets):+.1f}%  mean={np.mean(hold_rets):+.1f}%  "
          f"win={(hold_rets>0).mean()*100:.0f}%")
    if ff_pos:
        fp_rets = np.array([r["hold_mid_ret"] for r in ff_pos])
        print(f"  FF>0 only: median={np.median(fp_rets):+.1f}%  mean={np.mean(fp_rets):+.1f}%  "
              f"win={(fp_rets>0).mean()*100:.0f}%")

    # Max intra-period achieved fraction
    max_af = [r["max_achieved_frac"] for r in results if r["max_achieved_frac"] is not None]
    if max_af:
        max_af_arr = np.array(max_af)
        section("MAX INTRA-PERIOD ACHIEVED FRACTION (true upper bound for TP)")
        print("  What fraction of max_profit did the calendar reach DURING the period?")
        print("  (This is the actual peak, not the expiry value)\n")
        for pct in [10, 25, 50, 75, 90]:
            v = np.percentile(max_af_arr, pct)
            print(f"    p{pct:>3}: {v:+.3f}  ({v*100:+.1f}%)")
        for tp in TP_THRESHOLDS:
            rate = (max_af_arr >= tp).mean() * 100
            print(f"    % trades reaching TP={int(tp*100)}% intra-period: {rate:.1f}%")

    section("TP EXIT STRATEGIES vs Hold-to-Expiry  [pre-expiry triggers only]")
    print("  Early exit = exit BEFORE front_exp. TP hit on front_exp itself = hold-to-expiry.")
    print(f"\n{'Strategy':>12}{'n_hit':>8}{'hit%':>7}{'med_ret(hit)':>14}{'med_ret(hold)':>15}"
          f"{'% beat hold':>13}{'avg_days':>10}")
    for tp in TP_THRESHOLDS:
        sname = f"tp_{int(tp*100)}"
        hit_flag = f"{sname}_exit_val"
        ret_col  = f"{sname}_mid_ret"
        beat_col = f"{sname}_beat_hold"
        days_col = f"{sname}_days_held"

        hits   = [r for r in results if r[hit_flag] is not None]
        n_hit  = len(hits)
        if n_hit > 0:
            hit_rets    = np.array([r[ret_col] for r in hits])
            beat_pct    = np.mean([r[beat_col] for r in hits]) * 100
            avg_days    = np.mean([r[days_col] for r in hits])
            med_hold_for_hits = np.median([r["hold_mid_ret"] for r in hits])
            print(f"  TP={int(tp*100):>3}%  {n_hit:>6}{n_hit/n*100:>6.0f}%"
                  f"{np.median(hit_rets):>+13.1f}%{med_hold_for_hits:>+14.1f}%"
                  f"{beat_pct:>12.1f}%{avg_days:>9.1f}d")
        else:
            print(f"  TP={int(tp*100):>3}%  {0:>6}{'0':>6}%{'–':>14}{'–':>15}{'–':>13}{'–':>10}")

    def _ff_exit_table(row_set, label_suffix=""):
        print(f"{'Strategy':>15}{'n_hit':>8}{'hit%':>7}{'med_ret(hit)':>14}{'med_ret(hold)':>15}"
              f"{'% beat hold':>13}{'avg_days':>10}")
        nn = len(row_set)
        for thresh in FF_ABS_EXITS:
            sname = f"ff_abs_{thresh:.0f}".replace("-","m")
            hit_flag = f"{sname}_exit_val"
            ret_col  = f"{sname}_mid_ret"
            beat_col = f"{sname}_beat_hold"
            days_col = f"{sname}_days_held"
            hits = [r for r in row_set if r[hit_flag] is not None]
            n_hit = len(hits)
            if n_hit > 0:
                hit_rets = np.array([r[ret_col] for r in hits])
                beat_pct = np.mean([r[beat_col] for r in hits]) * 100
                avg_days = np.mean([r[days_col] for r in hits])
                med_hold_for_hits = np.median([r["hold_mid_ret"] for r in hits])
                print(f"  FF<{thresh:>4.0f}%  {n_hit:>6}{n_hit/nn*100:>6.0f}%"
                      f"{np.median(hit_rets):>+13.1f}%{med_hold_for_hits:>+14.1f}%"
                      f"{beat_pct:>12.1f}%{avg_days:>9.1f}d")
            else:
                print(f"  FF<{thresh:>4.0f}%  {0:>6}{'0':>6}%{'–':>14}{'–':>15}{'–':>13}{'–':>10}")

    section(f"FF ABSOLUTE EXIT (exit when FF drops below threshold) — ALL {n} trades")
    print("  WARNING: includes trades entered with negative FF (not a Campasano signal).")
    print("  See FF>0 sub-universe below for strategy-relevant analysis.\n")
    _ff_exit_table(results)

    if ff_pos:
        section(f"FF ABSOLUTE EXIT — FF>0 entry sub-universe (n={n_ff_pos})")
        print("  Only trades where FF > 0% at entry (term structure inverted = signal active).\n")
        _ff_exit_table(ff_pos)

    # FF relative exits — only meaningful for trades where entry_ff > 0
    if ff_pos:
        section(f"FF RELATIVE EXIT (exit when FF < entry_FF × ratio) — FF>0 entry (n={n_ff_pos})")
        print("  Ratio applies to entry FF. Exit fires when FF decays below that fraction.\n")
        print(f"{'Strategy':>15}{'n_hit':>8}{'hit%':>7}{'med_ret(hit)':>14}{'med_ret(hold)':>15}"
              f"{'% beat hold':>13}{'avg_days':>10}")
        for ratio in FF_REL_RATIOS:
            sname = f"ff_rel_{int(ratio*100)}"
            hit_flag = f"{sname}_exit_val"
            ret_col  = f"{sname}_mid_ret"
            beat_col = f"{sname}_beat_hold"
            days_col = f"{sname}_days_held"

            hits = [r for r in ff_pos if r[hit_flag] is not None]
            n_hit = len(hits)
            if n_hit > 0:
                hit_rets = np.array([r[ret_col] for r in hits])
                beat_pct = np.mean([r[beat_col] for r in hits]) * 100
                avg_days = np.mean([r[days_col] for r in hits])
                med_hold_for_hits = np.median([r["hold_mid_ret"] for r in hits])
                print(f"  {int(ratio*100):>3}% of FF  {n_hit:>6}{n_hit/n_ff_pos*100:>6.0f}%"
                      f"{np.median(hit_rets):>+13.1f}%{med_hold_for_hits:>+14.1f}%"
                      f"{beat_pct:>12.1f}%{avg_days:>9.1f}d")
            else:
                print(f"  {int(ratio*100):>3}% of FF  {0:>6}{'0':>6}%{'–':>14}{'–':>15}{'–':>13}{'–':>10}")

    # ── FF path statistics ────────────────────────────────────────────────────
    all_ff_obs = []
    for r in results:
        if r["entry_ff"] is not None and r["first_check_ff"] is not None:
            all_ff_obs.append({
                "entry_ff": r["entry_ff"],
                "first_check_ff": r["first_check_ff"],
            })
    if all_ff_obs:
        entry_ffs = np.array([o["entry_ff"] for o in all_ff_obs])
        check_ffs = np.array([o["first_check_ff"] for o in all_ff_obs])
        section("FF DECAY: entry FF vs first-check-date FF")
        print(f"  Entry FF  — mean={entry_ffs.mean():.1f}%  median={np.median(entry_ffs):.1f}%")
        print(f"  First-check FF — mean={check_ffs.mean():.1f}%  median={np.median(check_ffs):.1f}%")
        diff = check_ffs - entry_ffs
        print(f"  FF change (check − entry) — mean={diff.mean():+.1f}%  median={np.median(diff):+.1f}%")

    section("DATA LIMITATIONS")
    print(f"""
  1. SAMPLE SIZE: {n} trades across {len(windows)} opex windows (2024-2025 only).
     Results should be treated as directional, not statistically conclusive.
     A robust study requires 2019-2025 — but daily Dolt data starts ~2024.

  2. MID-PRICE EXITS: all P&L is mid-to-mid. Real exits cost bid-ask spread.
     Add ~0.5-1% to entry and exit to estimate realistic fills.

  3. TP / FF THRESHOLD CHOICE: optimal thresholds from this sample may not
     generalize. Treat them as a starting point for live observation.

  4. REGIME: 2024-2025 covers one bull+volatility regime. Strategy may behave
     differently in a bear market or sustained high-vol environment.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
