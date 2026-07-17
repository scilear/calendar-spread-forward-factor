#!/usr/bin/env python3
"""
ff_opex_validate.py — validate the monthly-opex calendar design BEFORE rebuilding ff_backtest.

Hypothesis: Dolt's historical (pre-2024) option_chain coverage is monthly-opex Fridays only.
If we structure calendars to enter on opex Friday (front=next opex, back=opex-after, exit=front
expiry opex), then (a) trades price cleanly across 2020-2023, and (b) spreads are tighter than
the random-date sample (so bid/ask becomes trustworthy).

This is a liquid-name proof of concept (not the full universe). Reuses ff_backtest pricing.
"""
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ff_backtest import _fetch_option_prices, _get_straddle  # reuse pricing
from ff_universe_scan import conn

# Liquid names + ETFs that should have options every year
TEST_TICKERS = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","BAC","XOM",
                "WMT","HD","DIS","NFLX","INTC","AMD","CSCO","PFE","KO","MCD",
                "SPY","QQQ","IWM","XLF","XLE"]


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    # first friday
    d += timedelta(days=(4 - d.weekday()) % 7)
    return d + timedelta(days=14)  # third friday


def covered_opex_dates(opt_conn, start_y=2020, end_y=2025) -> list[date]:
    """Third Fridays that actually have data in Dolt."""
    out = []
    with opt_conn.cursor() as cur:
        for y in range(start_y, end_y + 1):
            for m in range(1, 13):
                tf = third_friday(y, m)
                cur.execute("SELECT COUNT(*) FROM option_chain WHERE date=%s LIMIT 1", (tf.isoformat(),))
                if cur.fetchone()[0] > 0:
                    out.append(tf)
    return out


def atm_strike_from_chain(price_data, ticker, expiry):
    """ATM strike = call strike with |delta-0.5| smallest. Falls back to None."""
    td = price_data.get(ticker, {}).get(expiry, {})
    if not td:
        return None
    # We don't have delta in _fetch_option_prices output; approximate ATM as the strike
    # where call_mid and put_mid are closest (put-call parity => ATM).
    by_strike = defaultdict(dict)
    for (s, cp), px in td.items():
        cp0 = (cp or "").upper()[:1]
        if px.get("mid") is not None:
            by_strike[s][cp0] = px["mid"]
    best, best_gap = None, float("inf")
    for s, d in by_strike.items():
        if "C" in d and "P" in d:
            gap = abs(d["C"] - d["P"])
            if gap < best_gap:
                best_gap, best = gap, s
    return best


def main():
    opt_conn = conn("options", read_timeout=120)
    opex = covered_opex_dates(opt_conn)
    print(f"Covered opex dates {opex[0]} .. {opex[-1]}: {len(opex)} of "
          f"{(opex[-1].year-opex[0].year)*12+12} months\n")

    # Calendar structure: enter at snapshot[i], front=snapshot[i+1] (monthly M+1, = exit date),
    # back=snapshot[i+2] (monthly M+2). Exit at snapshot[i+1] (front expiry): back still listed,
    # front valued at intrinsic. Require consecutive ~monthly gaps.
    triples = [(opex[i], opex[i+1], opex[i+2]) for i in range(len(opex)-2)
               if 20 <= (opex[i+1]-opex[i]).days <= 40 and 20 <= (opex[i+2]-opex[i+1]).days <= 40]
    print(f"Consecutive-month triples (entry, front=exit, back): {len(triples)}\n")

    need_dates = sorted({d for t in triples for d in (t[0], t[1])})
    chains = {}
    for d in need_dates:
        chains[d] = _fetch_option_prices(opt_conn, d, TEST_TICKERS)
    opt_conn.close()

    def spot_from_chain(price_data, ticker, expiry, strike):
        """Estimate spot via put-call parity at ATM strike: S ~= K + (C_mid - P_mid)."""
        td = price_data.get(ticker, {}).get(expiry, {})
        c = td.get((strike, "Call"), {}).get("mid")
        p = td.get((strike, "Put"), {}).get("mid")
        if c is None or p is None:
            return None
        return strike + (c - p)

    results = []
    n_attempt = n_priced = 0
    for entry_d, front_exp, back_exp in triples:
        edata = chains.get(entry_d, {})
        xdata = chains.get(front_exp, {})   # exit at front expiry snapshot
        for tk in TEST_TICKERS:
            n_attempt += 1
            # ATM strike from the BACK chain at entry (longest-dated, most strikes near spot)
            strike = atm_strike_from_chain(edata, tk, back_exp)
            if strike is None:
                continue
            # Entry: both legs listed at entry snapshot
            fb, _ = _get_straddle(edata, tk, front_exp, strike, "bid")
            ba, _ = _get_straddle(edata, tk, back_exp,  strike, "ask")
            fm, _ = _get_straddle(edata, tk, front_exp, strike, "mid")
            bm, _ = _get_straddle(edata, tk, back_exp,  strike, "mid")
            # Exit: back leg priced from exit snapshot; front leg = intrinsic at expiry
            bx_bid, _ = _get_straddle(xdata, tk, back_exp, strike, "bid")
            bx_mid, _ = _get_straddle(xdata, tk, back_exp, strike, "mid")
            bx_ask, _ = _get_straddle(xdata, tk, back_exp, strike, "ask")
            s_exit = spot_from_chain(xdata, tk, back_exp, strike)
            if None in (fm, bm, bx_mid) or s_exit is None:
                continue
            front_intrinsic = abs(s_exit - strike)   # ATM straddle intrinsic at expiry
            entry_mid   = bm - fm
            entry_cross = (ba - fb) if (ba is not None and fb is not None) else None
            exit_mid    = bx_mid - front_intrinsic
            exit_cross  = (bx_bid - front_intrinsic) if bx_bid is not None else None
            if entry_mid <= 0.05:
                continue
            n_priced += 1
            results.append({
                "year": entry_d.year,
                "entry_mid": entry_mid, "entry_cross": entry_cross,
                "exit_mid": exit_mid, "exit_cross": exit_cross,
            })

    print(f"Priced {n_priced}/{n_attempt} liquid-name calendar attempts "
          f"({n_priced/max(n_attempt,1)*100:.0f}%)\n")

    # Drop rate + spread realism by year
    by_year = defaultdict(list)
    for r in results:
        by_year[r["year"]].append(r)
    print(f"{'year':>6}{'n':>6}{'entry mid':>11}{'entry spread%':>15}{'mid ret%':>10}{'cross ret%':>12}")
    IB = 0.052
    for y in sorted(by_year):
        rs = by_year[y]
        em = np.array([r["entry_mid"] for r in rs])
        ec = np.array([r["entry_cross"] for r in rs if r["entry_cross"] is not None])
        # spread% = (cross-mid)/mid for entries with both
        spr = np.array([(r["entry_cross"]-r["entry_mid"])/r["entry_mid"]*100
                        for r in rs if r["entry_cross"] is not None and r["entry_mid"]>0])
        mid_ret = np.array([(r["exit_mid"]-r["entry_mid"]-IB)/r["entry_mid"]*100 for r in rs])
        cross_ret = np.array([(r["exit_cross"]-r["entry_cross"]-IB)/r["entry_cross"]*100
                              for r in rs if r["entry_cross"] and r["exit_cross"] is not None and r["entry_cross"]>0])
        print(f"{y:>6}{len(rs):>6}{np.median(em):>11.2f}{np.median(spr) if len(spr) else float('nan'):>14.1f}%"
              f"{np.median(mid_ret):>+9.1f}%{(np.median(cross_ret) if len(cross_ret) else float('nan')):>+11.1f}%")

    # Compare to the random-date sample (the +113% cross blowup): here, how bad is cross?
    allspr = np.array([(r["entry_cross"]-r["entry_mid"])/r["entry_mid"]*100
                       for r in results if r["entry_cross"] is not None and r["entry_mid"]>0])
    if len(allspr):
        print(f"\nOpex-aligned liquid-name entry spread (cross vs mid): "
              f"median {np.median(allspr):.1f}%  p75 {np.percentile(allspr,75):.1f}%  p90 {np.percentile(allspr,90):.1f}%")
    else:
        print("\nNo priced trades — check expiry matching.")
    print("(Compare: random-date full-universe sample had ~200%+ spreads / -113% cross returns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
