#!/usr/bin/env python3
"""
ff_opex_signal.py — cheap checkpoint: does the FF signal separate winners from losers
on opex-aligned calendars across 2020-2025 regimes?

Uses the validated opex-calendar structure from ff_opex_validate.py:
  enter  = opex[i] (third Friday)
  front  = opex[i+1]  (exit date = front expiry)
  back   = opex[i+2]
  exit   = front-expiry opex snapshot; front valued at intrinsic

FF is computed on EXACTLY the expirations in the calendar (not via DTE-window search,
which for liquid weekly-options names would pick up a weekly instead of our monthly).
We use the IV from the nearest available call strike to ATM at each expiry.

Tests multiple FF thresholds (0, 8, 12, 16, 20%) to find the selectivity curve.
"""
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ff_backtest import _fetch_option_prices, _get_straddle
from ff_universe_scan import conn, forward_vol, compute_ff

# Same liquid-name universe as ff_opex_validate
TEST_TICKERS = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","BAC","XOM",
                "WMT","HD","DIS","NFLX","INTC","AMD","CSCO","PFE","KO","MCD",
                "SPY","QQQ","IWM","XLF","XLE"]

IB_COMM = 0.052   # $5.20 per share roundtrip (same as backtest)
FF_THRESHOLDS = [0, 8, 12, 16, 20]


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)
    return d + timedelta(days=14)


def covered_opex_dates(opt_conn, start_y=2020, end_y=2025) -> list[date]:
    out = []
    with opt_conn.cursor() as cur:
        for y in range(start_y, end_y + 1):
            for m in range(1, 13):
                tf = third_friday(y, m)
                cur.execute("SELECT COUNT(*) FROM option_chain WHERE date=%s LIMIT 1", (tf.isoformat(),))
                if cur.fetchone()[0] > 0:
                    out.append(tf)
    return out


def atm_iv_for_expiry(price_data, ticker, expiry, strike):
    """Get call IV at the ATM strike (searching within ±5% of strike if exact not available)."""
    td = price_data.get(ticker, {}).get(expiry, {})
    if not td:
        return None
    # Prefer exact strike, then nearest call within 5%
    best_iv, best_dist = None, float("inf")
    for (s, cp), px in td.items():
        cp0 = (cp or "").upper()[:1]
        if cp0 != "C":
            continue
        iv = px.get("iv")
        if iv is None or iv <= 0:
            continue
        dist = abs(s - strike)
        if dist < best_dist:
            best_dist, best_iv = dist, iv
    # Only accept if strike is within 5% of the ATM estimate
    if best_iv is not None and best_dist / max(strike, 1) > 0.05:
        return None
    return best_iv


def trade_ff(price_data, ticker, front_exp, back_exp, strike, entry_d):
    """Compute FF using EXACTLY the calendar's front and back expirations.
    Avoids the DTE-window mismatch bug where ff_for_ticker_date picks up weekly
    options instead of the monthly expirations we're actually trading."""
    front_iv = atm_iv_for_expiry(price_data, ticker, front_exp, strike)
    back_iv  = atm_iv_for_expiry(price_data, ticker, back_exp,  strike)
    if front_iv is None or back_iv is None:
        return None
    front_dte = (front_exp - entry_d).days
    back_dte  = (back_exp  - entry_d).days
    if front_dte <= 0 or back_dte <= front_dte:
        return None
    fwd = forward_vol(front_iv, front_dte / 365, back_iv, back_dte / 365)
    return compute_ff(front_iv, fwd)


def atm_strike_from_chain(price_data, ticker, expiry):
    """ATM strike = where |call_mid - put_mid| is smallest (put-call parity)."""
    td = price_data.get(ticker, {}).get(expiry, {})
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


def spot_from_chain(price_data, ticker, expiry, strike):
    td = price_data.get(ticker, {}).get(expiry, {})
    c = td.get((strike, "Call"), {}).get("mid")
    p = td.get((strike, "Put"), {}).get("mid")
    if c is None or p is None:
        return None
    return strike + (c - p)


def main():
    opt_conn = conn("options", read_timeout=120)
    opex = covered_opex_dates(opt_conn)
    print(f"Covered opex dates: {opex[0]} .. {opex[-1]}, {len(opex)} months\n")

    triples = [(opex[i], opex[i+1], opex[i+2]) for i in range(len(opex)-2)
               if 20 <= (opex[i+1]-opex[i]).days <= 40
               and 20 <= (opex[i+2]-opex[i+1]).days <= 40]
    print(f"Valid entry triples: {len(triples)}\n")

    # Collect all unique dates we need to fetch (same chain used for pricing + FF)
    need_dates = sorted({d for t in triples for d in (t[0], t[1])})
    chains = {}

    print(f"Fetching option chains for {len(need_dates)} dates...")
    for i, d in enumerate(need_dates):
        chains[d] = _fetch_option_prices(opt_conn, d, TEST_TICKERS)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(need_dates)}")
    opt_conn.close()
    print()

    records = []
    n_attempt = n_priced = n_ff_computed = 0

    for entry_d, front_exp, back_exp in triples:
        edata = chains.get(entry_d, {})
        xdata = chains.get(front_exp, {})

        for tk in TEST_TICKERS:
            n_attempt += 1

            # ATM strike from back chain at entry
            strike = atm_strike_from_chain(edata, tk, back_exp)
            if strike is None:
                continue

            # Entry pricing (mid)
            fm, _ = _get_straddle(edata, tk, front_exp, strike, "mid")
            bm, _ = _get_straddle(edata, tk, back_exp,  strike, "mid")
            # Exit pricing (back leg at front-expiry snapshot; front = intrinsic)
            bx_mid, _ = _get_straddle(xdata, tk, back_exp, strike, "mid")
            bx_bid, _ = _get_straddle(xdata, tk, back_exp, strike, "bid")
            fb_entry, _ = _get_straddle(edata, tk, front_exp, strike, "bid")
            ba_entry, _ = _get_straddle(edata, tk, back_exp,  strike, "ask")
            s_exit = spot_from_chain(xdata, tk, back_exp, strike)

            if None in (fm, bm, bx_mid) or s_exit is None:
                continue
            front_intrinsic = abs(s_exit - strike)
            entry_mid = bm - fm
            if entry_mid <= 0.05:
                continue

            exit_mid   = bx_mid - front_intrinsic
            entry_cross = (ba_entry - fb_entry) if (ba_entry is not None and fb_entry is not None) else None
            exit_cross  = (bx_bid - front_intrinsic) if bx_bid is not None else None

            # FF computed on exactly front_exp / back_exp — no DTE-window mismatch
            ff_val = trade_ff(edata, tk, front_exp, back_exp, strike, entry_d)
            if ff_val is not None:
                n_ff_computed += 1

            n_priced += 1
            t_front = (front_exp - entry_d).days
            t_back  = (back_exp  - entry_d).days
            t_fwd   = t_back - t_front
            mid_ret  = (exit_mid - entry_mid - IB_COMM) / entry_mid * 100
            cross_ret = None
            if entry_cross and exit_cross and entry_cross > 0:
                cross_ret = (exit_cross - entry_cross - IB_COMM) / entry_cross * 100

            records.append({
                "year":            entry_d.year,
                "entry_d":         entry_d.isoformat(),
                "front_exp":       front_exp.isoformat(),
                "back_exp":        back_exp.isoformat(),
                "ticker":          tk,
                "ff_pct":          ff_val,
                "entry_mid":       entry_mid,
                "back_straddle":   bm,        # back leg at entry (mid)
                "front_straddle":  fm,        # front leg at entry (mid)
                "t_front":         t_front,   # DTE of front at entry
                "t_back":          t_back,    # DTE of back at entry
                "t_fwd":           t_fwd,     # days in forward period
                "exit_mid":        exit_mid,  # raw exit value (calendar mid at expiry)
                "front_intrinsic": front_intrinsic,
                "entry_cross":     entry_cross,
                "exit_cross":      exit_cross,
                "mid_ret":         mid_ret,
                "cross_ret":       cross_ret,
            })

    print(f"Priced: {n_priced}/{n_attempt} ({n_priced/max(n_attempt,1)*100:.0f}%)")
    print(f"FF computed: {n_ff_computed}/{n_priced} ({n_ff_computed/max(n_priced,1)*100:.0f}%)\n")

    if not records:
        print("No records priced — check Dolt connection.")
        return 1

    recs = np.array(records)  # keep as list for indexing
    years = sorted({r["year"] for r in records})

    # ── SECTION 1: Unconditional by year ─────────────────────────────────────
    print("── Unconditional (all priced, mid returns) ──")
    print(f"{'year':>6}{'n':>6}{'med ret%':>10}{'mean ret%':>11}{'win%':>7}")
    for y in years:
        yr = [r for r in records if r["year"] == y]
        rets = np.array([r["mid_ret"] for r in yr])
        print(f"{y:>6}{len(yr):>6}{np.median(rets):>+9.1f}%{np.mean(rets):>+10.1f}%"
              f"{(rets>0).mean()*100:>6.0f}%")
    all_rets = np.array([r["mid_ret"] for r in records])
    print(f"{'ALL':>6}{len(records):>6}{np.median(all_rets):>+9.1f}%{np.mean(all_rets):>+10.1f}%"
          f"{(all_rets>0).mean()*100:>6.0f}%\n")

    # ── SECTION 2: FF threshold sweep ────────────────────────────────────────
    ff_records = [r for r in records if r["ff_pct"] is not None]
    print(f"Records with FF signal: {len(ff_records)}/{len(records)}\n")

    print("── FF threshold sweep (mid returns, top-FF filtered) ──")
    print(f"{'FF>':>6}{'n':>6}{'n_pass':>8}{'pass%':>7}{'med ret%':>10}{'mean ret%':>11}{'win%':>7}")
    for thresh in FF_THRESHOLDS:
        passed = [r for r in ff_records if r["ff_pct"] >= thresh]
        if not passed:
            print(f"{thresh:>5}%{len(ff_records):>6}{0:>8}{'–':>7}{'–':>10}{'–':>11}{'–':>7}")
            continue
        rets = np.array([r["mid_ret"] for r in passed])
        pct_pass = len(passed) / len(ff_records) * 100
        print(f"{thresh:>5}%{len(ff_records):>6}{len(passed):>8}{pct_pass:>6.0f}%"
              f"{np.median(rets):>+9.1f}%{np.mean(rets):>+10.1f}%{(rets>0).mean()*100:>6.0f}%")
    print()

    # ── SECTION 3: FF signal by year (best threshold = 16%) ──────────────────
    BEST_THRESH = 16
    print(f"── FF >= {BEST_THRESH}% filtered, by year ──")
    print(f"{'year':>6}{'n_all':>7}{'n_pass':>8}{'pass%':>7}{'med ret%':>10}{'mean ret%':>11}{'win%':>7}")
    for y in years:
        yr = [r for r in ff_records if r["year"] == y]
        passed = [r for r in yr if r["ff_pct"] >= BEST_THRESH]
        if not passed:
            print(f"{y:>6}{len(yr):>7}{0:>8}{'–':>7}{'–':>10}{'–':>11}{'–':>7}")
            continue
        rets = np.array([r["mid_ret"] for r in passed])
        pct = len(passed) / len(yr) * 100 if yr else 0
        print(f"{y:>6}{len(yr):>7}{len(passed):>8}{pct:>6.0f}%"
              f"{np.median(rets):>+9.1f}%{np.mean(rets):>+10.1f}%{(rets>0).mean()*100:>6.0f}%")
    all_passed = [r for r in ff_records if r["ff_pct"] >= BEST_THRESH]
    if all_passed:
        rets = np.array([r["mid_ret"] for r in all_passed])
        print(f"{'ALL':>6}{len(ff_records):>7}{len(all_passed):>8}{len(all_passed)/len(ff_records)*100:>6.0f}%"
              f"{np.median(rets):>+9.1f}%{np.mean(rets):>+10.1f}%{(rets>0).mean()*100:>6.0f}%")
    print()

    # ── SECTION 4: FF percentile distribution ────────────────────────────────
    if ff_records:
        ff_vals = np.array([r["ff_pct"] for r in ff_records])
        print(f"FF distribution (n={len(ff_vals)}): "
              f"p10={np.percentile(ff_vals,10):.1f}%  "
              f"p25={np.percentile(ff_vals,25):.1f}%  "
              f"med={np.median(ff_vals):.1f}%  "
              f"p75={np.percentile(ff_vals,75):.1f}%  "
              f"p90={np.percentile(ff_vals,90):.1f}%\n")

        # EV by FF quintile
        q_edges = np.percentile(ff_vals, [0, 20, 40, 60, 80, 100])
        print("── EV by FF quintile ──")
        print(f"{'FF range':>18}{'n':>6}{'med ret%':>10}{'mean ret%':>11}{'win%':>7}")
        for qi in range(5):
            lo, hi = q_edges[qi], q_edges[qi+1]
            bucket = [r for r in ff_records if lo <= r["ff_pct"] <= hi]
            if not bucket:
                continue
            rets = np.array([r["mid_ret"] for r in bucket])
            print(f"{lo:>7.1f}–{hi:<7.1f}%{len(bucket):>6}"
                  f"{np.median(rets):>+9.1f}%{np.mean(rets):>+10.1f}%{(rets>0).mean()*100:>6.0f}%")

    # Save raw records for exit research
    import csv
    out_path = Path(__file__).parent / "opex_trades.csv"
    if records:
        fieldnames = list(records[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(records)
        print(f"\nSaved {len(records)} records to {out_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
