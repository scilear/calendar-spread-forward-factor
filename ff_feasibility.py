#!/usr/bin/env python3
"""
Forward Factor feasibility check — counts how often FF > 16% fires on the
watchlist ETFs vs. individual stocks using the Dolt historical database.

Purpose: Before committing to T6b (full backtest), verify the signal
actually fires enough times on the proposed universe to be testable.

Usage:
    python ff_feasibility.py
    python ff_feasibility.py --tickers SPY QQQ IWM --samples 100

Dolt DB: localhost:3307, options.option_chain (2019-present)
         stocks.ohlcv (for spot price / ATM strike selection)
"""

import argparse
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import sys
from pathlib import Path as _P
_d = _P(__file__).resolve().parent
while not (_d / 'db.py').exists() and _d.parent != _d:
    _d = _d.parent
if str(_d) not in sys.path:
    sys.path.insert(0, str(_d))
from db import earningsvol_conn

import pandas as pd
import yfinance as yf

WATCHLIST_ETFS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "GLD", "TLT", "EFA"]

FRONT_DTE_WINDOW = (20, 45)
BACK_DTE_WINDOW = (45, 80)
FF_THRESHOLD = 16.0


def forward_vol(front_iv: float, front_t: float, back_iv: float, back_t: float) -> float:
    var = back_t * back_iv ** 2 - front_t * front_iv ** 2
    return math.sqrt(max(0.0, var / (back_t - front_t)))


def compute_ff(front_iv: float, fwd: float) -> float | None:
    # Campasano (2018) formula: 1mIV/FV(1,1) - 1, expressed as %
    if fwd <= 0:
        return None
    return (front_iv / fwd - 1) * 100





def get_spot(conn, ticker: str, quote_date: date) -> float | None:
    """Get closing price from stocks.ohlcv."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT close FROM ohlcv WHERE act_symbol=%s AND date=%s",
            (ticker, quote_date.isoformat()),
        )
        row = cur.fetchone()
    return float(row[0]) if row else None


def get_options_for_date(conn, ticker: str, quote_date: date) -> list[dict]:
    """Fetch all option rows for a ticker on a specific date."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT expiration, strike, call_put, vol, delta, bid, ask
            FROM option_chain
            WHERE act_symbol=%s AND date=%s
            ORDER BY expiration, strike
            """,
            (ticker, quote_date.isoformat()),
        )
        rows = cur.fetchall()

    result = []
    for r in rows:
        result.append({
            "expiry": r[0],
            "strike": float(r[1]) if r[1] is not None else None,
            "right": r[2],  # "Call" or "Put"
            "iv": float(r[3]) if r[3] is not None else None,
            "delta": float(r[4]) if r[4] is not None else None,
            "bid": float(r[5]) if r[5] is not None else None,
            "ask": float(r[6]) if r[6] is not None else None,
        })
    return result


def compute_ff_for_date(ticker: str, quote_date: date, spot: float, rows: list[dict]) -> dict | None:
    """
    From raw option rows, compute the Forward Factor for the best front/back pair.
    Uses ATM selection by delta closest to 0.50 for calls.
    """
    today = quote_date

    # Group by expiry
    exps: dict[date, list[dict]] = defaultdict(list)
    for r in rows:
        if r["expiry"] and r["right"] in ("Call", "C") and r["iv"] and r["iv"] > 0:
            exps[r["expiry"]].append(r)

    if not exps:
        return None

    # Find front expiry (~30 DTE) and back expiry (~60 DTE)
    front_exp = None
    back_exp = None
    front_dte_actual = None
    back_dte_actual = None

    for exp in sorted(exps.keys()):
        dte = (exp - today).days
        if FRONT_DTE_WINDOW[0] <= dte <= FRONT_DTE_WINDOW[1] and front_exp is None:
            front_exp = exp
            front_dte_actual = dte
        elif BACK_DTE_WINDOW[0] <= dte <= BACK_DTE_WINDOW[1] and back_exp is None:
            back_exp = exp
            back_dte_actual = dte

    if front_exp is None or back_exp is None:
        return None

    def atm_call(exp_rows):
        # Prefer delta-based ATM, fall back to strike-based
        by_delta = [(abs((r["delta"] or 0) - 0.50), r) for r in exp_rows if r["delta"] is not None]
        if by_delta:
            return min(by_delta, key=lambda x: x[0])[1]
        by_strike = [(abs((r["strike"] or 0) - spot), r) for r in exp_rows if r["strike"] is not None]
        return min(by_strike, key=lambda x: x[0])[1] if by_strike else None

    front_atm = atm_call(exps[front_exp])
    back_atm = atm_call(exps[back_exp])

    if not front_atm or not back_atm:
        return None
    if not front_atm["iv"] or not back_atm["iv"]:
        return None

    front_iv = front_atm["iv"]
    back_iv = back_atm["iv"]
    front_t = front_dte_actual / 365.0
    back_t = back_dte_actual / 365.0

    fwd = forward_vol(front_iv, front_t, back_iv, back_t)
    ff = compute_ff(front_iv, fwd)

    return {
        "date": quote_date,
        "ticker": ticker,
        "front_exp": front_exp,
        "back_exp": back_exp,
        "front_dte": front_dte_actual,
        "back_dte": back_dte_actual,
        "front_iv": round(front_iv, 4),
        "forward_vol": round(fwd, 4),
        "ff_pct": round(ff, 2) if ff is not None else None,
        "ff_signal": ff is not None and ff >= FF_THRESHOLD,
    }


def sample_dates(start: date, end: date, n: int) -> list[date]:
    """Return ~n evenly-spaced dates between start and end."""
    total_days = (end - start).days
    if total_days <= 0:
        return [start]
    step = max(1, total_days // n)
    dates = []
    d = start
    while d <= end and len(dates) < n:
        dates.append(d)
        d += timedelta(days=step)
    return dates


def run_feasibility(tickers: list[str], n_samples: int) -> pd.DataFrame:
    opt_conn = earningsvol_conn()
    stk_conn = earningsvol_conn()

    start = date(2019, 2, 1)
    end = date(2026, 6, 1)
    sample_dates_list = sample_dates(start, end, n_samples)

    records = []
    for ticker in tickers:
        print(f"\n  {ticker} ({len(sample_dates_list)} sample dates) ...")
        for d in sample_dates_list:
            spot = get_spot(stk_conn, ticker, d)
            if spot is None:
                continue  # not a trading day or no data
            rows = get_options_for_date(opt_conn, ticker, d)
            if not rows:
                continue
            rec = compute_ff_for_date(ticker, d, spot, rows)
            if rec is not None:
                records.append(rec)
                if rec["ff_signal"]:
                    print(f"    SIGNAL {d}: FF={rec['ff_pct']:.1f}%")

    opt_conn.close()
    stk_conn.close()

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    if df.empty:
        print("\nNo data — check Dolt connectivity and ticker availability.")
        return

    df["year"] = pd.to_datetime(df["date"]).dt.year

    print("\n=== FEASIBILITY SUMMARY ===")
    print(f"Total observations: {len(df)}")
    print(f"Observations with FF data: {df['ff_pct'].notna().sum()}")

    # Per-ticker summary
    print("\n--- FF > 16% frequency by ticker ---")
    print(f"{'Ticker':<8} {'Obs':>6} {'FF>16%':>8} {'Rate%':>8}  {'Median FF':>10}")
    for ticker, g in df.groupby("ticker"):
        signals = g["ff_signal"].sum()
        rate = signals / len(g) * 100 if len(g) > 0 else 0
        med = g["ff_pct"].median()
        print(f"  {ticker:<6}  {len(g):>6}  {signals:>6}  {rate:>7.1f}%  {med:>10.2f}%")

    # Per-year summary
    print("\n--- FF > 16% by year (all tickers) ---")
    for year, g in df.groupby("year"):
        signals = g["ff_signal"].sum()
        print(f"  {year}: {signals}/{len(g)} signals ({signals/len(g)*100:.1f}%)")

    # Verdict
    total_rate = df["ff_signal"].mean() * 100 if len(df) > 0 else 0
    print(f"\nOverall signal rate on ETF universe: {total_rate:.1f}%")
    if total_rate < 5:
        print("⚠ WARNING: Very sparse signal on ETF universe.")
        print("  Recommend: add liquid single names to universe for meaningful backtest.")
        print("  (Campasano edge is primarily single-name, idiosyncratic-event driven.)")
    elif total_rate < 15:
        print("NOTE: Moderate signal frequency. Backtest feasible but sample may be thin.")
        print("      Consider adding individual stocks to broaden signal base.")
    else:
        print("✓ Adequate signal frequency. Universe suitable for T6b backtest.")


def main():
    parser = argparse.ArgumentParser(description="FF feasibility check on Dolt historical data")
    parser.add_argument("--tickers", nargs="+", default=WATCHLIST_ETFS)
    parser.add_argument("--samples", type=int, default=60,
                        help="Number of sample dates per ticker (default 60)")
    args = parser.parse_args()

    print(f"FF Feasibility Check")
    print(f"Universe: {', '.join(args.tickers)}")
    print(f"Sample dates: {args.samples} per ticker, 2019-2026")

    df = run_feasibility(args.tickers, args.samples)

    if not df.empty:
        out_path = Path(__file__).parent / "ff_feasibility_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nFull results saved: {out_path}")

    print_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
