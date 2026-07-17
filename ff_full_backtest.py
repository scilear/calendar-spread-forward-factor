#!/usr/bin/env python3
"""
ff_full_backtest.py — comprehensive FF calendar-spread backtest.

Data reality (verified from Dolt):
  - Dolt stores exactly 3 expirations per (date, ticker): SHORT (~14d), MEDIUM (~28d), LONG (~60d)
  - Pre-2024 : monthly opex snapshots only (~12 per year per ticker)
  - 2024+    : Mon/Wed/Fri (~3/week, ~150/year)
  - A given MEDIUM expiry stays in the chain for ~4-6 consecutive observations (~1.5 weeks),
    then the chain rolls to the next expiry.

Strategy: for each (date, ticker), enter a MEDIUM-vs-LONG calendar if DTE windows satisfied.
Path tracking: follow the same (front_exp, back_exp) pair on each subsequent date until
the pair is no longer available in the chain (chain roll = natural close).
Exit also captured on front_exp day via stock close price + back straddle (hold-to-expiry).

Output files:
  ff_all_trades.csv    one row per trade — entry metadata + hold-to-last-quote return
  ff_daily_paths.csv   one row per (trade_id, obs_date) — calendar value + current FF

Usage:
  python ff_full_backtest.py                          # 2020-2025, all tickers
  python ff_full_backtest.py --start 2024-01-01       # specific start date
  python ff_full_backtest.py --tickers SPY QQQ AAPL   # specific tickers
"""
import argparse
import csv
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

sys.path.insert(0, str(Path(__file__).parent))
from ff_universe_scan import forward_vol, compute_ff

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = [
    # Mega-cap tech
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","AMD","INTC",
    "QCOM","AMAT","LRCX","KLAC","ADBE","CRM","ORCL","CSCO","IBM","TXN",
    "MU","NFLX","ADSK","INTU","AVGO","NOW","SNPS","CDNS","PANW","FTNT",
    # ETFs — broad + sector (QQQ/IWM not in Dolt options DB)
    "SPY","DIA","MDY",
    "XLK","XLF","XLV","XLE","XLI","XLU","XLY","XLP","XLB","XLC","XLRE",
    "XBI","XRT","XHB","XME","XOP","KRE","XSD","XAR",
    # Financials
    "JPM","BAC","GS","MS","C","WFC","USB","PNC","COF","SCHW","BLK","AXP",
    "V","MA","SPGI","MCO","CME","ICE","CB","TRV","AIG","PRU","MET","AFL",
    # Healthcare
    "UNH","CVS","JNJ","PFE","MRK","ABBV","BMY","GILD","AMGN","REGN","BIIB",
    "VRTX","MDT","ABT","DHR","TMO","HUM","EW","ILMN","ISRG","SYK","BSX",
    # Energy
    "XOM","CVX","COP","EOG","OXY","DVN","HAL","SLB","MPC","VLO","PSX","HES","FCX",
    # Consumer staples / discretionary
    "WMT","HD","TGT","LOW","COST","SBUX","MCD","NKE","KO","PEP","MO",
    "PM","MDLZ","CL","PG","KR","DLTR","DG","TJX","ROST","EL","ULTA","LULU",
    # Industrials / defense
    "CAT","BA","HON","GE","RTX","LMT","NOC","GD","MMM","DE","EMR","ITW",
    "FDX","UPS","NSC","UNP","CSX","WM","RSG","CTAS","FAST",
    # Communication / media
    "DIS","NFLX","CMCSA","T","VZ","TMUS","CHTR","FOXA",
    # Consumer discretionary / travel
    "AMZN","TSLA","ABNB","UBER","RCL","MAR","HLT","DAL","UAL","LVS","WYNN",
    "MGM","EXPE","BKNG","CMG","DPZ","YUM","MCD",
    # Real estate / utilities
    "NEE","DUK","SO","EXC","AEP","D","PEG","XEL","ETR","EIX","PCG",
    "PLD","AMT","CCI","EQIX","SPG","O","DLR","WELL","VTR","EQR",
    # Other large-caps
    "LLY","CAH","MCK","SHW","ECL","APD","PPG","LIN","DOW","DD","EMN",
    "FICO","VRSK","MSCI","IQV","A","DHI","LEN","PHM","NUE","FCX",
    "ADM","CF","FMC","MOS","VMC","MLM","URI","PWR","LDOS","LHX",
]
# Deduplicate while preserving order
seen = set()
_uniq = []
for t in DEFAULT_TICKERS:
    if t not in seen:
        seen.add(t)
        _uniq.append(t)
DEFAULT_TICKERS = _uniq

FRONT_DTE_MIN, FRONT_DTE_MAX = 20, 45
BACK_DTE_MIN,  BACK_DTE_MAX  = 46, 80
ATM_TOL_PCT  = 0.015   # ±1.5% of strike for ATM matching
FETCH_CHUNK  = 100     # increased from 25 — fewer queries per date
IB_COMM      = 0.052

OUT_DIR = Path(__file__).parent


# ── Dolt helpers ──────────────────────────────────────────────────────────────

def get_all_dolt_dates(c, start: date, end: date) -> list[date]:
    """
    Query quarter-by-quarter with a long per-query timeout.
    Each quarterly slice scans ≤3 months of the table; fast even on cold NFS cache.
    Uses a fresh high-timeout connection per chunk to avoid mid-loop disconnects.
    """
    quarters = []
    d = date(start.year, ((start.month - 1) // 3) * 3 + 1, 1)
    while d <= end:
        q_end_month = d.month + 2
        q_end_year  = d.year + (q_end_month - 1) // 12
        q_end_month = ((q_end_month - 1) % 12) + 1
        import calendar
        q_end = date(q_end_year, q_end_month,
                     calendar.monthrange(q_end_year, q_end_month)[1])
        quarters.append((max(start, d), min(end, q_end)))
        # advance to next quarter
        nm = d.month + 3
        d = date(d.year + (nm - 1) // 12, ((nm - 1) % 12) + 1, 1)

    all_dates = []
    for q_start, q_end in quarters:
        c_q = earningsvol_conn()
        try:
            with c_q.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT date FROM option_chain "
                    "WHERE act_symbol='SPY' AND date >= %s AND date <= %s ORDER BY date",
                    (q_start.isoformat(), q_end.isoformat()),
                )
                rows = cur.fetchall()
                all_dates.extend(r[0] for r in rows)
                print(f"    {q_start} .. {q_end}: {len(rows)} dates", flush=True)
        finally:
            c_q.close()
    return all_dates


def get_spot_close(c_stocks, ticker: str, d: date) -> float | None:
    """Close price from stocks.ohlcv for hold-to-expiry intrinsic calculation."""
    with c_stocks.cursor() as cur:
        cur.execute("SELECT close FROM ohlcv WHERE act_symbol=%s AND date=%s",
                    (ticker, d.isoformat()))
        row = cur.fetchone()
    return float(row[0]) if row else None


# ── Option chain fetching ─────────────────────────────────────────────────────

def fetch_chains(c, query_date: date, tickers: list[str]) -> dict:
    """
    Returns {ticker: {expiry_date: {(strike, cp): {bid, ask, mid, iv}}}}
    """
    result = defaultdict(lambda: defaultdict(dict))
    for i in range(0, len(tickers), FETCH_CHUNK):
        chunk = tickers[i : i + FETCH_CHUNK]
        ph = ",".join(["%s"] * len(chunk))
        with c.cursor() as cur:
            cur.execute(
                f"SELECT act_symbol, expiration, strike, call_put, bid, ask, vol "
                f"FROM option_chain WHERE date=%s AND act_symbol IN ({ph})",
                [query_date.isoformat()] + chunk,
            )
            for row in cur.fetchall():
                tk, exp, s_raw, cp, bid_, ask_, vol_ = row
                if s_raw is None:
                    continue
                s = round(float(s_raw), 2)
                b = float(bid_) if bid_ is not None else None
                a = float(ask_) if ask_ is not None else None
                m = (b + a) / 2 if (b is not None and a is not None) else None
                v = float(vol_) if vol_ is not None else None
                result[tk][exp][(s, cp)] = {"bid": b, "ask": a, "mid": m, "iv": v}
    return result


# ── Pricing ───────────────────────────────────────────────────────────────────

def find_expirations(chains: dict, ticker: str, obs_date: date):
    """
    Find (front_exp, back_exp) from whatever expirations are in the chain.
    Uses first expiry in FRONT_DTE range and first in BACK_DTE range.
    Returns (None, None) if not available.
    """
    exps = sorted(chains.get(ticker, {}).keys())
    front = back = None
    for exp in exps:
        dte = (exp - obs_date).days
        if FRONT_DTE_MIN <= dte <= FRONT_DTE_MAX and front is None:
            front = exp
        elif BACK_DTE_MIN <= dte <= BACK_DTE_MAX and back is None:
            back = exp
    return front, back


def atm_strike(chains: dict, ticker: str, expiry) -> float | None:
    """ATM = where |call_mid - put_mid| is smallest (put-call parity)."""
    td = chains.get(ticker, {}).get(expiry, {})
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


def straddle_mid(chains: dict, ticker: str, expiry, strike: float) -> float | None:
    """Call + put mid at nearest available strike within ATM_TOL_PCT of `strike`."""
    td = chains.get(ticker, {}).get(expiry, {})
    if not td:
        return None
    tol = max(0.76, ATM_TOL_PCT * strike)
    call_px = put_px = None
    best_c = best_p = tol
    for (s, cp), px in td.items():
        d = abs(s - strike)
        cp0 = (cp or "").upper()[:1]
        m = px.get("mid")
        if m is None:
            continue
        if cp0 == "C" and d < best_c:
            best_c, call_px = d, m
        elif cp0 == "P" and d < best_p:
            best_p, put_px = d, m
    return (call_px + put_px) if (call_px is not None and put_px is not None) else None


def iv_for_call(chains: dict, ticker: str, expiry, strike: float) -> float | None:
    """IV of nearest call within ATM_TOL_PCT of `strike`."""
    td = chains.get(ticker, {}).get(expiry, {})
    tol = max(0.76, ATM_TOL_PCT * strike)
    best_iv, best_d = None, tol
    for (s, cp), px in td.items():
        cp0 = (cp or "").upper()[:1]
        if cp0 != "C":
            continue
        iv = px.get("iv")
        if iv is None or iv <= 0:
            continue
        d = abs(s - strike)
        if d < best_d:
            best_d, best_iv = d, iv
    return best_iv


def calendar_ff_on_date(chains: dict, ticker: str, front_exp, back_exp,
                        strike: float, obs_date: date) -> float | None:
    """Campasano FF using current IVs on obs_date."""
    fiv = iv_for_call(chains, ticker, front_exp, strike)
    biv = iv_for_call(chains, ticker, back_exp,  strike)
    if fiv is None or biv is None:
        return None
    fd = (front_exp - obs_date).days
    bd = (back_exp  - obs_date).days
    if fd <= 0 or bd <= fd:
        return None
    fwd = forward_vol(fiv, fd / 365, biv, bd / 365)
    return compute_ff(fiv, fwd)


def calendar_mid_from_chain(chains: dict, ticker: str, front_exp, back_exp,
                             strike: float) -> float | None:
    """Calendar mid = back straddle - front straddle."""
    bm = straddle_mid(chains, ticker, back_exp,  strike)
    fm = straddle_mid(chains, ticker, front_exp, strike)
    if bm is None or fm is None:
        return None
    return bm - fm


def calendar_mid_at_expiry(chains: dict, c_stocks, ticker: str,
                            front_exp, back_exp, strike: float) -> float | None:
    """
    Calendar value when front has expired: back_straddle - |spot - strike|.
    Spot from stocks.ohlcv; back straddle from chain on front_exp date.
    """
    spot = get_spot_close(c_stocks, ticker, front_exp)
    if spot is None:
        return None
    bm = straddle_mid(chains, ticker, back_exp, strike)
    if bm is None:
        return None
    return bm - abs(spot - strike)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   default="2020-01-01")
    parser.add_argument("--end",     default="2025-12-31")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date   = date.fromisoformat(args.end)
    tickers    = args.tickers

    print(f"FF Full Backtest  {start_date} → {end_date}")
    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")

    c_opt  = earningsvol_conn()
    c_stk  = earningsvol_conn()

    print("\nFetching available Dolt dates ...")
    dolt_dates = get_all_dolt_dates(c_opt, start_date, end_date)
    print(f"  {len(dolt_dates)} dates  ({dolt_dates[0]} .. {dolt_dates[-1]})")

    # ── Streaming backtest ────────────────────────────────────────────────────
    # live_trades: {trade_id: trade_dict}
    # Laddering is intentional: multiple entries for same (ticker, front_exp, back_exp)
    # on different days are kept separate (different entry timing = different risk profile).
    live_trades:  dict[str, dict] = {}
    all_trades:   list[dict] = []
    all_paths:    list[dict] = []
    trade_counter = 0

    total_dates = len(dolt_dates)
    for di, obs_date in enumerate(dolt_dates):
        if (di + 1) % 20 == 0:
            print(f"  [{di+1}/{total_dates}]  {obs_date}  live={len(live_trades)}", flush=True)

        chains = fetch_chains(c_opt, obs_date, tickers)

        # 1. OPEN new trades ───────────────────────────────────────────────────
        for tk in tickers:
            fe, be = find_expirations(chains, tk, obs_date)
            if fe is None or be is None:
                continue

            sk = atm_strike(chains, tk, be)  # ATM from back expiry
            if sk is None:
                continue

            back_m  = straddle_mid(chains, tk, be, sk)
            front_m = straddle_mid(chains, tk, fe, sk)
            if back_m is None or front_m is None:
                continue
            entry_debit = back_m - front_m
            if entry_debit <= 0.05:
                continue

            t_front = (fe - obs_date).days
            t_back  = (be - obs_date).days
            t_fwd   = t_back - t_front
            ff_val  = calendar_ff_on_date(chains, tk, fe, be, sk, obs_date)
            max_theo   = back_m * math.sqrt(t_fwd / t_back) if t_back > 0 else None
            max_profit = (max_theo - entry_debit) if max_theo is not None else None

            tid = f"T{trade_counter:06d}"
            trade_counter += 1
            live_trades[tid] = {
                "trade_id":    tid,
                "entry_date":  obs_date.isoformat(),
                "ticker":      tk,
                "front_exp":   fe.isoformat(),
                "back_exp":    be.isoformat(),
                "strike":      sk,
                "entry_debit": round(entry_debit, 4),
                "back_straddle_entry": round(back_m, 4),
                "t_front_entry": t_front,
                "t_back_entry":  t_back,
                "t_fwd_entry":   t_fwd,
                "entry_ff":    round(ff_val, 4) if ff_val is not None else None,
                "max_theo":    round(max_theo, 4) if max_theo is not None else None,
                "max_profit":  round(max_profit, 4) if max_profit is not None else None,
                # filled at close:
                "exit_date":    None,
                "exit_reason":  None,   # "chain_roll", "expiry", "end_of_data"
                "exit_cal_val": None,
                "hold_mid_ret": None,
                "n_path_points": 0,
                # internal
                "_fe": fe,
                "_be": be,
            }

        # 2. UPDATE paths + CLOSE trades ──────────────────────────────────────
        to_close = []
        for tid, tr in live_trades.items():
            fe = tr["_fe"]
            be = tr["_be"]
            tk = tr["ticker"]
            sk = tr["strike"]
            ed = date.fromisoformat(tr["entry_date"])
            entry_debit = tr["entry_debit"]
            mp  = tr["max_profit"]

            if obs_date <= ed:
                continue  # don't add path point on entry date itself

            # Check if front_exp has expired (obs_date >= front_exp)
            if obs_date >= fe:
                # Try expiry exit: back straddle from chain + spot from stocks.ohlcv
                expiry_val = calendar_mid_at_expiry(chains, c_stk, tk, fe, be, sk)
                if expiry_val is not None:
                    ret = (expiry_val - entry_debit - IB_COMM) / entry_debit * 100
                    tr["exit_date"]    = obs_date.isoformat()
                    tr["exit_reason"]  = "expiry"
                    tr["exit_cal_val"] = round(expiry_val, 4)
                    tr["hold_mid_ret"] = round(ret, 4)
                    all_paths.append({
                        "trade_id": tid, "obs_date": obs_date.isoformat(),
                        "cal_val": round(expiry_val, 4), "current_ff": None,
                        "achieved_frac": round((expiry_val - entry_debit) / mp, 4) if mp and mp > 0 else None,
                        "days_in_trade": (obs_date - ed).days,
                        "is_expiry": 1,
                    })
                    tr["n_path_points"] += 1
                to_close.append(tid)
                continue

            # Pre-expiry: track calendar mid on current chain
            fe_in_chain = fe in chains.get(tk, {})
            be_in_chain = be in chains.get(tk, {})

            if not fe_in_chain or not be_in_chain:
                # Chain has rolled — close at last path point (already recorded)
                tr["exit_reason"] = "chain_roll"
                tr["exit_date"]   = obs_date.isoformat()
                to_close.append(tid)
                continue

            cal_val = calendar_mid_from_chain(chains, tk, fe, be, sk)
            ff_now  = calendar_ff_on_date(chains, tk, fe, be, sk, obs_date)

            if cal_val is not None:
                af = (cal_val - entry_debit) / mp if mp and mp > 0 else None
                all_paths.append({
                    "trade_id": tid, "obs_date": obs_date.isoformat(),
                    "cal_val": round(cal_val, 4),
                    "current_ff": round(ff_now, 4) if ff_now is not None else None,
                    "achieved_frac": round(af, 4) if af is not None else None,
                    "days_in_trade": (obs_date - ed).days,
                    "is_expiry": 0,
                })
                tr["n_path_points"] += 1
                # Update exit_cal_val with most recent value (for chain_roll exit)
                ret = (cal_val - entry_debit - IB_COMM) / entry_debit * 100
                tr["exit_cal_val"] = round(cal_val, 4)
                tr["hold_mid_ret"] = round(ret, 4)

        for tid in to_close:
            tr = live_trades.pop(tid)
            tr.pop("_fe", None)
            tr.pop("_be", None)
            all_trades.append(tr)

    # Close remaining open trades at end of data
    for tid, tr in live_trades.items():
        tr["exit_reason"] = "end_of_data"
        tr.pop("_fe", None)
        tr.pop("_be", None)
        tr.pop("_key", None)
        all_trades.append(tr)

    c_opt.close()
    c_stk.close()

    print(f"\nTotal trades: {len(all_trades)}")
    print(f"Total path rows: {len(all_paths)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    by_year = defaultdict(list)
    for tr in all_trades:
        by_year[tr["entry_date"][:4]].append(tr)

    print(f"\n{'Year':>6}{'trades':>8}{'with_exit':>10}{'FF>0%':>8}{'FF>16%':>9}"
          f"{'exit_reason: expiry/roll/eod':>32}")
    for y in sorted(by_year):
        yt = by_year[y]
        we   = [t for t in yt if t["hold_mid_ret"] is not None]
        ff0  = [t for t in yt if t["entry_ff"] is not None and t["entry_ff"] > 0]
        ff16 = [t for t in yt if t["entry_ff"] is not None and t["entry_ff"] > 16]
        exp  = sum(1 for t in yt if t["exit_reason"] == "expiry")
        roll = sum(1 for t in yt if t["exit_reason"] == "chain_roll")
        eod  = sum(1 for t in yt if t["exit_reason"] == "end_of_data")
        print(f"{y:>6}{len(yt):>8}{len(we):>10}{len(ff0)/max(len(yt),1)*100:>7.0f}%"
              f"{len(ff16)/max(len(yt),1)*100:>8.0f}%"
              f"   {exp:>4} / {roll:>4} / {eod:>4}")

    # FF distribution
    ff_vals = [t["entry_ff"] for t in all_trades if t["entry_ff"] is not None]
    if ff_vals:
        import numpy as np
        ffa = np.array(ff_vals)
        print(f"\nFF distribution (n={len(ffa)}): "
              f"p10={np.percentile(ffa,10):.1f}  p25={np.percentile(ffa,25):.1f}  "
              f"med={np.median(ffa):.1f}  p75={np.percentile(ffa,75):.1f}  "
              f"p90={np.percentile(ffa,90):.1f}")
        for thresh in [0, 8, 12, 16, 20]:
            pct = (ffa > thresh).mean() * 100
            print(f"  FF > {thresh:>2}%: {(ffa > thresh).sum():>5} trades ({pct:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    trades_path = OUT_DIR / "ff_all_trades.csv"
    paths_path  = OUT_DIR / "ff_daily_paths.csv"

    if all_trades:
        fields = list(all_trades[0].keys())
        with open(trades_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_trades)
        print(f"\nSaved {len(all_trades)} trades → {trades_path.name}")

    if all_paths:
        fields_p = list(all_paths[0].keys())
        with open(paths_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields_p)
            w.writeheader()
            w.writerows(all_paths)
        print(f"Saved {len(all_paths)} path rows → {paths_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
