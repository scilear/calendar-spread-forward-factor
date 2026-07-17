#!/usr/bin/env python3
"""
ff_backtest.py — T6b: historical backtest to validate composite_score components.

Default: every Monday 2022–2025 (~208 scan dates). Weekly cadence gives 4x more data
than monthly and lets you see whether the signal is persistent (same calendar scoring
high across consecutive weeks = genuine edge, not noise). Same (ticker, front_expiry)
appearing in multiple weeks = correlated observations, but that's informative: it tests
whether entry timing within a calendar window matters (the ff_quality hypothesis).

For strictly non-overlapping inference, filter trades.csv to first scan_date per
(ticker, front_expiry) in post-processing.

Two cost models:
  mid  : (back_mid − front_mid) at entry/exit  — Campasano basis (mid-to-mid)
  cross: back_ask − front_bid at entry, back_bid − front_ask at exit — live fills

In-sample: 2022–2023 | Out-of-sample: 2024–2025 (CLAUDE.md OOS requirement)

Scope: single ATM straddle calendars only (uniform structure to isolate FF/scoring edge).

Phases:
  --scan-only    Run historical scans only (slow, ~6-8h overnight; saves to backtest/scans/)
  --price-only   Price trades from cached scans (fast, ~10 min)
  (default)      Run both phases

Usage:
  python3 ff_backtest.py                                         # weekly (default)
  python3 ff_backtest.py --monthly                               # monthly only (~48 dates, ~1.5h)
  python3 ff_backtest.py --scan-only
  python3 ff_backtest.py --price-only
  python3 ff_backtest.py --start 2024-01-01 --end 2024-12-31
  python3 ff_backtest.py --force-rescan
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import sys as _sys
from pathlib import Path as _P
_d = _P(__file__).resolve().parent
while not (_d / 'db.py').exists() and _d.parent != _d:
    _d = _d.parent
if str(_d) not in _sys.path:
    _sys.path.insert(0, str(_d))
from db import earningsvol_conn

sys.path.insert(0, str(Path(__file__).parent))
from ff_universe_scan import run_scan, rank_candidates  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

BACKTEST_DIR   = Path(__file__).parent / "backtest"
SCAN_CACHE_DIR = BACKTEST_DIR / "scans"
RESULTS_DIR    = BACKTEST_DIR / "results"

FF_MIN        = 15.0
EXIT_DTE      = 15       # exit when front has ~this DTE remaining
EXIT_TOL_DAYS = 7        # accept exit date within ±7 cal days of ideal exit
IS_CUTOFF     = date(2024, 1, 1)
PRICE_CHUNK   = 80       # tickers per option-chain batch query


# ── Trading date helpers ──────────────────────────────────────────────────────

def get_trading_dates_range(start: date, end: date) -> list[date]:
    """Return Mon-Fri dates in [start, end]. Holidays produce empty DB results
    and are silently skipped downstream."""
    from datetime import timedelta
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def every_monday(start: date, end: date, td_set: set) -> list[date]:
    """Every Monday that is a trading day in [start, end]."""
    result = []
    d = start
    # advance to first Monday
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d <= end:
        if d in td_set:
            result.append(d)
        d += timedelta(days=7)
    return result


def first_monday_of_months(start: date, end: date, td_set: set) -> list[date]:
    """First trading Monday of each calendar month in [start, end]."""
    result = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        for offset in range(14):
            d = date(y, m, 1) + timedelta(days=offset)
            if d.month != m:
                break
            if d.weekday() == 0 and d in td_set and start <= d <= end:
                result.append(d)
                break
        if m == 12:
            m, y = 1, y + 1
        else:
            m += 1
    return result


def find_exit_date(front_expiry: date, td_sorted: list[date], min_date: date | None = None) -> date | None:
    """
    Exit when front has ~EXIT_DTE days remaining.
    Ideal exit = front_expiry - EXIT_DTE (calendar days).
    Accept nearest trading date within EXIT_TOL_DAYS that is strictly after min_date (entry).
    """
    ideal = front_expiry - timedelta(days=EXIT_DTE)
    best, best_diff = None, float("inf")
    for d in td_sorted:
        if d >= front_expiry:
            break
        if min_date and d <= min_date:  # exit must be after entry
            continue
        diff = abs((d - ideal).days)
        if diff < best_diff:
            best_diff = diff
            best = d
    return best if best_diff <= EXIT_TOL_DAYS else None


# ── Phase 1: historical scans ─────────────────────────────────────────────────

def _extract_candidates(scan_result: dict, ff_min: float) -> list[dict]:
    """Same extraction logic as ff_universe_scan.write_outputs()."""
    out = []
    for ticker, data in scan_result.items():
        stats = data.get("stats", {})
        current_ff = stats.get("current_ff")
        if current_ff is None or current_ff < ff_min:
            continue
        series = data.get("series", [])
        latest = series[-1] if series else {}
        out.append({
            "ticker":              ticker,
            "ff_pct":              current_ff,
            "ff_5d_ago":           stats.get("ff_5d_ago"),
            "ff_10d_ago":          stats.get("ff_10d_ago"),
            "trend":               stats.get("trend"),
            "trend_slope_5d":      stats.get("trend_slope_5d"),
            "days_above_thresh":   stats.get("days_above_thresh"),
            "consec_above_thresh": stats.get("consec_above_thresh"),
            "n_observations":      stats.get("n_observations"),
            "front_expiry":        latest.get("front_exp"),
            "back_expiry":         latest.get("back_exp"),
            "front_dte":           latest.get("front_dte"),
            "back_dte":            latest.get("back_dte"),
            "front_iv":            latest.get("front_iv"),
        })
    return out


def _cache_is_complete(path: Path) -> bool:
    """True if the cache file exists and represents a completed scan (not a failed one).
    Failed scans write 'null'; completed scans (even with 0 candidates) write '[]' or a JSON array.
    """
    if not path.exists():
        return False
    try:
        content = path.read_text().strip()
        return content != "null" and content != ""
    except OSError:
        return False


def run_scan_phase(scan_dates: list[date], ff_min: float, force: bool = False):
    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = 0
    for i, sd in enumerate(scan_dates):
        cache_file = SCAN_CACHE_DIR / f"{sd}_candidates.json"
        if not force and _cache_is_complete(cache_file):
            try:
                n = len(json.loads(cache_file.read_text()))
            except Exception:
                n = "?"
            print(f"[{i+1}/{len(scan_dates)}] {sd} — cached ({n} candidates)")
            cached += 1
            continue

        print(f"\n[{i+1}/{len(scan_dates)}] Scanning {sd} ...")
        try:
            scan_result = run_scan(sd, n_days=20, ff_min=ff_min)
        except Exception as e:
            print(f"  ERROR: {e} — writing null sentinel (will retry next run)")
            cache_file.write_text("null")
            continue

        if not scan_result:
            print(f"  No data for this date")
            cache_file.write_text("[]")
            continue

        candidates = _extract_candidates(scan_result, ff_min)
        print(f"  {len(candidates)} candidates at FF≥{ff_min}%. Ranking ...", flush=True)
        try:
            candidates = rank_candidates(candidates, scan_result, sd)
        except Exception as e:
            print(f"  WARNING ranking failed ({e}), saving unranked")

        # Earnings coverage check (advisor note: missing data silently gives 20 pts)
        n_earn = sum(1 for c in candidates if c.get("next_earnings") or c.get("days_to_earnings") is not None)
        earn_cov = n_earn / len(candidates) * 100 if candidates else 0
        print(f"  Earnings data coverage: {earn_cov:.0f}% of candidates")

        with open(cache_file, "w") as f:
            json.dump(candidates, f, default=str)
        print(f"  Saved → {cache_file.name}")

    print(f"\nScan phase done. {cached}/{len(scan_dates)} from cache.")


# ── Phase 2: pricing ──────────────────────────────────────────────────────────

def _load_cached_candidates(scan_dates: list[date]) -> list[dict]:
    all_cands = []
    for sd in scan_dates:
        cache_file = SCAN_CACHE_DIR / f"{sd}_candidates.json"
        if not cache_file.exists():
            print(f"WARNING: missing cache {sd}")
            continue
        with open(cache_file) as f:
            cands = json.load(f)
        if not cands:  # null sentinel (scan failed) or empty list (no candidates)
            if cands is None:
                print(f"  {sd}: null sentinel — scan failed, skipping (re-run --scan-only)")
            continue
        for c in cands:
            c["scan_date"] = sd.isoformat()
        all_cands.extend(cands)
    print(f"Loaded {len(all_cands)} candidates from {len(scan_dates)} scan dates")
    return all_cands


def _fetch_option_prices(opt_conn, query_date: date, tickers: list[str]) -> dict:
    """
    Returns {ticker: {expiry_date: {(strike_float, call_put): {bid,ask,mid,iv}}}}
    Batched to avoid large IN clauses.
    """
    result: dict = defaultdict(lambda: defaultdict(dict))
    if not tickers:
        return result
    for i in range(0, len(tickers), PRICE_CHUNK):
        chunk = tickers[i : i + PRICE_CHUNK]
        ph = ",".join(["%s"] * len(chunk))
        with opt_conn.cursor() as cur:
            cur.execute(
                f"SELECT act_symbol, expiration, strike, call_put, bid, ask, vol "
                f"FROM option_chain WHERE date=%s AND act_symbol IN ({ph})",
                [query_date.isoformat()] + chunk,
            )
            for row in cur.fetchall():
                tkr, exp, strike_raw, cp, bid_, ask_, vol_ = row
                if strike_raw is None:
                    continue
                s = round(float(strike_raw), 2)
                b = float(bid_) if bid_ is not None else None
                a = float(ask_) if ask_ is not None else None
                m = (b + a) / 2 if (b is not None and a is not None) else None
                v = float(vol_) if vol_ is not None else None
                result[tkr][exp][(s, cp)] = {"bid": b, "ask": a, "mid": m, "iv": v}
    return result


def _get_put(price_data: dict, ticker: str, expiry, strike: float, side: str):
    """Get nearest ATM put price within 0.76 of requested strike."""
    td = price_data.get(ticker, {}).get(expiry, {})
    if not td:
        return None, None
    best_px = best_iv = None
    best_diff = 0.76
    for (s, cp), px in td.items():
        d = abs(s - strike)
        cp0 = (cp or "").upper()[:1]
        if cp0 == "P" and d < best_diff:
            best_diff = d
            best_px = px.get(side)
            best_iv = px.get("iv")
    return best_px, best_iv


def _get_straddle(price_data: dict, ticker: str, expiry, strike: float, side: str):
    """
    Sum call+put price at nearest strike within 0.76 of requested.
    side: 'bid', 'ask', or 'mid'.
    Returns (straddle_price, iv) — iv from call leg.
    """
    td = price_data.get(ticker, {}).get(expiry, {})
    if not td:
        return None, None

    call_px = call_iv = put_px = None
    best_diff = 0.76

    for (s, cp), px in td.items():
        d = abs(s - strike)
        if d > best_diff:
            continue
        cp0 = (cp or "").upper()[:1]
        if cp0 == "C" and call_px is None:
            call_px = px.get(side)
            call_iv = px.get("iv")
        elif cp0 == "P" and put_px is None:
            put_px = px.get(side)

    if call_px is None or put_px is None:
        return None, call_iv
    return call_px + put_px, call_iv


def _price_trade(entry_data, exit_data, ticker, front_exp, back_exp, strike):
    """Compute straddle calendar and ATM put calendar pricing."""
    def safe(a, b): return round(a - b, 4) if (a is not None and b is not None) else None
    def pct(pnl, cost): return round(pnl / cost * 100, 2) if (pnl is not None and cost and cost > 0.01) else None

    # ── Straddle calendar ─────────────────────────────────────────────────────
    front_bid, f_iv_e  = _get_straddle(entry_data, ticker, front_exp, strike, "bid")
    back_ask,  b_iv_e  = _get_straddle(entry_data, ticker, back_exp,  strike, "ask")
    front_mid_e, _     = _get_straddle(entry_data, ticker, front_exp, strike, "mid")
    back_mid_e,  _     = _get_straddle(entry_data, ticker, back_exp,  strike, "mid")
    front_ask_x, f_iv_x = _get_straddle(exit_data, ticker, front_exp, strike, "ask")
    back_bid_x,  b_iv_x = _get_straddle(exit_data, ticker, back_exp,  strike, "bid")
    front_mid_x, _     = _get_straddle(exit_data, ticker, front_exp, strike, "mid")
    back_mid_x,  _     = _get_straddle(exit_data, ticker, back_exp,  strike, "mid")

    st_entry_cross = safe(back_ask,   front_bid)
    st_entry_mid   = safe(back_mid_e, front_mid_e)
    st_exit_cross  = safe(back_bid_x, front_ask_x)
    st_exit_mid    = safe(back_mid_x, front_mid_x)
    st_pnl_cross   = safe(st_exit_cross, st_entry_cross)
    st_pnl_mid     = safe(st_exit_mid,   st_entry_mid)

    # ── ATM put calendar ──────────────────────────────────────────────────────
    fp_bid_e,  fp_iv_e  = _get_put(entry_data, ticker, front_exp, strike, "bid")
    bp_ask_e,  bp_iv_e  = _get_put(entry_data, ticker, back_exp,  strike, "ask")
    fp_mid_e,  _        = _get_put(entry_data, ticker, front_exp, strike, "mid")
    bp_mid_e,  _        = _get_put(entry_data, ticker, back_exp,  strike, "mid")
    fp_ask_x,  fp_iv_x  = _get_put(exit_data,  ticker, front_exp, strike, "ask")
    bp_bid_x,  bp_iv_x  = _get_put(exit_data,  ticker, back_exp,  strike, "bid")
    fp_mid_x,  _        = _get_put(exit_data,  ticker, front_exp, strike, "mid")
    bp_mid_x,  _        = _get_put(exit_data,  ticker, back_exp,  strike, "mid")

    put_entry_cross = safe(bp_ask_e,  fp_bid_e)
    put_entry_mid   = safe(bp_mid_e,  fp_mid_e)
    put_exit_cross  = safe(bp_bid_x,  fp_ask_x)
    put_exit_mid    = safe(bp_mid_x,  fp_mid_x)
    put_pnl_cross   = safe(put_exit_cross, put_entry_cross)
    put_pnl_mid     = safe(put_exit_mid,   put_entry_mid)

    return {
        # Straddle calendar
        "entry_cross_debit":    st_entry_cross,
        "entry_mid_debit":      st_entry_mid,
        "exit_cross_value":     st_exit_cross,
        "exit_mid_value":       st_exit_mid,
        "pnl_cross":            st_pnl_cross,
        "pnl_mid":              st_pnl_mid,
        "pnl_pct_cross":        pct(st_pnl_cross, st_entry_cross),
        "pnl_pct_mid":          pct(st_pnl_mid,   st_entry_mid),
        "front_iv_entry":       round(f_iv_e,  4) if f_iv_e  else None,
        "back_iv_entry":        round(b_iv_e,  4) if b_iv_e  else None,
        "front_iv_exit":        round(f_iv_x,  4) if f_iv_x  else None,
        "back_iv_exit":         round(b_iv_x,  4) if b_iv_x  else None,
        # ATM put calendar
        "put_entry_cross_debit": put_entry_cross,
        "put_entry_mid_debit":   put_entry_mid,
        "put_exit_cross_value":  put_exit_cross,
        "put_exit_mid_value":    put_exit_mid,
        "put_pnl_cross":         put_pnl_cross,
        "put_pnl_mid":           put_pnl_mid,
        "put_pnl_pct_cross":     pct(put_pnl_cross, put_entry_cross),
        "put_pnl_pct_mid":       pct(put_pnl_mid,   put_entry_mid),
        "front_put_iv_entry":    round(fp_iv_e, 4) if fp_iv_e else None,
        "back_put_iv_entry":     round(bp_iv_e, 4) if bp_iv_e else None,
        "front_put_iv_exit":     round(fp_iv_x, 4) if fp_iv_x else None,
        "back_put_iv_exit":      round(bp_iv_x, 4) if bp_iv_x else None,
    }


def run_price_phase(scan_dates: list[date], td_sorted: list[date]) -> list[dict]:
    all_cands = _load_cached_candidates(scan_dates)

    # Parse dates; compute exit dates
    valid, skipped = [], 0
    for c in all_cands:
        sd_str = c.get("scan_date", "")
        fe_str = c.get("front_expiry", "")
        be_str = c.get("back_expiry", "")
        atm    = c.get("atm_strike")
        try:
            c["_sd"]  = datetime.strptime(sd_str, "%Y-%m-%d").date()
            c["_fe"]  = datetime.strptime(fe_str, "%Y-%m-%d").date()
            c["_be"]  = datetime.strptime(be_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            skipped += 1
            continue
        if atm is None:
            skipped += 1
            continue
        c["_atm"] = float(atm)
        c["_ed"]  = find_exit_date(c["_fe"], td_sorted, min_date=c["_sd"])
        if c["_ed"] is None:
            skipped += 1
            continue
        valid.append(c)

    print(f"Valid for pricing: {len(valid)} | Skipped (no exit/strike): {skipped}")

    # Batch-fetch entry prices grouped by scan_date
    by_sd: dict[date, list] = defaultdict(list)
    for c in valid:
        by_sd[c["_sd"]].append(c)

    by_ed: dict[date, list] = defaultdict(list)
    for c in valid:
        by_ed[c["_ed"]].append(c)

    opt_conn = earningsvol_conn()

    print(f"\nFetching entry prices ({len(by_sd)} dates) ...")
    entry_px: dict[date, dict] = {}
    for sd in sorted(by_sd):
        tickers = list({c["ticker"] for c in by_sd[sd]})
        entry_px[sd] = _fetch_option_prices(opt_conn, sd, tickers)
        print(f"  {sd}: {len(tickers)} tickers")

    print(f"\nFetching exit prices ({len(by_ed)} dates) ...")
    exit_px: dict[date, dict] = {}
    for ed in sorted(by_ed):
        tickers = list({c["ticker"] for c in by_ed[ed]})
        exit_px[ed] = _fetch_option_prices(opt_conn, ed, tickers)
        print(f"  {ed}: {len(tickers)} tickers")

    opt_conn.close()

    # Price each trade
    trades = []
    for c in valid:
        sd, fe, be, ed, strike = c["_sd"], c["_fe"], c["_be"], c["_ed"], c["_atm"]
        pricing = _price_trade(
            entry_px.get(sd, {}), exit_px.get(ed, {}),
            c["ticker"], fe, be, strike,
        )
        trades.append({
            "scan_date":       sd.isoformat(),
            "period":          "OOS" if sd >= IS_CUTOFF else "IS",
            "ticker":          c["ticker"],
            "composite_score": c.get("composite_score"),
            "ff_pct":          c.get("ff_pct"),
            "ff_quality_pts":  c.get("ff_quality_pts"),
            "earn_pts":        c.get("earn_pts"),
            "ivr_pts":         c.get("ivr_pts"),
            "ivhv_pts":        c.get("ivhv_pts"),
            "trend_pts":       c.get("trend_pts"),
            "iv_hv_ratio":     c.get("iv_hv_ratio"),
            "iv_rank_20d":     c.get("iv_rank_20d"),
            "trend_strength":  c.get("trend_strength"),
            "earnings_risk":   c.get("earnings_risk"),
            "days_to_earnings": c.get("days_to_earnings"),
            "front_dte":       c.get("front_dte"),
            "back_dte":        c.get("back_dte"),
            "front_expiry":    fe.isoformat(),
            "back_expiry":     be.isoformat(),
            "atm_strike":      strike,
            "exit_date":       ed.isoformat(),
            "days_held":       (ed - sd).days,
            **pricing,
        })

    n_priced = sum(1 for t in trades if t.get("pnl_pct_mid") is not None)
    drop_rate = round((len(trades) - n_priced) / max(len(trades), 1) * 100, 1)
    print(f"\nPriced: {n_priced}/{len(trades)} | Drop rate: {drop_rate}%")
    if drop_rate > 25:
        print(f"  WARNING: high drop rate — check Dolt option chain coverage for exit dates")

    return trades


# ── Analysis ──────────────────────────────────────────────────────────────────

def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "win_rate": None, "avg": None, "median": None,
                "max_loss": None, "max_gain": None, "std": None}
    n = len(vals)
    s = sorted(vals)
    wins = sum(1 for v in vals if v > 0)
    mean = sum(vals) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
    return {
        "n":        n,
        "win_rate": round(wins / n * 100, 1),
        "avg":      round(mean, 2),
        "median":   round(s[n // 2], 2),
        "max_loss": round(min(vals), 2),
        "max_gain": round(max(vals), 2),
        "std":      round(std, 2),
    }


def summarize(trades: list[dict]) -> list[dict]:
    score_tiers = [
        ("0–40",  lambda t: (t.get("composite_score") or 0) < 40),
        ("40–60", lambda t: 40 <= (t.get("composite_score") or 0) < 60),
        ("60–80", lambda t: 60 <= (t.get("composite_score") or 0) < 80),
        ("80+",   lambda t: (t.get("composite_score") or 0) >= 80),
    ]
    ff_tiers = [
        ("0–5",  lambda t: (t.get("ff_quality_pts") or 0) < 5),
        ("5–8",  lambda t: 5 <= (t.get("ff_quality_pts") or 0) < 8),
        ("8–10", lambda t: 8 <= (t.get("ff_quality_pts") or 0) < 10),
        ("10",   lambda t: (t.get("ff_quality_pts") or 0) >= 10),
    ]

    rows = []
    for period_label, period_fn in [
        ("ALL", lambda t: True),
        ("IS",  lambda t: t.get("period") == "IS"),
        ("OOS", lambda t: t.get("period") == "OOS"),
    ]:
        pt = [t for t in trades if period_fn(t)]
        for btype, tiers in [("composite_score", score_tiers), ("ff_quality_pts", ff_tiers)]:
            for blabel, bfn in tiers:
                bucket   = [t for t in pt if bfn(t)]
                mid_p    = [t["pnl_pct_mid"]      for t in bucket if t.get("pnl_pct_mid")      is not None]
                cross_p  = [t["pnl_pct_cross"]    for t in bucket if t.get("pnl_pct_cross")    is not None]
                put_mid  = [t["put_pnl_pct_mid"]  for t in bucket if t.get("put_pnl_pct_mid")  is not None]
                put_cross= [t["put_pnl_pct_cross"] for t in bucket if t.get("put_pnl_pct_cross") is not None]
                ms, cs, ps, pcs = _stats(mid_p), _stats(cross_p), _stats(put_mid), _stats(put_cross)
                rows.append({
                    "period":            period_label,
                    "bucket_type":       btype,
                    "bucket":            blabel,
                    "n_total":           len(bucket),
                    # straddle
                    "mid_n":             ms["n"],
                    "mid_win_rate":      ms["win_rate"],
                    "mid_avg":           ms["avg"],
                    "mid_median":        ms["median"],
                    "mid_max_loss":      ms["max_loss"],
                    "mid_max_gain":      ms["max_gain"],
                    "mid_std":           ms["std"],
                    "cross_n":           cs["n"],
                    "cross_win_rate":    cs["win_rate"],
                    "cross_avg":         cs["avg"],
                    "cross_median":      cs["median"],
                    "cross_max_loss":    cs["max_loss"],
                    "cross_max_gain":    cs["max_gain"],
                    # put calendar
                    "put_mid_n":         ps["n"],
                    "put_mid_win_rate":  ps["win_rate"],
                    "put_mid_avg":       ps["avg"],
                    "put_mid_median":    ps["median"],
                    "put_mid_max_loss":  ps["max_loss"],
                    "put_mid_max_gain":  ps["max_gain"],
                    "put_mid_std":       ps["std"],
                    "put_cross_n":       pcs["n"],
                    "put_cross_win_rate":pcs["win_rate"],
                    "put_cross_avg":     pcs["avg"],
                    "put_cross_median":  pcs["median"],
                    "put_cross_max_loss":pcs["max_loss"],
                    "put_cross_max_gain":pcs["max_gain"],
                })
    return rows


# ── Output ────────────────────────────────────────────────────────────────────

TRADE_FIELDS = [
    "scan_date", "period", "ticker",
    "composite_score", "ff_pct", "ff_quality_pts",
    "earn_pts", "ivr_pts", "ivhv_pts", "trend_pts",
    "iv_hv_ratio", "iv_rank_20d", "trend_strength",
    "earnings_risk", "days_to_earnings",
    "front_dte", "back_dte", "front_expiry", "back_expiry", "atm_strike",
    "exit_date", "days_held",
    # Straddle calendar
    "entry_cross_debit", "entry_mid_debit",
    "exit_cross_value",  "exit_mid_value",
    "pnl_cross", "pnl_mid", "pnl_pct_cross", "pnl_pct_mid",
    "front_iv_entry", "back_iv_entry", "front_iv_exit", "back_iv_exit",
    # ATM put calendar
    "put_entry_cross_debit", "put_entry_mid_debit",
    "put_exit_cross_value",  "put_exit_mid_value",
    "put_pnl_cross", "put_pnl_mid", "put_pnl_pct_cross", "put_pnl_pct_mid",
    "front_put_iv_entry", "back_put_iv_entry", "front_put_iv_exit", "back_put_iv_exit",
]

SUMM_FIELDS = [
    "period", "bucket_type", "bucket", "n_total",
    "mid_n", "mid_win_rate", "mid_avg", "mid_median", "mid_max_loss", "mid_max_gain", "mid_std",
    "cross_n", "cross_win_rate", "cross_avg", "cross_median", "cross_max_loss", "cross_max_gain",
    "put_mid_n", "put_mid_win_rate", "put_mid_avg", "put_mid_median", "put_mid_max_loss", "put_mid_max_gain", "put_mid_std",
    "put_cross_n", "put_cross_win_rate", "put_cross_avg", "put_cross_median", "put_cross_max_loss", "put_cross_max_gain",
]


def _html_report(trades: list[dict], summary: list[dict], path: Path):
    score_labels = ["0–40", "40–60", "60–80", "80+"]
    ff_labels    = ["0–5",  "5–8",   "8–10",  "10"]

    def row(period, btype, bucket):
        for r in summary:
            if r["period"] == period and r["bucket_type"] == btype and r["bucket"] == bucket:
                return r
        return {}

    def js(x): return json.dumps(x)

    def extract(period, btype, labels, field):
        return [row(period, btype, b).get(field) or 0 for b in labels]

    is_wr_score   = extract("IS",  "composite_score", score_labels, "mid_win_rate")
    oos_wr_score  = extract("OOS", "composite_score", score_labels, "mid_win_rate")
    is_avg_mid    = extract("IS",  "composite_score", score_labels, "mid_avg")
    oos_avg_mid   = extract("OOS", "composite_score", score_labels, "mid_avg")
    is_avg_cross  = extract("IS",  "composite_score", score_labels, "cross_avg")
    oos_avg_cross = extract("OOS", "composite_score", score_labels, "cross_avg")
    is_wr_ff      = extract("IS",  "ff_quality_pts",  ff_labels,    "mid_win_rate")
    oos_wr_ff     = extract("OOS", "ff_quality_pts",  ff_labels,    "mid_win_rate")
    is_n_score    = extract("IS",  "composite_score", score_labels, "mid_n")
    oos_n_score   = extract("OOS", "composite_score", score_labels, "mid_n")

    all_mid   = [t["pnl_pct_mid"]   for t in trades if t.get("pnl_pct_mid")   is not None]
    all_cross = [t["pnl_pct_cross"] for t in trades if t.get("pnl_pct_cross") is not None]
    n_total   = len(trades)
    n_priced  = len(all_mid)
    drop_rate = round((n_total - n_priced) / max(n_total, 1) * 100, 1)
    n_is  = sum(1 for t in trades if t.get("period") == "IS")
    n_oos = sum(1 for t in trades if t.get("period") == "OOS")

    def wr(vals): return round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1) if vals else 0
    def avg(vals): return round(sum(vals) / len(vals), 1) if vals else 0

    # Summary table rows
    table_rows = ""
    for period in ["IS", "OOS"]:
        for bl in score_labels:
            r = row(period, "composite_score", bl)
            n  = r.get("mid_n", 0)
            wr_ = r.get("mid_win_rate") or 0
            ma  = r.get("mid_avg") or 0
            mm  = r.get("mid_median") or 0
            xwr = r.get("cross_win_rate") or 0
            xa  = r.get("cross_avg") or 0
            ml  = r.get("mid_max_loss") or 0
            pc  = "green" if ma > 0 else "tomato"
            xpc = "green" if xa > 0 else "tomato"
            table_rows += (
                f"<tr><td>{bl}</td><td>{period}</td><td>{n}</td>"
                f"<td>{wr_}%</td>"
                f"<td style='color:{pc}'>{ma:+.1f}%</td>"
                f"<td>{mm:+.1f}%</td>"
                f"<td>{xwr}%</td>"
                f"<td style='color:{xpc}'>{xa:+.1f}%</td>"
                f"<td style='color:tomato'>{ml:.1f}%</td></tr>\n"
            )

    warn_html = '<p style="color:#ffaa00">⚠ Drop rate &gt; 25% — check Dolt exit-date coverage.</p>' if drop_rate > 25 else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>FF Calendar Backtest T6b</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body{{font-family:monospace;background:#1a1a2e;color:#e0e0e0;margin:24px}}
  h1{{color:#00d4ff}}h2{{color:#aaa;border-bottom:1px solid #333;padding-bottom:4px}}
  .meta{{color:#888;font-size:.9em;margin-bottom:12px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:24px}}
  .box{{background:#16213e;border-radius:8px;padding:16px}}
  canvas{{max-height:260px}}
  table{{border-collapse:collapse;width:100%;font-size:.85em;margin-top:8px}}
  th,td{{padding:5px 10px;text-align:right;border:1px solid #333}}
  th{{background:#333;color:#aaa;text-align:center}}
  tr:nth-child(even){{background:#1f2f4f}}
</style></head><body>
<h1>FF Calendar Backtest — T6b</h1>
<div class="meta">
  {date.today()} &nbsp;|&nbsp; Candidates: {n_total} &nbsp;|&nbsp;
  Priced: {n_priced} &nbsp;|&nbsp; Drop rate: {drop_rate}% &nbsp;|&nbsp;
  IS (2022–23): {n_is} &nbsp;|&nbsp; OOS (2024–25): {n_oos}
</div>
<div class="meta">
  <b>Overall mid-to-mid:</b> Win {wr(all_mid)}% &nbsp;Avg {avg(all_mid):+.1f}% &nbsp;|&nbsp;
  <b>Overall bid/ask-cross:</b> Win {wr(all_cross)}% &nbsp;Avg {avg(all_cross):+.1f}%
</div>
{warn_html}
<div class="grid">
  <div class="box"><h2>Win Rate by Composite Score (mid)</h2><canvas id="c1"></canvas></div>
  <div class="box"><h2>Avg P&amp;L % by Score Tier</h2><canvas id="c2"></canvas></div>
  <div class="box"><h2>Win Rate by FF Quality Score</h2><canvas id="c3"></canvas></div>
  <div class="box"><h2>Mid vs Cross Gap (IS)</h2><canvas id="c4"></canvas></div>
</div>
<h2>Detail — Composite Score Tiers</h2>
<table>
<tr><th>Score</th><th>Period</th><th>N</th>
<th>Win% mid</th><th>Avg P&amp;L mid</th><th>Median mid</th>
<th>Win% cross</th><th>Avg P&amp;L cross</th><th>Max loss</th></tr>
{table_rows}
</table>
<script>
const opts = (yLabel) => ({{
  plugins:{{legend:{{labels:{{color:'#aaa'}}}}}},
  scales:{{
    y:{{title:{{display:true,text:yLabel,color:'#888'}},ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}},
    x:{{ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}}
  }}
}});
new Chart('c1',{{type:'bar',data:{{labels:{js(score_labels)},datasets:[
  {{label:'IS',data:{js(is_wr_score)},backgroundColor:'rgba(0,212,255,.7)'}},
  {{label:'OOS',data:{js(oos_wr_score)},backgroundColor:'rgba(255,140,0,.7)'}}
]}},options:{{...opts('%'),scales:{{...opts('%').scales,y:{{...opts('%').scales.y,min:0,max:100}}}}}}}});
new Chart('c2',{{type:'bar',data:{{labels:{js(score_labels)},datasets:[
  {{label:'IS mid',data:{js(is_avg_mid)},backgroundColor:'rgba(0,212,255,.7)'}},
  {{label:'OOS mid',data:{js(oos_avg_mid)},backgroundColor:'rgba(255,140,0,.7)'}},
  {{label:'IS cross',data:{js(is_avg_cross)},backgroundColor:'rgba(100,255,130,.5)'}},
  {{label:'OOS cross',data:{js(oos_avg_cross)},backgroundColor:'rgba(255,80,80,.5)'}}
]}},options:opts('Avg P&L %')}});
new Chart('c3',{{type:'bar',data:{{labels:{js(ff_labels)},datasets:[
  {{label:'IS',data:{js(is_wr_ff)},backgroundColor:'rgba(160,100,255,.7)'}},
  {{label:'OOS',data:{js(oos_wr_ff)},backgroundColor:'rgba(255,200,50,.7)'}}
]}},options:{{...opts('%'),scales:{{...opts('%').scales,y:{{...opts('%').scales.y,min:0,max:100}}}}}}}});
new Chart('c4',{{type:'bar',data:{{labels:{js(score_labels)},datasets:[
  {{label:'IS mid',data:{js(is_avg_mid)},backgroundColor:'rgba(0,212,255,.7)'}},
  {{label:'IS cross',data:{js(is_avg_cross)},backgroundColor:'rgba(255,80,80,.7)'}}
]}},options:opts('Avg P&L %')}});
</script></body></html>"""

    path.write_text(html)


def write_results(trades: list[dict], summary: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trade_path = RESULTS_DIR / "trades.csv"
    with open(trade_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(trades)

    summ_path = RESULTS_DIR / "summary.csv"
    with open(summ_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMM_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    html_path = RESULTS_DIR / "report.html"
    _html_report(trades, summary, html_path)

    print(f"\nTrades CSV : {trade_path}  ({len(trades)} rows)")
    print(f"Summary CSV: {summ_path}")
    print(f"HTML report: {html_path}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_console_summary(trades: list[dict], summary: list[dict]):
    print(f"\n{'='*90}")
    print(f"{'RESULTS — COMPOSITE SCORE TIERS':^90}")
    print(f"{'='*90}")
    hdr = (f"{'Score':<7} {'Per':<4} {'N':>4}  "
           f"{'Win%(str)':>9}  {'Avg str':>8}  "
           f"{'Win%(put)':>9}  {'Avg put':>8}  {'MaxLoss str':>11}")
    print(hdr)
    print("-" * 90)
    for period in ["IS", "OOS"]:
        for r in [r for r in summary if r["period"] == period and r["bucket_type"] == "composite_score"]:
            n    = r.get("mid_n") or 0
            wr   = r.get("mid_win_rate") or 0
            ma   = r.get("mid_avg") or 0
            pwr  = r.get("put_mid_win_rate") or 0
            pma  = r.get("put_mid_avg") or 0
            ml   = r.get("mid_max_loss") or 0
            flag = " ★" if (pma > 5 and pwr > 55) else ""
            print(f"  {r['bucket']:<7} {period:<4} {n:>4}  "
                  f"{wr:>8.1f}%  {ma:>+7.1f}%  "
                  f"{pwr:>8.1f}%  {pma:>+7.1f}%  {ml:>10.1f}%{flag}")
        print()

    print(f"\n{'FF QUALITY SCORE TIERS':^90}")
    print("-" * 90)
    for period in ["IS", "OOS"]:
        for r in [r for r in summary if r["period"] == period and r["bucket_type"] == "ff_quality_pts"]:
            n   = r.get("mid_n") or 0
            wr  = r.get("mid_win_rate") or 0
            ma  = r.get("mid_avg") or 0
            pwr = r.get("put_mid_win_rate") or 0
            pma = r.get("put_mid_avg") or 0
            print(f"  ff_q {r['bucket']:<5} {period:<4} {n:>4}  "
                  f"{wr:>8.1f}%  {ma:>+7.1f}%  {pwr:>8.1f}%  {pma:>+7.1f}%")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FF Calendar Backtest — T6b")
    parser.add_argument("--start",        default="2022-01-01")
    parser.add_argument("--end",          default="2025-12-31")
    parser.add_argument("--ff-min",       type=float, default=FF_MIN)
    parser.add_argument("--monthly",      action="store_true",
                        help="Use monthly scan dates only (~48 total, ~1.5h). Default: weekly (~208 total, ~7h).")
    parser.add_argument("--scan-only",    action="store_true")
    parser.add_argument("--price-only",   action="store_true")
    parser.add_argument("--force-rescan", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    print("Fetching trading date calendar ...")
    all_td  = get_trading_dates_range(start - timedelta(days=90), end)
    td_set  = set(all_td)

    if args.monthly:
        scan_dates = first_monday_of_months(start, end, td_set)
        freq_label = "monthly"
    else:
        scan_dates = every_monday(start, end, td_set)
        freq_label = "weekly"

    n_is  = sum(1 for d in scan_dates if d < IS_CUTOFF)
    n_oos = len(scan_dates) - n_is
    print(f"Scan dates: {len(scan_dates)} ({freq_label}) — IS: {n_is}  OOS: {n_oos}")
    if scan_dates:
        print(f"  {scan_dates[0]} → {scan_dates[-1]}")
    print(f"In-sample cutoff: {IS_CUTOFF}  |  FF min: {args.ff_min}%  |  Exit DTE target: {EXIT_DTE}")

    if not args.price_only:
        run_scan_phase(scan_dates, args.ff_min, force=args.force_rescan)

    if not args.scan_only:
        # Price phase needs td list covering full backtest period for exit date lookup
        td_price = get_trading_dates_range(start, end + timedelta(days=30))
        trades   = run_price_phase(scan_dates, td_price)
        if not trades:
            print("No trades to analyze.")
            return 1
        summary = summarize(trades)
        write_results(trades, summary)
        print_console_summary(trades, summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
