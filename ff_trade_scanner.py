#!/usr/bin/env python3
"""
FF Trade Scanner — prices double-calendar spread recommendations for FF > 16% signals.

Takes FF signal candidates from ff_scanner.py and outputs ready-to-trade recommendations
with pricing, slippage estimates, and entry readiness checks.

Trade structure: Long double calendar (straddle calendar)
    SELL front straddle (ATM call + put, ~30 DTE) → receive bid
    BUY  back  straddle (ATM call + put, ~60 DTE) → pay ask
    Net debit = back_total_ask - front_total_bid

Usage:
    # Run FF scan then price signals (typical daily workflow):
    python ff_trade_scanner.py --no-ib

    # With universe file for FF trend data + pre-filtered tickers:
    python ff_trade_scanner.py --universe universe/latest_candidates.csv --no-ib

    # Use existing scan CSV (skip re-scan):
    python ff_trade_scanner.py --scan-file daily_scans/2026-06-05_ff_scan.csv --no-ib

    # Lower FF threshold to price more candidates:
    python ff_trade_scanner.py --ff-min 10.0 --no-ib

Output:
    trade_recommendations/YYYY-MM-DD_trade_recommendations.csv
    reports/TICKER_YYYY-MM-DD_report.html  (one per candidate)
"""

import argparse
import concurrent.futures
import csv
import json
import math
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

OPTTRADER_DIR = Path.home() / "Documents" / "OptionTrader"
OPTCHAIN_SCRIPT = OPTTRADER_DIR / "tools" / "option_chain.py"
SCAN_DIR    = Path(__file__).parent / "daily_scans"
OUTPUT_DIR  = Path(__file__).parent / "trade_recommendations"
REPORTS_DIR = Path(__file__).parent / "reports"

FF_MIN_DEFAULT = 16.0
SLIPPAGE_FACTOR = 1.5        # 50% of bid-ask crossed on entry + exit
WIDE_SPREAD_THRESHOLD = 0.50 # per leg; flag if front or back total > $0.50
MIN_VOLUME_PER_LEG = 1000

# ── Ready-state history ───────────────────────────────────────────────────────
# Tracks how often each ticker actually clears the entry gate, so tickers with a
# track record of being tradable get priced (and reported) first.
READY_STATS_FILE = Path(__file__).parent / "ready_stats.json"
READY_PRIOR_RATE = 0.5   # Beta prior mean for unproven tickers
READY_PRIOR_N    = 3.0   # prior strength, in pseudo-observations
GOOD_MIN_SEEN    = 3     # scans needed before a ticker can be called "good"
GOOD_MIN_RATE    = 0.40  # ready-rate needed to be called "good"


def load_ready_stats() -> dict:
    if not READY_STATS_FILE.exists():
        return {"tickers": {}}
    try:
        data = json.loads(READY_STATS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ready_stats.json unreadable ({e}) — starting fresh", file=sys.stderr)
        return {"tickers": {}}
    data.setdefault("tickers", {})
    return data


def save_ready_stats(stats: dict, scan_date: str):
    stats["updated"] = scan_date
    try:
        READY_STATS_FILE.write_text(json.dumps(stats, indent=2, sort_keys=True))
    except OSError as e:
        print(f"  Could not write {READY_STATS_FILE}: {e}", file=sys.stderr)


def ready_rate(stats: dict, ticker: str) -> float:
    """Smoothed ready-rate. Unproven tickers get the prior, so they rank between
    proven-good and proven-bad rather than at either extreme."""
    t = stats.get("tickers", {}).get(ticker)
    seen  = (t or {}).get("seen", 0)
    ready = (t or {}).get("ready", 0)
    return (ready + READY_PRIOR_RATE * READY_PRIOR_N) / (seen + READY_PRIOR_N)


def is_good_ticker(stats: dict, ticker: str) -> bool:
    """Proven tradable: enough observations AND a real ready-rate.
    Unproven tickers are deliberately NOT good — they only earn it via full runs."""
    t = stats.get("tickers", {}).get(ticker) or {}
    seen = t.get("seen", 0)
    if seen < GOOD_MIN_SEEN:
        return False
    return t.get("ready", 0) / seen >= GOOD_MIN_RATE


def update_ready_stats(stats: dict, results: list[dict], scan_date: str):
    """Record this scan's outcome for every ticker we actually priced."""
    tickers = stats.setdefault("tickers", {})
    for rec in results:
        t = rec.get("ticker")
        if not t:
            continue
        entry = tickers.setdefault(t, {"seen": 0, "ready": 0, "last_ready": None,
                                       "last_seen": None, "consec_not_ready": 0})
        entry["seen"] = entry.get("seen", 0) + 1
        entry["last_seen"] = scan_date
        if rec.get("entry_ready"):
            entry["ready"] = entry.get("ready", 0) + 1
            entry["last_ready"] = scan_date
            entry["consec_not_ready"] = 0
        else:
            entry["consec_not_ready"] = entry.get("consec_not_ready", 0) + 1
        entry["ready_rate"] = round(entry["ready"] / entry["seen"], 3)


# ── Option chain fetching ──────────────────────────────────────────────────────

def fetch_chain_for_expiry(ticker: str, expiry: str, no_ib: bool) -> dict | None:
    """Fetch option chain for a specific expiry date."""
    cmd = [
        sys.executable, str(OPTCHAIN_SCRIPT),
        "--ticker", ticker,
        "--expiry", expiry,
        "--output", "json",
    ]
    if no_ib:
        cmd.append("--no-ib")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(OPTTRADER_DIR),
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None


def find_atm_call(rows: list[dict], spot: float) -> dict | None:
    """Return the ATM call row. Prefers rows with valid IV/delta; falls back to
    nearest-strike when yfinance returns zero IV (avoids dropping liquid tickers)."""
    all_calls = [r for r in rows if r.get("right") in ("C", "Call")]
    if not all_calls:
        return None
    # Prefer rows with IV so delta selection works; fall back to all calls
    iv_calls = [r for r in all_calls if (r.get("iv") or 0) > 0]
    pool = iv_calls or all_calls
    by_delta = [r for r in pool if r.get("delta") is not None]
    if by_delta:
        return min(by_delta, key=lambda r: abs(r["delta"] - 0.50))
    return min(pool, key=lambda r: abs((r.get("strike") or 0) - spot))


def get_row_at_strike(rows: list[dict], strike: float, right_key: str) -> dict | None:
    """Return the row closest to target strike with matching right (C or P)."""
    right_map = {"C": ("C", "Call"), "P": ("P", "Put")}
    valid = right_map.get(right_key, (right_key,))
    candidates = [r for r in rows if r.get("right") in valid]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs((r.get("strike") or 0) - strike))


def safe_float(val, default: float = 0.0) -> float:
    return float(val) if val is not None and float(val) > 0 else default


# ── Expiry selection (yfinance — lightweight, no chain fetch) ─────────────────

_FRONT_DTE_WINDOW = (22, 45)
_BACK_DTE_WINDOW  = (46, 80)

def pick_live_expiry(ticker: str) -> tuple[str | None, str | None]:
    """
    Select fresh front/back expiry dates from yfinance for a ticker.
    Uses only yf.Ticker().options (no chain fetch) — fast, ~1 API call.
    Returns (front_expiry_str, back_expiry_str) or (None, None) on failure.
    """
    try:
        import yfinance as yf
        exps = yf.Ticker(ticker).options   # tuple of YYYY-MM-DD strings
        if not exps:
            return None, None
        today = date.today()
        front_str = back_str = None
        for exp_str in sorted(exps):
            dte = (date.fromisoformat(exp_str) - today).days
            if front_str is None and _FRONT_DTE_WINDOW[0] <= dte <= _FRONT_DTE_WINDOW[1]:
                front_str = exp_str
            elif back_str is None and _BACK_DTE_WINDOW[0] <= dte <= _BACK_DTE_WINDOW[1]:
                back_str = exp_str
            if front_str and back_str:
                break
        return front_str, back_str
    except Exception:
        return None, None


# ── IV term structure (IB batch fetch — all expirations up to 6 months) ───────

def fetch_term_structure(ticker: str, spot: float, max_dte: int = 185, no_ib: bool = False) -> list[dict]:
    """
    Fetch ATM IV for every available expiration up to max_dte calendar days out.
    Batch-fetches from IB via parallel option_chain.py calls (real-time).
    Falls back to yfinance if IB unavailable.
    Returns [{dte, expiry, iv}] sorted by dte.
    """
    try:
        import yfinance as yf
        # Get expiry list from yfinance (fast, just metadata)
        exps = yf.Ticker(ticker).options
        if not exps:
            return []

        today = date.today()
        exps_to_fetch = []
        for exp_str in exps:
            exp = date.fromisoformat(exp_str)
            dte = (exp - today).days
            if 5 <= dte <= max_dte:
                exps_to_fetch.append((exp_str, dte))

        if not exps_to_fetch:
            return []

        result = []
        # Parallel fetch from IB (or fallback to yfinance)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_chain_for_expiry, ticker, exp_str, no_ib): (exp_str, dte)
                      for exp_str, dte in exps_to_fetch}
            for fut in concurrent.futures.as_completed(futures):
                exp_str, dte = futures[fut]
                try:
                    chain_data = fut.result()
                    if not chain_data or "rows" not in chain_data:
                        continue
                    rows = chain_data.get("rows", [])
                    atm = find_atm_call(rows, spot)
                    if atm and atm.get("iv") and atm["iv"] > 0:
                        result.append({"dte": dte, "expiry": exp_str, "iv": atm["iv"]})
                except Exception:
                    continue

        return sorted(result, key=lambda x: x["dte"])
    except Exception:
        return []


# ── Pricing engine ─────────────────────────────────────────────────────────────

def price_candidate(ticker: str, front_expiry: str, back_expiry: str,
                    ff_pct: float, no_ib: bool) -> dict | None:
    """
    Fetch front + back chains in parallel (IB/yfinance), plus full IV term structure
    (IB batch parallel, all expirations ≤6 months) for the HTML report.

    Net debit = (back_call_ask + back_put_ask) - (front_call_bid + front_put_bid)
    Positive = we pay a debit to enter (normal for long calendar).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_front = pool.submit(fetch_chain_for_expiry, ticker, front_expiry, no_ib)
        f_back  = pool.submit(fetch_chain_for_expiry, ticker, back_expiry, no_ib)
        front_data = f_front.result()
        back_data  = f_back.result()

    if not front_data or not back_data:
        return None

    spot       = front_data.get("spot") or back_data.get("spot") or 0

    # Full IV term structure (IB batch fetch, all expirations ≤6 months — for chart only)
    term_structure = fetch_term_structure(ticker, spot, no_ib=no_ib) if spot else []
    front_rows = front_data.get("rows", [])
    back_rows  = back_data.get("rows", [])

    if not front_rows or not back_rows:
        return None

    # ATM strike from front chain
    atm_call = find_atm_call(front_rows, spot)
    if not atm_call:
        return None
    strike = atm_call.get("strike")
    if not strike:
        return None

    # Fetch all four legs at the ATM strike
    front_call = get_row_at_strike(front_rows, strike, "C")
    front_put  = get_row_at_strike(front_rows, strike, "P")
    back_call  = get_row_at_strike(back_rows,  strike, "C")
    back_put   = get_row_at_strike(back_rows,  strike, "P")

    if not all([front_call, front_put, back_call, back_put]):
        return None

    # Prices — selling front (use bid), buying back (use ask)
    fc_bid = safe_float(front_call.get("bid"))
    fp_bid = safe_float(front_put.get("bid"))
    bc_ask = safe_float(back_call.get("ask"))
    bp_ask = safe_float(back_put.get("ask"))

    # Zero bid/ask on any leg = unquoted / illiquid = not tradeable
    zero_legs = []
    if fp_bid == 0: zero_legs.append("front put bid")
    if bp_ask == 0: zero_legs.append("back put ask")
    if fc_bid == 0: zero_legs.append("front call bid")
    if bc_ask == 0: zero_legs.append("back call ask")
    stale = bool(zero_legs)

    # Primary structure: ATM PUT calendar (sell front put, buy back put)
    put_cal_unfilled  = bp_ask - fp_bid
    put_cal_realistic = put_cal_unfilled * SLIPPAGE_FACTOR

    # Comparison structure: ATM STRADDLE calendar (sell front straddle, buy back straddle)
    straddle_cal_unfilled  = (bc_ask + bp_ask) - (fc_bid + fp_bid)
    straddle_cal_realistic = straddle_cal_unfilled * SLIPPAGE_FACTOR

    # Primary net debit = put calendar (used for entry_ready gate)
    net_debit_unfilled  = put_cal_unfilled
    net_debit_realistic = put_cal_realistic

    # Bid-ask widths (quality indicator — use put legs for primary)
    fc_ask = safe_float(front_call.get("ask"))
    fp_ask = safe_float(front_put.get("ask"))
    bc_bid = safe_float(back_call.get("bid"))
    bp_bid = safe_float(back_put.get("bid"))
    bid_ask_width_front = fp_ask - fp_bid          # put leg only
    bid_ask_width_back  = bp_ask - bp_bid           # put leg only
    bid_ask_width_front_straddle = (fc_ask - fc_bid) + (fp_ask - fp_bid)
    bid_ask_width_back_straddle  = (bc_ask - bc_bid) + (bp_ask - bp_bid)

    # Volume at ATM strike
    vol_front = (front_call.get("volume") or 0) + (front_put.get("volume") or 0)
    vol_back  = (back_call.get("volume") or 0)  + (back_put.get("volume") or 0)

    # Entry readiness — gated on put calendar quality
    wide_spread    = bid_ask_width_front > WIDE_SPREAD_THRESHOLD or bid_ask_width_back > WIDE_SPREAD_THRESHOLD
    debit_positive = net_debit_unfilled > 0
    entry_ready    = debit_positive and not stale and not wide_spread

    max_loss_dollars = round(net_debit_realistic * 100, 2) if net_debit_realistic > 0 else None

    not_ready_reasons: list[str] = []
    if stale:
        not_ready_reasons.append(f"unquoted legs: {', '.join(zero_legs)} — illiquid / market closed")
    if not debit_positive:
        not_ready_reasons.append("credit received instead of debit — likely bad quote from IB")
    if wide_spread:
        not_ready_reasons.append(
            f"wide put spread (front ${bid_ask_width_front:.2f} + back ${bid_ask_width_back:.2f})"
        )

    # Mid prices for each leg
    fc_mid = (fc_bid + fc_ask) / 2
    fp_mid = (fp_bid + fp_ask) / 2
    bc_mid = (bc_bid + bc_ask) / 2
    bp_mid = (bp_bid + bp_ask) / 2

    # Fill scenarios — PUT calendar
    fill_aggressive = bp_ask - fp_bid              # worst case
    fill_mid        = bp_mid - fp_mid              # fair value
    fill_passive    = bp_bid - fp_ask              # best case

    # Fill scenarios — STRADDLE calendar (comparison)
    straddle_fill_aggressive = (bc_ask + bp_ask) - (fc_bid + fp_bid)
    straddle_fill_mid        = (bc_mid + bp_mid) - (fc_mid + fp_mid)
    straddle_fill_passive    = (bc_bid + bp_bid) - (fc_ask + fp_ask)

    # Live IVs from chain
    front_call_iv = safe_float(front_call.get("iv") or 0) if front_call else 0.0
    front_put_iv  = safe_float(front_put.get("iv")  or 0) if front_put  else 0.0
    back_call_iv  = safe_float(back_call.get("iv")  or 0) if back_call  else 0.0
    back_put_iv   = safe_float(back_put.get("iv")   or 0) if back_put   else 0.0
    front_iv_live = (front_call_iv + front_put_iv) / 2 if (front_call_iv + front_put_iv) > 0 else None
    back_iv_live  = (back_call_iv  + back_put_iv)  / 2 if (back_call_iv  + back_put_iv)  > 0 else None

    # Forward vol and live FF (using IVs from the chain, not Dolt)
    try:
        T1 = max((date.fromisoformat(front_expiry) - date.today()).days, 1) / 365.0
        T2 = max((date.fromisoformat(back_expiry)  - date.today()).days, 1) / 365.0
    except (ValueError, TypeError):
        T1 = T2 = None
    fwd_vol_live = ff_live = back_iv_floor = None
    # Skip live FF computation when any leg is unquoted — zero-priced legs produce
    # garbage IVs (e.g. front_iv_live halved by a zero call IV) → extreme ff_live values.
    if not stale and front_iv_live and back_iv_live and T1 and T2 and T2 > T1:
        fwd_var = (T2 * back_iv_live**2 - T1 * front_iv_live**2) / (T2 - T1)
        fwd_vol_live = math.sqrt(max(0.0, fwd_var))
        ff_live = (front_iv_live - fwd_vol_live) / front_iv_live * 100
        # Minimum back IV to keep FF >= 16%:
        # fwd_vol = 0.84 * front_iv → solve for back_iv
        back_iv_floor = front_iv_live * math.sqrt(max(0.0, (0.84**2 * (T2 - T1) + T1) / T2))
    # Cap: fwd_var clamped to 0 produces FF≈100% artifact (illiquid / outside-hours data)
    if ff_live is not None and ff_live >= 99.0:
        fwd_vol_live = ff_live = back_iv_floor = None

    # When spreads are wide (pre-market / illiquid), IB IV computation is unreliable —
    # mids are distorted by stale asks. Fall back to Dolt FF for the signal-gone gate.
    if wide_spread and ff_live is not None:
        ff_to_check = ff_pct
        src = "Dolt (wide spread — live IV unreliable)"
    else:
        ff_to_check = ff_live if ff_live is not None else ff_pct
        src = "live" if ff_live is not None else "Dolt"
    if ff_to_check is not None and ff_to_check < 16:
        entry_ready = False
        not_ready_reasons.insert(0, f"FF signal gone: {src} FF {ff_to_check:.1f}% < 16% (was {ff_pct:.1f}% at scan)")

    return {
        "scan_date":            date.today().isoformat(),
        "ticker":               ticker,
        "ff_pct":               round(ff_pct, 2),
        "front_expiry":         front_expiry,
        "back_expiry":          back_expiry,
        "strike":               strike,
        "front_call_bid":       round(fc_bid, 3),
        "front_put_bid":        round(fp_bid, 3),
        "back_call_ask":        round(bc_ask, 3),
        "back_put_ask":         round(bp_ask, 3),
        "net_debit_unfilled":   round(net_debit_unfilled, 3),
        "net_debit_realistic":  round(net_debit_realistic, 3),
        "max_loss":             max_loss_dollars,
        "max_profit":           round(net_debit_realistic, 3),
        "entry_ready":          entry_ready,
        "not_ready_reasons":    not_ready_reasons,
        "bid_ask_width_front":  round(bid_ask_width_front, 3),
        "bid_ask_width_back":   round(bid_ask_width_back, 3),
        "volume_30dte":         int(vol_front),
        "volume_60dte":         int(vol_back),
        "stale_data":           stale,
        "wide_spread_flag":     wide_spread,
        # Fill calculator — PUT calendar (primary)
        "fill_aggressive":      round(fill_aggressive, 3),
        "fill_mid":             round(fill_mid, 3),
        "fill_passive":         round(fill_passive, 3),
        "front_put_mid":        round(fp_mid, 3),
        "back_put_mid":         round(bp_mid, 3),
        # Fill calculator — STRADDLE calendar (comparison)
        "straddle_fill_aggressive": round(straddle_fill_aggressive, 3),
        "straddle_fill_mid":        round(straddle_fill_mid, 3),
        "straddle_fill_passive":    round(straddle_fill_passive, 3),
        "front_straddle_bid":   round(fc_bid + fp_bid, 3),
        "front_straddle_mid":   round(fc_mid + fp_mid, 3),
        "front_straddle_ask":   round(fc_ask + fp_ask, 3),
        "back_straddle_bid":    round(bc_bid + bp_bid, 3),
        "back_straddle_mid":    round(bc_mid + bp_mid, 3),
        "back_straddle_ask":    round(bc_ask + bp_ask, 3),
        # IV term structure
        "front_iv_live":        round(front_iv_live, 4) if front_iv_live else None,
        "back_iv_live":         round(back_iv_live,  4) if back_iv_live  else None,
        "fwd_vol_live":         round(fwd_vol_live,  4) if fwd_vol_live  else None,
        "ff_live":              round(ff_live, 2)       if ff_live       else None,
        "back_iv_floor":        round(back_iv_floor, 4) if back_iv_floor else None,
        "_term_structure":      term_structure,   # [{dte, expiry, iv}] — chart only
    }


# ── IB batch pricing (single connection, all tickers at once) ─────────────────

_OT_PYTHON       = OPTTRADER_DIR / ".venv" / "bin" / "python3"
_IB_BATCH_PRICER = OPTTRADER_DIR / "tools" / "ff_ib_batch_pricer.py"


def price_candidates_via_ib(candidates: list[dict]) -> dict[str, dict]:
    """
    Call ff_ib_batch_pricer.py once for all candidates.
    Returns {ticker: price_data} — price_data["ok"] = True on success.
    Returns {} if IB unavailable or script missing.
    """
    if not _OT_PYTHON.exists() or not _IB_BATCH_PRICER.exists():
        return {}

    payload = []
    for c in candidates:
        fe = c.get("front_expiry")
        be = c.get("back_expiry")
        if fe and be:
            payload.append({"ticker": c["ticker"], "front_expiry": fe, "back_expiry": be})

    if not payload:
        return {}

    try:
        result = subprocess.run(
            [str(_OT_PYTHON), str(_IB_BATCH_PRICER)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=str(OPTTRADER_DIR),
            timeout=120,
        )
        if result.returncode != 0 or not result.stdout.strip():
            if result.stderr.strip():
                print(f"  IB batch stderr: {result.stderr.strip()[:200]}")
            return {}
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
        print(f"  IB batch failed: {exc}")
        return {}


def build_rec_from_ib(cand: dict, raw: dict) -> dict | None:
    """
    Build the same result dict as price_candidate() but from pre-fetched IB data.
    raw comes from ff_ib_batch_pricer.py output for one ticker.
    """
    ticker      = cand["ticker"]
    ff_pct      = float(cand.get("ff_pct") or 0)
    spot        = raw["spot"]
    strike      = raw["strike"]
    front_expiry = raw["front_expiry"]
    back_expiry  = raw["back_expiry"]

    fc = raw["front_call"]
    fp = raw["front_put"]
    bc = raw["back_call"]
    bp = raw["back_put"]

    fc_bid = safe_float(fc.get("bid"))
    fp_bid = safe_float(fp.get("bid"))
    bc_ask = safe_float(bc.get("ask"))
    bp_ask = safe_float(bp.get("ask"))
    fc_ask = safe_float(fc.get("ask"))
    fp_ask = safe_float(fp.get("ask"))
    bc_bid = safe_float(bc.get("bid"))
    bp_bid = safe_float(bp.get("bid"))

    zero_legs = []
    if fp_bid == 0: zero_legs.append("front put bid")
    if bp_ask == 0: zero_legs.append("back put ask")
    if fc_bid == 0: zero_legs.append("front call bid")
    if bc_ask == 0: zero_legs.append("back call ask")
    stale = bool(zero_legs)

    put_cal_unfilled   = bp_ask - fp_bid
    put_cal_realistic  = put_cal_unfilled * SLIPPAGE_FACTOR
    straddle_cal_unfilled  = (bc_ask + bp_ask) - (fc_bid + fp_bid)
    straddle_cal_realistic = straddle_cal_unfilled * SLIPPAGE_FACTOR

    net_debit_unfilled  = put_cal_unfilled
    net_debit_realistic = put_cal_realistic

    bid_ask_width_front = fp_ask - fp_bid
    bid_ask_width_back  = bp_ask - bp_bid

    vol_front = int(fc.get("volume") or 0) + int(fp.get("volume") or 0)
    vol_back  = int(bc.get("volume") or 0) + int(bp.get("volume") or 0)

    wide_spread    = bid_ask_width_front > WIDE_SPREAD_THRESHOLD or bid_ask_width_back > WIDE_SPREAD_THRESHOLD
    debit_positive = net_debit_unfilled > 0
    entry_ready    = debit_positive and not stale and not wide_spread

    max_loss_dollars = round(net_debit_realistic * 100, 2) if net_debit_realistic > 0 else None

    not_ready_reasons: list[str] = []
    if stale:
        not_ready_reasons.append(f"unquoted legs: {', '.join(zero_legs)} — illiquid / market closed")
    if not debit_positive:
        not_ready_reasons.append("credit received instead of debit — likely bad quote from IB")
    if wide_spread:
        not_ready_reasons.append(
            f"wide put spread (front ${bid_ask_width_front:.2f} + back ${bid_ask_width_back:.2f})"
        )

    fc_mid = (fc_bid + fc_ask) / 2
    fp_mid = (fp_bid + fp_ask) / 2
    bc_mid = (bc_bid + bc_ask) / 2
    bp_mid = (bp_bid + bp_ask) / 2

    fill_aggressive = bp_ask - fp_bid
    fill_mid        = bp_mid - fp_mid
    fill_passive    = bp_bid - fp_ask

    straddle_fill_aggressive = (bc_ask + bp_ask) - (fc_bid + fp_bid)
    straddle_fill_mid        = (bc_mid + bp_mid) - (fc_mid + fp_mid)
    straddle_fill_passive    = (bc_bid + bp_bid) - (fc_ask + fp_ask)

    # IVs from batch pricer (already computed via BS solve)
    front_call_iv = safe_float(fc.get("iv") or 0)
    front_put_iv  = safe_float(fp.get("iv") or 0)
    back_call_iv  = safe_float(bc.get("iv") or 0)
    back_put_iv   = safe_float(bp.get("iv") or 0)
    front_iv_live = (front_call_iv + front_put_iv) / 2 if (front_call_iv + front_put_iv) > 0 else None
    back_iv_live  = (back_call_iv  + back_put_iv)  / 2 if (back_call_iv  + back_put_iv)  > 0 else None

    # Forward vol and live FF
    try:
        T1 = max((date.fromisoformat(front_expiry) - date.today()).days, 1) / 365.0
        T2 = max((date.fromisoformat(back_expiry)  - date.today()).days, 1) / 365.0
    except (ValueError, TypeError):
        T1 = T2 = None
    fwd_vol_live = ff_live = back_iv_floor = None
    if not stale and front_iv_live and back_iv_live and T1 and T2 and T2 > T1:
        fwd_var = (T2 * back_iv_live**2 - T1 * front_iv_live**2) / (T2 - T1)
        fwd_vol_live = math.sqrt(max(0.0, fwd_var))
        ff_live = (front_iv_live - fwd_vol_live) / front_iv_live * 100
        back_iv_floor = front_iv_live * math.sqrt(max(0.0, (0.84**2 * (T2 - T1) + T1) / T2))
    if ff_live is not None and ff_live >= 99.0:
        fwd_vol_live = ff_live = back_iv_floor = None

    # When spreads are wide (pre-market / illiquid), IB IV computation is unreliable —
    # mids are distorted by stale asks. Fall back to Dolt FF for the signal-gone gate.
    if wide_spread and ff_live is not None:
        ff_to_check = ff_pct
        src = "Dolt (wide spread — live IV unreliable)"
    else:
        ff_to_check = ff_live if ff_live is not None else ff_pct
        src = "live" if ff_live is not None else "Dolt"
    if ff_to_check is not None and ff_to_check < 16:
        entry_ready = False
        not_ready_reasons.insert(0, f"FF signal gone: {src} FF {ff_to_check:.1f}% < 16% (was {ff_pct:.1f}% at scan)")

    return {
        "scan_date":            date.today().isoformat(),
        "ticker":               ticker,
        "ff_pct":               round(ff_pct, 2),
        "front_expiry":         front_expiry,
        "back_expiry":          back_expiry,
        "strike":               strike,
        "front_call_bid":       round(fc_bid, 3),
        "front_put_bid":        round(fp_bid, 3),
        "back_call_ask":        round(bc_ask, 3),
        "back_put_ask":         round(bp_ask, 3),
        "net_debit_unfilled":   round(net_debit_unfilled, 3),
        "net_debit_realistic":  round(net_debit_realistic, 3),
        "max_loss":             max_loss_dollars,
        "max_profit":           round(net_debit_realistic, 3),
        "entry_ready":          entry_ready,
        "not_ready_reasons":    not_ready_reasons,
        "bid_ask_width_front":  round(bid_ask_width_front, 3),
        "bid_ask_width_back":   round(bid_ask_width_back, 3),
        "volume_30dte":         vol_front,
        "volume_60dte":         vol_back,
        "stale_data":           stale,
        "wide_spread_flag":     wide_spread,
        "fill_aggressive":      round(fill_aggressive, 3),
        "fill_mid":             round(fill_mid, 3),
        "fill_passive":         round(fill_passive, 3),
        "front_put_mid":        round(fp_mid, 3),
        "back_put_mid":         round(bp_mid, 3),
        "straddle_fill_aggressive": round(straddle_fill_aggressive, 3),
        "straddle_fill_mid":        round(straddle_fill_mid, 3),
        "straddle_fill_passive":    round(straddle_fill_passive, 3),
        "front_straddle_bid":   round(fc_bid + fp_bid, 3),
        "front_straddle_mid":   round(fc_mid + fp_mid, 3),
        "front_straddle_ask":   round(fc_ask + fp_ask, 3),
        "back_straddle_bid":    round(bc_bid + bp_bid, 3),
        "back_straddle_mid":    round(bc_mid + bp_mid, 3),
        "back_straddle_ask":    round(bc_ask + bp_ask, 3),
        "front_iv_live":        round(front_iv_live, 4) if front_iv_live else None,
        "back_iv_live":         round(back_iv_live,  4) if back_iv_live  else None,
        "fwd_vol_live":         round(fwd_vol_live,  4) if fwd_vol_live  else None,
        "ff_live":              round(ff_live, 2)       if ff_live       else None,
        "back_iv_floor":        round(back_iv_floor, 4) if back_iv_floor else None,
        "_term_structure":      raw.get("term_structure", []),
    }


# ── FF scan integration ────────────────────────────────────────────────────────

def load_or_run_scan(scan_file: str | None, no_ib: bool, ff_min: float) -> list[dict]:
    """
    Load candidates from an existing scan CSV, or run ff_scanner.py and read its output.
    Returns rows with FF >= ff_min.
    """
    if scan_file:
        path = Path(scan_file)
    else:
        # Run ff_scanner.py and use today's output
        scanner = Path(__file__).parent / "ff_scanner.py"
        today = date.today().isoformat()
        cmd = [sys.executable, str(scanner), "--date", today]
        if no_ib:
            cmd.append("--no-ib")
        print("Running FF scan...")
        subprocess.run(cmd, cwd=str(Path(__file__).parent))
        path = SCAN_DIR / f"{today}_ff_scan.csv"

    if not path.exists():
        print(f"Scan file not found: {path}")
        return []

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    candidates = []
    for r in rows:
        try:
            ff = float(r.get("ff_pct") or 0)
            if ff >= ff_min:
                candidates.append(r)
        except (ValueError, TypeError):
            pass
    return candidates


# ── Universe / trend data loader ─────────────────────────────────────────────

def load_universe(path: str) -> dict:
    """Load ff_universe_scan.py output — accepts either:
      - YYYY-MM-DD_candidates.csv / latest_candidates.csv  (daily workflow, preferred)
      - YYYY-MM-DD_universe.json  (full time-series, also works)
    Returns {ticker: {"series": [...], "stats": {...}, "front_expiry": ..., ...}}
    The top-level front_expiry is required by the candidate extraction loop.
    """
    import csv as _csv
    if path.endswith(".csv"):
        # Try to load the companion universe JSON for FF time-series (needed for charts).
        # latest_candidates.csv → latest_universe.json
        # YYYY-MM-DD_candidates.csv → YYYY-MM-DD_universe.json
        import os as _os
        json_path = path.replace("_candidates.csv", "_universe.json").replace("latest.csv", "latest_universe.json")
        universe_series: dict = {}
        # json_path == path means neither rewrite matched — don't json.load the CSV.
        if json_path != path and _os.path.exists(json_path):
            with open(json_path) as _jf:
                _ju = json.load(_jf)
            universe_series = {t: d.get("series", []) for t, d in _ju.items()}
            print(f"  Loaded FF series from {json_path}")

        out = {}
        with open(path, newline="") as f:
            for row in _csv.DictReader(f):
                t = row.get("ticker", "").strip()
                if not t:
                    continue
                def _f(k):
                    v = row.get(k)
                    try: return float(v) if v not in (None, "", "None") else None
                    except ValueError: return None
                out[t] = {
                    "series": universe_series.get(t, []),
                    "stats": {
                        "current_ff":        _f("ff_pct"),
                        "ff_5d_ago":         _f("ff_5d_ago"),
                        "ff_10d_ago":        _f("ff_10d_ago"),
                        "trend":             row.get("trend"),
                        "trend_slope_5d":    _f("trend_slope_5d"),
                        "days_above_thresh": _f("days_above_thresh"),
                        "consec_above_thresh": _f("consec_above_thresh"),
                        "n_observations":    _f("n_observations"),
                    },
                    # pass through ranking fields for HTML reports
                    "composite_score":  _f("composite_score"),
                    "front_expiry":     row.get("front_expiry"),
                    "back_expiry":      row.get("back_expiry"),
                    "front_dte":        _f("front_dte"),
                    "back_dte":         _f("back_dte"),
                    "front_iv":         _f("front_iv"),
                    "back_iv":          _f("back_iv"),           # Dolt IV — ML fallback
                    "back_straddle":    _f("back_straddle"),     # Dolt BS approx — ML fallback
                    "entry_debit":      _f("entry_debit"),       # Dolt BS approx — ML fallback
                    "iv_rank_20d":      _f("iv_rank_20d"),
                    "iv_hv_ratio":      _f("iv_hv_ratio"),
                    "earnings_risk":    row.get("earnings_risk"),
                    "suggested_structure": row.get("suggested_structure"),
                }
        return out

    # JSON path — extract front_expiry from series[-1] so candidate loop works identically
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for t, d in raw.items():
        series = d.get("series", [])
        latest = series[-1] if series else {}
        out[t] = {
            **d,
            "front_expiry": latest.get("front_exp"),
            "back_expiry":  latest.get("back_exp"),
            "front_dte":    latest.get("front_dte"),
            "back_dte":     latest.get("back_dte"),
            "front_iv":     latest.get("front_iv"),
        }
    return out


def enrich_with_trend(rec: dict, universe: dict) -> dict:
    """Add FF trend fields from universe data to a trade recommendation."""
    ticker = rec.get("ticker", "")
    data   = universe.get(ticker, {})
    stats  = data.get("stats", {})
    series = data.get("series", [])

    rec["current_ff_dolt"]      = stats.get("current_ff")
    rec["ff_5d_ago"]            = stats.get("ff_5d_ago")
    rec["ff_10d_ago"]           = stats.get("ff_10d_ago")
    rec["trend"]                = stats.get("trend")
    rec["trend_slope_5d"]       = stats.get("trend_slope_5d")
    rec["days_above_thresh"]    = stats.get("days_above_thresh")
    rec["consec_above_thresh"]  = stats.get("consec_above_thresh")
    rec["_ff_series"]           = series  # private — used for HTML report, not CSV
    # ML gate fields + ranking fields — in candidates CSV but not in price_candidate() output
    for _k in ("iv_rank_20d", "iv_hv_ratio", "front_iv", "earnings_risk",
               "days_to_earnings", "hv20", "ff_quality_pts", "earn_pts", "composite_score",
               "back_iv", "back_straddle", "entry_debit"):  # Dolt fallbacks for ML scorer
        if _k not in rec or rec[_k] is None:
            _v = data.get(_k)
            if _v is not None:
                rec[_k] = _v
    return rec


# ── HTML report generator ─────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{ticker} FF Report — {scan_date}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/nouislider@15/dist/nouislider.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/nouislider@15/dist/nouislider.min.css">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background:#0f1117; color:#e0e0e0; margin:0; padding:24px; }}
  h1   {{ color:#fff; margin-bottom:4px; }}
  .sub {{ color:#888; font-size:14px; margin-bottom:24px; }}
  .badge {{ display:inline-block; padding:4px 12px; border-radius:4px;
            font-weight:600; font-size:13px; margin-left:8px; }}
  .ready   {{ background:#1a4731; color:#4ade80; }}
  .notready{{ background:#3d1515; color:#f87171; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px; }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:24px; }}
  .card {{ background:#1c1f2b; border:1px solid #2d3148; border-radius:8px; padding:16px; }}
  .card h3 {{ margin:0 0 12px; font-size:13px; color:#9ca3af; text-transform:uppercase;
              letter-spacing:.05em; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  td,th {{ padding:6px 10px; text-align:left; border-bottom:1px solid #2d3148; }}
  th    {{ color:#9ca3af; font-weight:500; }}
  .pos {{ color:#4ade80; }} .neg {{ color:#f87171; }} .warn {{ color:#f59e0b; }}
  .chart-wrap {{ background:#1c1f2b; border:1px solid #2d3148; border-radius:8px;
                padding:16px; margin-bottom:24px; }}
  .chart-wrap h3 {{ margin:0 0 12px; font-size:13px; color:#9ca3af;
                    text-transform:uppercase; letter-spacing:.05em; }}
  canvas {{ max-height:260px; }}
  .fill-target {{ background:#1e2d1a; border:1px solid #2d5a3d; border-radius:6px;
                  padding:10px 14px; margin-top:10px; font-size:13px; color:#4ade80; }}
  .floor-warn  {{ background:#2d2000; border:1px solid #5a4200; border-radius:6px;
                  padding:10px 14px; margin-top:8px; font-size:13px; color:#f59e0b; }}
  .gate-row  {{ display:flex; align-items:center; padding:7px 0; border-bottom:1px solid #2d3148; gap:10px; font-size:14px; }}
  .gate-row:last-child {{ border-bottom:none; }}
  .gate-dot  {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
  .gate-label{{ flex:1; color:#9ca3af; }}
  .gate-val  {{ font-weight:600; min-width:64px; text-align:right; }}
  .gate-thresh{{ color:#555; font-size:11px; min-width:90px; text-align:right; }}
  .ml-score  {{ display:flex; align-items:baseline; justify-content:center; gap:6px;
               margin-top:14px; padding:10px 12px; border-radius:6px; }}
  .ml-score-num  {{ font-size:28px; font-weight:800; }}
  .ml-score-lbl  {{ font-size:12px; color:#9ca3af; }}
  .ml-verdict{{ margin-top:6px; padding:7px 12px; border-radius:6px; font-size:12px; font-weight:600; text-align:center; }}
  .ml-pass   {{ background:#1a3a20; color:#4ade80; border:1px solid #2d5a3d; }}
  .ml-warn   {{ background:#2d2000; color:#f59e0b; border:1px solid #5a4200; }}
  .ml-fail   {{ background:#3d1515; color:#f87171; border:1px solid #5a2020; }}
  .exit-rule {{ background:#151a2e; border:1px solid #2d3148; border-radius:6px;
                padding:10px 14px; margin-top:8px; font-size:13px; }}
  .exit-rule b {{ color:#f59e0b; }}
  .exit-rule .good {{ color:#4ade80; }}
  .exit-rule .bad  {{ color:#f87171; }}
  /* noUiSlider dark theme */
  .noUi-target {{ background:#2d3148; border:none; box-shadow:none;
                  border-radius:4px; height:5px; margin:12px 10px 4px; }}
  .noUi-connect {{ background:#60a5fa; }}
  .noUi-handle {{ background:#e0e0e0; border:2px solid #60a5fa; border-radius:50%;
                  width:16px; height:16px; top:-6px; right:-8px;
                  box-shadow:none; cursor:pointer; }}
  .noUi-handle::before, .noUi-handle::after {{ display:none; }}
  .noUi-handle:focus {{ outline:none; }}
  .ff-range-labels {{ display:flex; justify-content:space-between;
                      font-size:12px; color:#6b7280; margin-top:4px; }}
</style>
</head>
<body>
<h1>{ticker}
  <span class="badge {badge_class}">{badge_text}</span>
</h1>
<div class="sub">FF Signal Report — {scan_date} &nbsp;|&nbsp; Strike: {strike} &nbsp;|&nbsp; Front: {front_expiry} ({front_dte}d) &nbsp;|&nbsp; Back: {back_expiry} ({back_dte}d)</div>

<!-- Row 1: Trade + Signal -->
<div class="grid">
  <div class="card">
    <h3>Trade Details</h3>
    <table>
      <tr><th>Structure</th><td><b>ATM Put Calendar</b> <span style="color:#4ade80;font-size:12px">★ primary</span> <span style="color:#555;font-size:11px">(limit order — target mid, max mid+15%)</span></td></tr>
      <tr><th>Sell front put</th><td>{front_put_bid:.2f} &nbsp;<span style="color:#888;font-size:12px">(bid)</span></td></tr>
      <tr><th>Buy back put</th><td>{back_put_ask:.2f} &nbsp;<span style="color:#888;font-size:12px">(ask)</span></td></tr>
      <tr><th>Net debit aggressive</th><td><b>${fill_aggressive:.2f}/sh</b> &nbsp;<span style="color:#888;font-size:12px">(back ask − front bid)</span></td></tr>
      <tr><th>Net debit mid-market</th><td><b style="color:#4ade80">${fill_mid:.2f}/sh</b> &nbsp;<span style="color:#4ade80;font-size:12px">← target fill</span></td></tr>
      <tr><th>Net debit passive</th><td><b>${fill_passive:.2f}/sh</b> &nbsp;<span style="color:#888;font-size:12px">(back bid − front ask)</span></td></tr>
      <tr><th>Max fill (mid+15%)</th><td><b style="color:{put_mid_col}">${put_fill_max_price:.2f}/sh</b> &nbsp;<span style="color:#888;font-size:12px">skip if aggressive &gt; this</span></td></tr>
      <tr><th>Straddle cal mid (if both legs)</th><td>${straddle_fill_mid:.2f}/sh &nbsp;<span style="color:#555;font-size:11px">(combo ask viable only near mid)</span></td></tr>
      <tr><th>Entry ready</th>
          <td class="{entry_class}"><b>{entry_text}</b>{not_ready_detail}</td></tr>
    </table>
  </div>
  <div class="card">
    <h3>FF Signal Stats</h3>
    <table>
      <tr><th>Forward Factor (Dolt)</th>
          <td class="{ff_class}"><b>{ff_pct:+.1f}%</b></td></tr>
      <tr><th>Forward Factor (live IVs)</th>
          <td class="{ff_live_class}"><b>{ff_live_str}</b></td></tr>
      <tr><th>FF 5 days ago</th><td>{ff_5d}</td></tr>
      <tr><th>FF 10 days ago</th><td>{ff_10d}</td></tr>
      <tr><th>Trend (5d slope)</th><td>{trend} ({trend_slope:+.1f}%/day)</td></tr>
      <tr><th>Days above 16% (of {n_obs})</th><td>{days_above}</td></tr>
      <tr><th>Consecutive days above 16%</th><td>{consec_above}</td></tr>
      <tr><th>Volume (front / back)</th><td>{vol_front} / {vol_back}</td></tr>
    </table>
  </div>
</div>

<!-- Row 1b: ML Entry Gates + Exit Rules -->
<div class="grid">
  <div class="card">
    <h3>ML Entry Gates <span style="color:#555;font-size:11px;font-weight:400;text-transform:none">(backtest-derived — LR model, 17 features, post-commission target)</span></h3>
    <div class="gate-row">
      <div class="gate-dot" style="background:{dps_dot}"></div>
      <div class="gate-label">Debit / spot price</div>
      <div class="gate-val" style="color:{dps_color}">{dps_str}</div>
      <div class="gate-thresh">&lt;2% ✓ &lt;3% ⚠ &gt;3% ✗</div>
    </div>
    <div class="gate-row">
      <div class="gate-dot" style="background:{comm_drag_dot}"></div>
      <div class="gate-label">IB commission drag <span style="color:#555;font-size:11px">($5.20/trade ÷ debit)</span></div>
      <div class="gate-val" style="color:{comm_drag_color}">{comm_drag_str}</div>
      <div class="gate-thresh">&lt;5% ✓ &lt;10% ⚠ &gt;10% ✗</div>
    </div>
    <div class="gate-row">
      <div class="gate-dot" style="background:{ivhv_dot}"></div>
      <div class="gate-label">IV / HV ratio <span style="color:#555;font-size:11px">(#1 failure driver)</span></div>
      <div class="gate-val" style="color:{ivhv_color}">{ivhv_str}</div>
      <div class="gate-thresh">&lt;2.5 ✓ &lt;4 ⚠ &gt;4 ✗</div>
    </div>
    <div class="gate-row">
      <div class="gate-dot" style="background:{ff_gate_dot}"></div>
      <div class="gate-label">Forward Factor level {ff_gate_source}</div>
      <div class="gate-val" style="color:{ff_gate_color}">{ff_gate_str}</div>
      <div class="gate-thresh">15–40% ✓ 40–60% ⚠ else ✗</div>
    </div>
    <div class="gate-row">
      <div class="gate-dot" style="background:{dte_dot}"></div>
      <div class="gate-label">Front DTE (from today)</div>
      <div class="gate-val" style="color:{dte_color}">{dte_str}</div>
      <div class="gate-thresh">≥30d ✓ 20–29d ⚠ &lt;20 ✗</div>
    </div>
    <div class="gate-row">
      <div class="gate-dot" style="background:{ivr_dot}"></div>
      <div class="gate-label">IV Rank (20d)</div>
      <div class="gate-val" style="color:{ivr_color}">{ivr_str}</div>
      <div class="gate-thresh">&lt;60 ✓ 60–80 ⚠ &gt;80 ✗</div>
    </div>
    <div style="display:flex;gap:8px;margin-bottom:8px">
      <div class="ml-score {ml_put_score_class}" style="flex:1">
        <span class="ml-score-num">{ml_put_score:.0f}%</span>
        <span class="ml-score-lbl">Put cal ★ primary</span>
      </div>
      <div class="ml-score {ml_score_class}" style="flex:1">
        <span class="ml-score-num">{ml_score:.0f}%</span>
        <span class="ml-score-lbl">Straddle cal</span>
      </div>
    </div>
    <div class="ml-verdict {ml_verdict_class}">{ml_verdict_text}</div>
    <div style="margin-top:10px;padding:8px 10px;background:#0f1117;border-radius:6px;font-size:12px">
      <div style="margin-bottom:4px">
        <span style="color:#888">FF quality pts:</span>
        <span style="color:{ff_qual_color};font-weight:600;margin-left:6px">{ff_qual_str}</span>
      </div>
      <div>
        <span style="color:#888">Composite score:</span>
        <span style="color:{comp_color};margin-left:6px">{comp_str}</span>
      </div>
      <div style="margin-top:6px">
        <span style="color:#888;font-size:11px">ML data: </span>
        <span style="color:{ml_data_src_color};font-size:11px;font-weight:600" title="{ml_data_src_title}">{ml_data_src}</span>
      </div>
      <div style="color:#555;font-size:11px;margin-top:4px">P(bigwin) = P(return&gt;+50%). LR v2, AUC 0.795, 177k trades 2020–2025. Top driver: DTE structure. ≥45% = top tier.</div>
    </div>
  </div>
  <div class="card">
    <h3>Exit Rules <span style="color:#555;font-size:11px;font-weight:400;text-transform:none">(primary: TP40 | secondary: FF risk trigger)</span></h3>
    <table>
      <tr><th>Entry FF (today)</th><td><b>{ff_pct:+.1f}%</b></td></tr>
      <tr><th>★ Profit exit (TP40)</th>
          <td class="good"><b>Put cal target: {tp40_str}</b>
          <br><span style="color:#888;font-size:11px">TP40 confirmed optimal (177k trades). Target = put_debit + 40% × max_profit_put.</span></td></tr>
      <tr><th>Risk exit (FF rise)</th>
          <td class="bad"><b>+15pt rule: exit if FF rises to {ff_exit_trigger:.0f}%</b>
          <br><span style="color:#888;font-size:11px">Front IV not compressing — term structure worsening</span></td></tr>
      <tr><th>Time exit</th><td>Hold to ~7 DTE on front &nbsp;<span style="color:#888;font-size:11px">({front_expiry} − 7d)</span></td></tr>
      <tr><th>Failure profile</th>
          <td><span style="color:#9ca3af;font-size:12px">Deep losses occur when IV/HV &gt;4 at entry. This ticker: <b style="color:{ivhv_color}">{ivhv_str}</b></span></td></tr>
    </table>
    <div class="exit-rule" style="margin-top:10px">
      <b>Fill cost vs EV (FF&gt;20%, put limit orders, 21k trades):</b><br>
      <span class="good">mid fill (+0%) → +23% EV, 56% win</span><br>
      <span class="good">mid+15% (typical limit) → +7% EV, 36% win</span><br>
      <span class="bad">mid+25% (combo ask) → −1% EV &nbsp;·&nbsp; mid+35% → −9% EV</span><br>
      <span style="color:#f59e0b;font-size:11px">Works as put limit order. Straddle only if both legs fill near mid.</span>
    </div>
    <div class="exit-rule">
      <b>Monitor:</b> Re-check FF daily. Take profit at TP40; cut if FF rises above risk trigger.<br>
      <span class="bad">Risk: FF &gt; {ff_exit_trigger:.0f}% → exit immediately (front IV not compressing)</span>
    </div>
  </div>
</div>

<!-- Row 2: Fill Calculator + IV Term Structure -->
<div class="grid">
  <div class="card">
    <h3>Fill Calculator — <span style="color:#4ade80">Put Calendar</span> <span style="color:#4ade80;font-size:11px;font-weight:400;text-transform:none">★ primary</span></h3>
    <table>
      <tr>
        <th></th>
        <th style="color:#f87171">Sell front put</th>
        <th style="color:#4ade80">Buy back put</th>
        <th>Put debit</th>
        <th style="color:#9ca3af;font-size:11px">/spot</th>
      </tr>
      <tr>
        <td><b>Aggressive</b> <span style="color:#888;font-size:11px">mkt</span></td>
        <td>{front_put_bid:.2f} (bid)</td>
        <td>{back_put_ask:.2f} (ask)</td>
        <td><b>${fill_aggressive:.2f}</b></td>
        <td style="color:{put_agg_col};font-weight:600">{put_agg_pct}</td>
      </tr>
      <tr style="background:#151a26">
        <td><b>Mid-market</b> <span style="color:#4ade80;font-size:11px">★ target</span></td>
        <td>{front_put_mid:.2f} (mid)</td>
        <td>{back_put_mid:.2f} (mid)</td>
        <td><b style="color:#4ade80">${fill_mid:.2f}</b></td>
        <td style="color:{put_mid_col};font-weight:600">{put_mid_pct}</td>
      </tr>
      <tr>
        <td><b>Passive</b> <span style="color:#888;font-size:11px">limit</span></td>
        <td>{front_put_ask:.2f} (ask)</td>
        <td>{back_put_bid:.2f} (bid)</td>
        <td><b>${fill_passive:.2f}</b></td>
        <td style="color:{put_pas_col};font-weight:600">{put_pas_pct}</td>
      </tr>
      <tr style="border-top:1px solid #3d4168">
        <td colspan="4" style="color:#6b7280;font-size:12px;padding-top:8px">
          Fill limit: target ≤ <b style="color:#4ade80">${put_fill_target_price:.2f}</b> (1%/spot ✓)
          &nbsp;·&nbsp; max <b style="color:#f59e0b">${put_fill_max_price:.2f}</b> (1.5%/spot ⚠)
          &nbsp;·&nbsp; <span style="color:#f87171">skip if aggressive &gt; ${put_fill_max_price:.2f}</span>
        </td>
        <td></td>
      </tr>
    </table>
    <h3 style="margin-top:16px;font-size:12px;color:#6b7280">Straddle Calendar <span style="font-weight:400;text-transform:none">(if both legs fill near mid)</span></h3>
    <table style="font-size:13px">
      <tr>
        <th></th>
        <th style="color:#f87171">Sell front straddle</th>
        <th style="color:#4ade80">Buy back straddle</th>
        <th>Straddle debit</th>
        <th style="color:#9ca3af;font-size:11px">/spot</th>
      </tr>
      <tr>
        <td><b>Aggressive</b></td>
        <td>{front_straddle_bid:.2f} (bid)</td>
        <td>{back_straddle_ask:.2f} (ask)</td>
        <td>${straddle_fill_aggressive:.2f}</td>
        <td style="color:{str_agg_col}">{str_agg_pct}</td>
      </tr>
      <tr style="background:#151a26">
        <td><b>Mid</b></td>
        <td>{front_straddle_mid:.2f} (mid)</td>
        <td>{back_straddle_mid:.2f} (mid)</td>
        <td>${straddle_fill_mid:.2f}</td>
        <td style="color:{str_mid_col}">{str_mid_pct}</td>
      </tr>
      <tr>
        <td><b>Passive</b></td>
        <td>{front_straddle_ask:.2f} (ask)</td>
        <td>{back_straddle_bid:.2f} (bid)</td>
        <td>${straddle_fill_passive:.2f}</td>
        <td style="color:{str_pas_col}">{str_pas_pct}</td>
      </tr>
    </table>
    <div class="fill-target">FF check: signal is <b>{ff_live_str}</b> at current IVs (front {front_iv_live_pct} / back {back_iv_live_pct}). Verify FF still above 16% at entry time.</div>
    <div class="floor-warn">⚠ FF floor: back IV must stay above <b>{back_iv_floor_pct}</b> for FF ≥ 16% to hold. Current back IV: {back_iv_live_pct} → margin {iv_margin_bp}.</div>
  </div>
  <div class="card" style="display:flex;flex-direction:column">
    <h3>IV Term Structure — all expirations ≤ 6 months (IB real-time)</h3>
    <canvas id="tsChart" style="flex:1"></canvas>
    <div style="font-size:11px;color:#6b7280;margin-top:8px">
      <span style="color:#f87171">●</span> Front ({front_dte}d) IV: {front_iv_live_pct} &nbsp;
      <span style="color:#f59e0b">●</span> Fwd vol: {fwd_vol_live_pct} &nbsp;
      <span style="color:#4ade80">●</span> Back ({back_dte}d) IV: {back_iv_live_pct}
    </div>
  </div>
</div>

<!-- Row 3: FF History -->
<div class="chart-wrap">
  <h3>Forward Factor History — {n_obs} observations
    &nbsp;<span style="color:#555;font-size:11px">(Dolt; run --days 252 for 1yr)</span>
    &nbsp;|&nbsp;
    <span style="color:#f59e0b;font-size:12px">⬤ entry ({ff_pct:+.1f}%)</span>
    &nbsp;
    <span style="color:#f87171;font-size:12px">── exit trigger ({ff_exit_trigger:.0f}%)</span>
    &nbsp;
    <span style="color:#9ca3af;font-size:12px">(profit: TP40 — calendar value, not FF level)</span>
  </h3>
  <canvas id="ffChart" style="max-height:300px"></canvas>
  <!-- Date range slider (noUiSlider — two connected handles) -->
  <div id="ffRangeSlider"></div>
  <div class="ff-range-labels">
    <span>From: <b id="ffSlFromLbl" style="color:#e0e0e0"></b></span>
    <span>To: <b id="ffSlToLbl" style="color:#e0e0e0"></b></span>
  </div>
  <div style="font-size:11px;color:#6b7280;margin-top:8px">
    <b style="color:#e0e0e0">How to read:</b>
    Yellow dot = today's entry. Red dashed line = risk exit (FF &gt; {ff_exit_trigger:.0f}%).
    FF rising above the red line = front IV not compressing; exit immediately.
    Profit exit (TP40) is based on calendar value, not FF level — see Exit Rules above.
    Drag sliders to zoom into a date range and auto-rescale Y.
  </div>
</div>

<script>
const allLabels    = {labels_json};
const allFfData    = {ff_data_json};
const fullDates    = {full_dates_json};
const threshold    = 16.0;
const exitTrigger  = {ff_exit_trigger};

function makePointStyles(data) {{
  const n = data.length;
  return {{
    radii:   data.map((_, i) => i === n - 1 ? 7 : 3),
    colors:  data.map((_, i) => i === n - 1 ? '#f59e0b' : '#60a5fa'),
    borders: data.map((_, i) => i === n - 1 ? '#fff'    : '#60a5fa'),
    widths:  data.map((_, i) => i === n - 1 ? 2         : 1),
  }};
}}

// Segment colour reads live chart data so it works after slicing
function segmentColor(ctx) {{
  const i    = ctx.p0DataIndex;
  const data = ctx.chart.data.datasets[0].data;
  if (i >= data.length - 1) return '#60a5fa';
  const prev = data[i], next = data[i + 1];
  if (prev == null || next == null) return '#60a5fa';
  return next > prev ? '#f87171' : next < prev ? '#4ade80' : '#60a5fa';
}}

const ps0 = makePointStyles(allFfData);
const ffChart = new Chart(document.getElementById('ffChart'), {{
  type: 'line',
  data: {{
    labels: allLabels.slice(),
    datasets: [
      {{
        label: 'Forward Factor (%)',
        data: allFfData.slice(),
        borderColor: '#60a5fa',
        segment: {{ borderColor: segmentColor }},
        backgroundColor: 'rgba(96,165,250,0.06)',
        tension: 0.2,
        pointRadius: ps0.radii,
        pointBackgroundColor: ps0.colors,
        pointBorderColor: ps0.borders,
        pointBorderWidth: ps0.widths,
        fill: true,
      }},
      {{
        label: 'Entry threshold (16%)',
        data: allLabels.map(() => threshold),
        borderColor: '#f59e0b',
        borderDash: [6, 3],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
      }},
      {{
        label: 'Exit trigger (' + exitTrigger.toFixed(0) + '%)',
        data: allLabels.map(() => exitTrigger),
        borderColor: 'rgba(248,113,113,0.7)',
        borderDash: [4, 2],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
      }},
    ],
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: '#9ca3af', boxWidth: 20, font: {{ size: 11 }} }} }},
      tooltip: {{
        callbacks: {{
          title: items => {{
            // fullDates[0..N] — chart labels are already the right slice,
            // so pull the full date from the same offset in fullDates
            const chartLabel = items[0].label;
            const fullIdx = allLabels.indexOf(chartLabel);
            return fullIdx >= 0 ? (fullDates[fullIdx] || chartLabel) : chartLabel;
          }},
          label: ctx => {{
            if (ctx.datasetIndex !== 0) return null;
            const v = ctx.parsed.y;
            return v != null ? 'FF: ' + v.toFixed(1) + '%' : null;
          }},
        }},
        filter: item => item.datasetIndex === 0,
        backgroundColor: '#1c1f2b',
        titleColor: '#e0e0e0',
        bodyColor: '#4ade80',
        borderColor: '#2d3148',
        borderWidth: 1,
        padding: 10,
      }},
    }},
    scales: {{
      x: {{ ticks: {{ color:'#6b7280', maxRotation:45 }}, grid:{{ color:'#2d3148' }} }},
      y: {{ ticks: {{ color:'#6b7280', callback: v => v.toFixed(0)+'%' }},
            grid:{{ color:'#2d3148' }} }},
    }},
  }},
}});

// ── Range slider (noUiSlider) ─────────────────────────────────────────────────
const nPts = allLabels.length;

function applyRange(f, t) {{
  const visLabels = allLabels.slice(f, t + 1);
  const visFfData = allFfData.slice(f, t + 1);
  const ps = makePointStyles(visFfData);

  const ds = ffChart.data.datasets;
  ffChart.data.labels            = visLabels;
  ds[0].data                     = visFfData;
  ds[0].pointRadius              = ps.radii;
  ds[0].pointBackgroundColor     = ps.colors;
  ds[0].pointBorderColor         = ps.borders;
  ds[0].pointBorderWidth         = ps.widths;
  ds[1].data = visLabels.map(() => threshold);
  ds[2].data = visLabels.map(() => exitTrigger);

  // Auto-rescale Y to visible data + reference lines
  const visible = visFfData.filter(v => v != null);
  if (visible.length > 0) {{
    const refs = [threshold, exitTrigger];
    const yMin = Math.min(...visible, ...refs);
    const yMax = Math.max(...visible, ...refs);
    const pad  = Math.max((yMax - yMin) * 0.12, 3);
    ffChart.options.scales.y.min = Math.floor(yMin - pad);
    ffChart.options.scales.y.max = Math.ceil(yMax  + pad);
  }}

  ffChart.update('none');
  document.getElementById('ffSlFromLbl').textContent = fullDates[f] || allLabels[f] || '';
  document.getElementById('ffSlToLbl').textContent   = fullDates[t] || allLabels[t] || '';
}}

// Initialise noUiSlider
const rangeEl = document.getElementById('ffRangeSlider');
noUiSlider.create(rangeEl, {{
  start:   [0, nPts - 1],
  connect: true,
  step:    1,
  range:   {{ min: 0, max: Math.max(nPts - 1, 1) }},
}});
// Set initial labels
document.getElementById('ffSlFromLbl').textContent = fullDates[0]        || allLabels[0]        || '';
document.getElementById('ffSlToLbl').textContent   = fullDates[nPts - 1] || allLabels[nPts - 1] || '';

rangeEl.noUiSlider.on('update', function(values) {{
  applyRange(Math.round(parseFloat(values[0])), Math.round(parseFloat(values[1])));
}});

const tsDtes   = {ts_dtes_json};
const tsIvs    = {ts_ivs_json};
const tsExps   = {ts_exps_json};
const tsPtColors = {ts_pt_colors_json};

new Chart(document.getElementById('tsChart'), {{
  type: 'line',
  data: {{
    labels: tsDtes.map((d, i) => tsExps[i] + ' (' + d + 'd)'),
    datasets: [{{
      label: 'ATM IV',
      data: tsIvs,
      borderColor: '#60a5fa',
      backgroundColor: 'rgba(96,165,250,0.06)',
      tension: 0.3,
      fill: true,
      pointRadius: 5,
      pointBackgroundColor: tsPtColors,
      pointBorderColor: tsPtColors,
    }}],
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => 'IV: ' + (ctx.parsed.y * 100).toFixed(1) + '%' }} }},
    }},
    scales: {{
      x: {{ ticks: {{ color:'#6b7280', maxRotation:45, font:{{size:10}} }}, grid:{{ color:'#2d3148' }} }},
      y: {{ ticks: {{ color:'#6b7280', callback: v => (v*100).toFixed(0)+'%' }},
            grid:{{ color:'#2d3148' }}, title:{{ display:true, text:'ATM IV', color:'#6b7280' }} }},
    }},
  }},
}});
</script>
</body>
</html>
"""


def generate_html_report(rec: dict, scan_date: str) -> str:
    """Generate a self-contained HTML report for one trade candidate."""
    ticker   = rec.get("ticker", "")
    series   = rec.get("_ff_series", [])
    ff_pct   = rec.get("ff_pct") or 0
    entry    = rec.get("entry_ready", False)
    stats    = rec  # all stats are in the rec dict already

    # Chart data
    labels   = [r["date"][-5:] for r in series]   # MM-DD
    ff_vals  = [r.get("ff_pct") for r in series]

    def fmt_opt(v, fmt="{:+.1f}%"):
        return fmt.format(v) if v is not None else "n/a"

    # Compute DTE from today — front_dte/back_dte in rec are from scan date, not today
    def _dte_from_today(expiry_str: str) -> str:
        if not expiry_str or expiry_str == "n/a":
            return "?"
        try:
            return str((date.fromisoformat(expiry_str) - date.today()).days)
        except ValueError:
            return "?"

    front_dte_live = _dte_from_today(rec.get("front_expiry", ""))
    back_dte_live  = _dte_from_today(rec.get("back_expiry", ""))

    # Reasons why entry is not ready (shown inline when not ready)
    reasons = rec.get("not_ready_reasons") or []
    not_ready_detail = ""
    if not entry and reasons:
        not_ready_detail = " &mdash; " + "; ".join(reasons)

    contract_cost      = (rec.get("net_debit_realistic") or 0) * 100
    fill_mid_val       = rec.get("fill_mid") or 0
    fill_mid_contract  = fill_mid_val * 100

    # Live IV / FF fields
    front_iv_live  = rec.get("front_iv_live")
    back_iv_live   = rec.get("back_iv_live")
    fwd_vol_live   = rec.get("fwd_vol_live")
    ff_live_val    = rec.get("ff_live")
    back_iv_floor  = rec.get("back_iv_floor")

    def _pct(v):
        return f"{v*100:.1f}%" if v is not None else "n/a"

    front_iv_live_pct = _pct(front_iv_live)
    back_iv_live_pct  = _pct(back_iv_live)
    fwd_vol_live_pct  = _pct(fwd_vol_live)
    back_iv_floor_pct = _pct(back_iv_floor)

    ff_live_str   = f"{ff_live_val:+.1f}%" if ff_live_val is not None else "n/a (market closed)"
    ff_live_class = ("pos" if (ff_live_val or 0) >= 16 else "neg") if ff_live_val is not None else "warn"

    # IV margin in basis points
    if back_iv_live is not None and back_iv_floor is not None:
        iv_margin_bp = f"{(back_iv_live - back_iv_floor)*10000:+.0f} bp"
    else:
        iv_margin_bp = "n/a"

    # ── ML Entry Gates ────────────────────────────────────────────────────────
    def _gate(val, green_fn, yellow_fn=None, fmt_fn=None):
        """Return (dot_color, text_color, display_str) for a gate value."""
        if val is None:
            return "#6b7280", "#6b7280", "n/a"
        display = fmt_fn(val) if fmt_fn else str(val)
        if green_fn(val):
            return "#4ade80", "#4ade80", display
        if yellow_fn and yellow_fn(val):
            return "#f59e0b", "#f59e0b", display
        return "#f87171", "#f87171", display

    # debit / spot  (use straddle mid — model trained on straddle debit)
    fill_mid_v  = rec.get("fill_mid") or 0               # put cal mid — kept for put table
    str_mid_v   = rec.get("straddle_fill_mid") or 0      # straddle mid — for gates + ML
    strike_v    = float(rec.get("strike") or 1)
    debit_pct_v = str_mid_v / strike_v * 100 if strike_v > 0 else None
    dps_dot, dps_color, dps_str = _gate(
        debit_pct_v,
        lambda v: v < 2.0,
        lambda v: v < 3.0,
        lambda v: f"{v:.2f}%",
    )
    # IB commission as % of straddle debit — $5.20/trade = $0.052/share
    _comm_drag_val = (0.052 / str_mid_v * 100) if str_mid_v > 0 else None
    comm_drag_dot, comm_drag_color, comm_drag_str = _gate(
        _comm_drag_val,
        lambda v: v < 5.0,
        lambda v: v < 10.0,
        lambda v: f"{v:.1f}% of debit",
    )

    # IV / HV ratio
    iv_hv_v = rec.get("iv_hv_ratio")
    ivhv_dot, ivhv_color, ivhv_str = _gate(
        iv_hv_v,
        lambda v: v < 2.5,
        lambda v: v < 4.0,
        lambda v: f"{v:.2f}×",
    )

    # FF level gate — prefer live FF (current IVs) over Dolt EOD when market is open
    ff_gate_v = ff_live_val if ff_live_val is not None else ff_pct
    ff_gate_dot, ff_gate_color, ff_gate_str = _gate(
        ff_gate_v,
        lambda v: 15 <= v <= 40,
        lambda v: 40 < v <= 60,
        lambda v: f"{v:+.1f}%",
    )

    # Front DTE gate (from today, not scan date)
    front_dte_int = int(front_dte_live) if str(front_dte_live).lstrip("-").isdigit() else 0
    dte_dot, dte_color, dte_str = _gate(
        front_dte_int,
        lambda v: v >= 30,
        lambda v: v >= 20,
        lambda v: f"{v}d",
    )

    # IV Rank gate (iv_rank_20d stored as 0-100 in candidates CSV)
    ivr_v = rec.get("iv_rank_20d")
    ivr_dot, ivr_color, ivr_str = _gate(
        ivr_v,
        lambda v: v < 60,
        lambda v: v < 80,
        lambda v: f"{v:.0f}%",
    )

    # Overall verdict
    gates_pass = sum([
        dps_dot  == "#4ade80",
        ivhv_dot == "#4ade80",
        ff_gate_dot == "#4ade80",
        dte_dot  == "#4ade80",
        ivr_dot  == "#4ade80",
    ])
    gates_warn = sum([
        dps_dot  == "#f59e0b",
        ivhv_dot == "#f59e0b",
        ff_gate_dot == "#f59e0b",
        dte_dot  == "#f59e0b",
        ivr_dot  == "#f59e0b",
    ])
    if gates_pass >= 4:
        ml_verdict_class = "ml-pass"
        ml_verdict_text  = f"✓ {gates_pass}/5 gates pass — strong setup"
    elif gates_pass >= 2 or (gates_pass + gates_warn) >= 4:
        ml_verdict_class = "ml-warn"
        ml_verdict_text  = f"⚠ {gates_pass}/5 gates pass, {gates_warn} marginal — trade with caution"
    else:
        ml_verdict_class = "ml-fail"
        ml_verdict_text  = f"✗ {gates_pass}/5 gates pass — high failure risk, skip"

    # ── ML composite win-probability scores (straddle v2 + put v1) ───────────
    # Both LR bin_bigwin models, AUC 0.795 OOF, 177k trades 2020-2025.
    ml_score_raw = _ml_score_v2(rec)
    ml_score = round(ml_score_raw * 100, 1)
    if ml_score >= 45:
        ml_score_class = "ml-pass"
    elif ml_score >= 35:
        ml_score_class = "ml-warn"
    else:
        ml_score_class = "ml-fail"

    ml_put_score_raw = _ml_score_put_v2(rec)
    ml_put_score = round(ml_put_score_raw * 100, 1)
    if ml_put_score >= 45:
        ml_put_score_class = "ml-pass"
    elif ml_put_score >= 35:
        ml_put_score_class = "ml-warn"
    else:
        ml_put_score_class = "ml-fail"

    # Data source: "live" when chain IVs available, "Dolt EOD" when fell back to prev-day data
    _using_live = (rec.get("back_iv_live") is not None
                   and rec.get("back_straddle_mid") is not None
                   and rec.get("straddle_fill_mid") is not None)
    if _using_live:
        ml_data_src       = "live"
        ml_data_src_color = "#4ade80"
        ml_data_src_title = "Scored on live chain prices"
    else:
        ml_data_src       = "⚠ Dolt EOD (prev day)"
        ml_data_src_color = "#f59e0b"
        ml_data_src_title = "Live chain data unavailable — ML scored on yesterday's Dolt snapshot. Re-run during market hours for live score."

    # ── FF quality pts and composite score — with backtest-derived annotations ─
    ff_qual_raw  = rec.get("ff_quality_pts")
    ff_qual      = float(ff_qual_raw) if ff_qual_raw is not None else None
    comp_raw     = rec.get("composite_score")
    comp_score   = float(comp_raw) if comp_raw is not None else None

    # ff_quality_pts 5-8 is the backtest sweet spot (OOS: 57.7% win, +123% avg P&L)
    if ff_qual is not None:
        if 5 <= ff_qual <= 8:
            ff_qual_color = "#4ade80"
            ff_qual_badge = f"★ sweet spot (5–8)"
        elif ff_qual < 5:
            ff_qual_color = "#9ca3af"
            ff_qual_badge = "below sweet spot"
        else:
            ff_qual_color = "#f59e0b"
            ff_qual_badge = "above sweet spot (8+)"
        ff_qual_str = f"{ff_qual:.0f} pts — {ff_qual_badge}"
    else:
        ff_qual_color = "#555"
        ff_qual_str   = "n/a"

    # composite_score 80+ tier underperforms in backtest (48.1% OOS win, worst tier)
    if comp_score is not None:
        if comp_score >= 80:
            comp_color   = "#f59e0b"
            comp_warning = " ⚠ top tier underperforms (backtest: 48% win)"
        elif comp_score >= 60:
            comp_color   = "#9ca3af"
            comp_warning = ""
        else:
            comp_color   = "#4ade80"
            comp_warning = " ✓ lower tier outperforms (backtest: 54% win)"
        comp_str = f"{comp_score:.0f}{comp_warning}"
    else:
        comp_color   = "#555"
        comp_str     = "n/a"

    # ── Fill viability — debit/spot% for straddle and put scenarios ──────────
    def _fill_color(debit: float, strike: float, thresholds=(2.0, 3.0)):
        pct = debit / strike * 100 if strike else 0
        col = "#4ade80" if pct < thresholds[0] else ("#f59e0b" if pct < thresholds[1] else "#f87171")
        return col, f"{pct:.1f}%"

    _fv_strike = float(rec.get("strike") or 1)
    str_agg_col, str_agg_pct = _fill_color(rec.get("straddle_fill_aggressive") or 0, _fv_strike)
    str_mid_col, str_mid_pct = _fill_color(rec.get("straddle_fill_mid")        or 0, _fv_strike)
    str_pas_col, str_pas_pct = _fill_color(rec.get("straddle_fill_passive")    or 0, _fv_strike)
    fill_target_price = round(_fv_strike * 0.02, 2)
    fill_max_price    = round(_fv_strike * 0.03, 2)

    # Put-specific thresholds: ≈ half of straddle (1% / 1.5% of spot)
    put_agg_col, put_agg_pct = _fill_color(rec.get("fill_aggressive") or 0, _fv_strike, (1.0, 1.5))
    put_mid_col, put_mid_pct = _fill_color(rec.get("fill_mid")        or 0, _fv_strike, (1.0, 1.5))
    put_pas_col, put_pas_pct = _fill_color(rec.get("fill_passive")    or 0, _fv_strike, (1.0, 1.5))
    put_fill_target_price = round(_fv_strike * 0.010, 2)   # 1%/spot  ✓
    put_fill_max_price    = round(_fv_strike * 0.015, 2)   # 1.5%/spot ⚠

    # ── Exit levels ───────────────────────────────────────────────────────────
    ff_exit_trigger = round(ff_pct + 15, 1)
    # TP40 calculated on put debit (primary structure)
    _tp_put_debit  = fill_mid_val   # put calendar mid fill
    _tp_back_put   = rec.get("back_put_mid") or 0
    _tp_t_fwd  = max(
        (int(back_dte_live)  if str(back_dte_live).isdigit()  else 0) -
        (int(front_dte_live) if str(front_dte_live).isdigit() else 0),
        0,
    )
    _tp_t_back = int(back_dte_live) if str(back_dte_live).isdigit() else 1
    if _tp_put_debit > 0 and _tp_back_put > 0 and _tp_t_fwd > 0:
        _tp_max_put = _tp_back_put * math.sqrt(_tp_t_fwd / _tp_t_back) - _tp_put_debit
        tp40_pnl    = round(0.40 * _tp_max_put, 2)
        tp40_target = round(_tp_put_debit + tp40_pnl, 2)
        tp40_ret    = round(tp40_pnl / _tp_put_debit * 100, 0) if _tp_put_debit > 0 else 0
        tp40_str    = f"${tp40_target:.2f} (P&amp;L: +${tp40_pnl:.2f} / +{tp40_ret:.0f}%)"
    else:
        tp40_str    = "n/a"
        tp40_pnl    = tp40_target = tp40_ret = 0

    # IV term structure — prefer live data (IB batch or yfinance), fall back to 2-point chart.
    # Always inject front_iv_live / back_iv_live as anchors so the traded legs are exact.
    ts_raw = rec.get("_term_structure") or []
    front_dte_int = int(front_dte_live) if str(front_dte_live).isdigit() else None
    back_dte_int  = int(back_dte_live)  if str(back_dte_live).isdigit()  else None

    # Build a working list; we'll merge/replace with live leg IVs below
    ts_working = [{"dte": p["dte"], "iv": p["iv"],
                   "expiry": p.get("expiry", ""), "source": "live"}
                  for p in ts_raw]

    # Inject live leg IVs — replace nearest ts point (within 3d) or insert
    def _inject_live_leg(pts, dte_val, iv_val, label):
        if dte_val is None or iv_val is None:
            return pts
        match_idx = next((i for i, p in enumerate(pts)
                          if abs(p["dte"] - dte_val) <= 3), None)
        entry = {"dte": dte_val, "iv": round(iv_val, 4),
                 "expiry": label, "source": "live_leg"}
        if match_idx is not None:
            pts[match_idx] = entry          # replace ts point with exact live value
        else:
            pts.append(entry)               # insert if no nearby point
        return pts

    ts_working = _inject_live_leg(ts_working, front_dte_int, front_iv_live,
                                  f"Front ({front_dte_live}d)")
    ts_working = _inject_live_leg(ts_working, back_dte_int,  back_iv_live,
                                  f"Back ({back_dte_live}d)")
    ts_working.sort(key=lambda p: p["dte"])

    ts_dtes  = [p["dte"]              for p in ts_working]
    ts_ivs   = [round(p["iv"], 4)     for p in ts_working]
    ts_exps  = [(p["expiry"][-5:] if len(p["expiry"]) >= 5 else p["expiry"])
                for p in ts_working]
    ts_point_colors = []
    for p in ts_working:
        if p.get("source") == "live_leg" and front_dte_int and abs(p["dte"] - front_dte_int) <= 3:
            ts_point_colors.append("#f87171")   # front leg — red
        elif p.get("source") == "live_leg" and back_dte_int and abs(p["dte"] - back_dte_int) <= 3:
            ts_point_colors.append("#4ade80")   # back leg — green
        elif front_dte_int and abs(p["dte"] - front_dte_int) <= 3:
            ts_point_colors.append("#f87171")
        elif back_dte_int and abs(p["dte"] - back_dte_int) <= 3:
            ts_point_colors.append("#4ade80")
        else:
            ts_point_colors.append("#60a5fa")   # other — blue

    # FF at each fill scenario — this is the key calculator output.
    # The FF is driven by IVs (not fill price directly), but we show whether
    # the signal is preserved by confirming live FF > 16%.
    ff_at_fills = {}
    if front_iv_live and back_iv_live and fwd_vol_live is not None:
        ff_at_fills["current"] = ff_live_val  # at current IVs
    # Fill price range as pct of mid — context for how much slippage eats into EV
    fill_mid_val2 = rec.get("fill_mid") or 0
    fill_agg_val  = rec.get("fill_aggressive") or 0
    fill_slippage_pct = ((fill_agg_val - fill_mid_val2) / fill_mid_val2 * 100
                         if fill_mid_val2 > 0 else None)

    return _HTML_TEMPLATE.format(
        ticker           = ticker,
        scan_date        = scan_date,
        badge_class      = "ready" if entry else "notready",
        badge_text       = "ENTRY READY" if entry else "NOT READY",
        strike           = rec.get("strike", "n/a"),
        front_expiry     = rec.get("front_expiry", "n/a"),
        back_expiry      = rec.get("back_expiry", "n/a"),
        front_dte        = front_dte_live,
        back_dte         = back_dte_live,
        front_call_bid   = rec.get("front_call_bid", 0),
        front_put_bid    = rec.get("front_put_bid", 0),
        back_call_ask    = rec.get("back_call_ask", 0),
        back_put_ask     = rec.get("back_put_ask", 0),
        net_debit_unfilled   = rec.get("net_debit_unfilled") or 0,
        net_debit_realistic  = rec.get("net_debit_realistic") or 0,
        net_debit_contract   = contract_cost,
        fill_mid             = fill_mid_val,
        fill_mid_contract    = fill_mid_contract,
        fill_aggressive      = rec.get("fill_aggressive") or 0,
        fill_passive         = rec.get("fill_passive") or 0,
        front_put_mid        = rec.get("front_put_mid") or 0,
        back_put_mid         = rec.get("back_put_mid") or 0,
        front_put_ask        = rec.get("front_put_bid", 0),   # sell at bid — show ask for passive row
        back_put_bid         = rec.get("back_put_ask", 0),    # buy at ask — show bid for passive row
        straddle_fill_aggressive     = rec.get("straddle_fill_aggressive") or 0,
        straddle_fill_mid            = rec.get("straddle_fill_mid") or 0,
        straddle_fill_passive        = rec.get("straddle_fill_passive") or 0,
        straddle_realistic           = (rec.get("straddle_fill_mid") or 0) * 1.5,
        straddle_contract_cost       = (rec.get("straddle_fill_mid") or 0) * 1.5 * 100,
        straddle_aggressive_contract = (rec.get("straddle_fill_aggressive") or 0) * 100,
        straddle_passive_contract    = (rec.get("straddle_fill_passive") or 0) * 100,
        str_agg_col      = str_agg_col,
        str_agg_pct      = str_agg_pct,
        str_mid_col      = str_mid_col,
        str_mid_pct      = str_mid_pct,
        str_pas_col      = str_pas_col,
        str_pas_pct      = str_pas_pct,
        fill_target_price= fill_target_price,
        fill_max_price   = fill_max_price,
        put_agg_col      = put_agg_col,
        put_agg_pct      = put_agg_pct,
        put_mid_col      = put_mid_col,
        put_mid_pct      = put_mid_pct,
        put_pas_col      = put_pas_col,
        put_pas_pct      = put_pas_pct,
        put_fill_target_price = put_fill_target_price,
        put_fill_max_price    = put_fill_max_price,
        front_straddle_bid   = rec.get("front_straddle_bid") or 0,
        front_straddle_mid   = rec.get("front_straddle_mid") or 0,
        front_straddle_ask   = rec.get("front_straddle_ask") or 0,
        back_straddle_bid    = rec.get("back_straddle_bid") or 0,
        back_straddle_mid    = rec.get("back_straddle_mid") or 0,
        back_straddle_ask    = rec.get("back_straddle_ask") or 0,
        back_iv_floor_pct= back_iv_floor_pct,
        back_iv_live_pct = back_iv_live_pct,
        front_iv_live_pct= front_iv_live_pct,
        fwd_vol_live_pct = fwd_vol_live_pct,
        iv_margin_bp     = iv_margin_bp,
        ff_pct           = ff_pct,
        ff_class         = "pos" if ff_pct >= 16 else "neg",
        ff_gate_source   = "<span style='color:#888;font-size:10px'>(live)</span>" if ff_live_val is not None else "<span style='color:#888;font-size:10px'>(Dolt EOD)</span>",
        ff_live_str      = ff_live_str,
        ff_live_class    = ff_live_class,
        ff_5d            = fmt_opt(rec.get("ff_5d_ago")),
        ff_10d           = fmt_opt(rec.get("ff_10d_ago")),
        trend            = rec.get("trend") or "n/a",
        trend_slope      = rec.get("trend_slope_5d") or 0,
        days_above       = rec.get("days_above_thresh") or "n/a",
        consec_above     = rec.get("consec_above_thresh") or "n/a",
        n_obs            = rec.get("n_observations") or len(series),
        vol_front        = rec.get("volume_30dte") or "n/a",
        vol_back         = rec.get("volume_60dte") or "n/a",
        entry_class      = "pos" if entry else "neg",
        entry_text       = "YES ✓" if entry else "NO ✗",
        not_ready_detail = not_ready_detail,
        labels_json      = json.dumps(labels),
        ff_data_json     = json.dumps(ff_vals),
        full_dates_json  = json.dumps([r["date"] for r in series]),
        ts_dtes_json     = json.dumps(ts_dtes),
        ts_ivs_json      = json.dumps(ts_ivs),
        ts_exps_json     = json.dumps(ts_exps),
        ts_pt_colors_json= json.dumps(ts_point_colors),
        # ML gates
        dps_dot          = dps_dot,
        dps_color        = dps_color,
        dps_str          = dps_str,
        comm_drag_dot    = comm_drag_dot,
        comm_drag_color  = comm_drag_color,
        comm_drag_str    = comm_drag_str,
        ivhv_dot         = ivhv_dot,
        ivhv_color       = ivhv_color,
        ivhv_str         = ivhv_str,
        ff_gate_dot      = ff_gate_dot,
        ff_gate_color    = ff_gate_color,
        ff_gate_str      = ff_gate_str,
        dte_dot          = dte_dot,
        dte_color        = dte_color,
        dte_str          = dte_str,
        ivr_dot          = ivr_dot,
        ivr_color        = ivr_color,
        ivr_str          = ivr_str,
        ml_verdict_class  = ml_verdict_class,
        ml_verdict_text   = ml_verdict_text,
        ml_score          = ml_score,
        ml_score_class    = ml_score_class,
        ml_put_score      = ml_put_score,
        ml_put_score_class= ml_put_score_class,
        # FF quality + composite score (with backtest annotations)
        ff_qual_str      = ff_qual_str,
        ff_qual_color    = ff_qual_color,
        comp_str         = comp_str,
        comp_color       = comp_color,
        # ML data source indicator
        ml_data_src       = ml_data_src,
        ml_data_src_color = ml_data_src_color,
        ml_data_src_title = ml_data_src_title,
        # Exit rules
        ff_exit_trigger  = ff_exit_trigger,
        tp40_str         = tp40_str,
    )


# ── ML v2 scorer (LR bin_bigwin, 177k-trade dataset, AUC 0.795) ──────────────

_ML_V2_MODEL:     dict | None = None
_ML_PUT_V2_MODEL: dict | None = None


def _load_ml_v2() -> dict | None:
    global _ML_V2_MODEL
    if _ML_V2_MODEL is None:
        path = Path(__file__).parent / "ff_ml_lr_coefs_v2.json"
        if path.exists():
            with open(path) as fh:
                _ML_V2_MODEL = json.load(fh)
        else:
            print(f"WARNING: {path} not found — ML v2 scoring disabled", file=sys.stderr)
    return _ML_V2_MODEL


def _load_ml_put_v2() -> dict | None:
    global _ML_PUT_V2_MODEL
    if _ML_PUT_V2_MODEL is None:
        path = Path(__file__).parent / "ff_ml_lr_coefs_put_v1.json"
        if path.exists():
            with open(path) as fh:
                _ML_PUT_V2_MODEL = json.load(fh)
        else:
            print(f"WARNING: {path} not found — put ML scoring disabled", file=sys.stderr)
    return _ML_PUT_V2_MODEL


def _ml_score_v2(rec: dict) -> float:
    """
    Score a priced trade rec using the LR bin_bigwin v2 model.
    Returns P(hold-return > +50%) in [0,1]. Falls back to 0.0 if model missing.

    Uses live pricing fields when available: ff_live, back_iv_live, back_straddle_mid,
    straddle_fill_mid (= entry_debit), front_expiry, back_expiry.
    """
    model = _load_ml_v2()
    if model is None:
        return 0.0

    # When any leg is unquoted (stale), live IVs/prices are unreliable — use Dolt fallbacks.
    _stale = rec.get("stale_data", False)
    ff     = (rec.get("ff_live") if (not _stale and rec.get("ff_live") is not None)
              else None) or rec.get("ff_pct")
    b_iv   = (None if _stale else rec.get("back_iv_live"))   or rec.get("back_iv")
    bstrad = (None if _stale else rec.get("back_straddle_mid")) or rec.get("back_straddle")
    debit  = (None if _stale else rec.get("straddle_fill_mid")) or rec.get("entry_debit")
    ticker = rec.get("ticker", "")

    fe_str = rec.get("front_expiry", "")
    be_str = rec.get("back_expiry", "")
    try:
        t_front = (date.fromisoformat(fe_str) - date.today()).days
    except (ValueError, TypeError):
        t_front = None
    try:
        t_back = (date.fromisoformat(be_str) - date.today()).days
    except (ValueError, TypeError):
        t_back = None

    if any(v is None for v in [ff, b_iv, bstrad, debit, t_front, t_back]):
        return 0.0
    if bstrad <= 0:
        return 0.0

    t_fwd = t_back - t_front
    max_p = bstrad * math.sqrt(t_fwd / max(t_back, 1)) - debit

    d     = date.today()
    month = d.month
    quarter = (month - 1) // 3 + 1

    _KNOWN_ETFS = {
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLP", "XLU", "XLB", "XLRE", "GLD", "SLV", "TLT", "IEF", "HYG",
        "LQD", "EEM", "EFA", "VNQ", "DBC", "USO", "GDX",
    }

    features = {
        "entry_ff":          ff,
        "entry_ff_sq":       ff ** 2,
        "entry_ff_pos":      max(ff, 0.0),
        "entry_debit":       debit,
        "log_debit":         math.log(max(debit, 0.01)),
        "back_straddle":     bstrad,
        "log_back_straddle": math.log(max(bstrad, 0.01)),
        "debit_ratio":       debit / bstrad,
        "t_front":           float(t_front),
        "t_back":            float(t_back),
        "t_fwd":             float(t_fwd),
        "t_fwd_ratio":       t_fwd / max(t_back, 1),
        "max_profit_ratio":  max_p / bstrad,
        "iv_proxy":          b_iv * math.sqrt(2.0 / math.pi),
        "ff_x_tfwd":         ff * math.sqrt(max(t_fwd, 1)),
        "month_sin":         math.sin(2 * math.pi * month / 12),
        "month_cos":         math.cos(2 * math.pi * month / 12),
        "quarter":           float(quarter),
        "is_etf":            1.0 if ticker in _KNOWN_ETFS else 0.0,
    }

    coefs = model["coefs"]
    means = model["scaler_mean"]
    stds  = model["scaler_std"]
    logit = model["intercept"]
    for feat, val in features.items():
        std = stds.get(feat, 1.0)
        mu  = means.get(feat, 0.0)
        z   = (val - mu) / std if std > 0 else 0.0
        logit += coefs.get(feat, 0.0) * z

    return 1.0 / (1.0 + math.exp(-logit))


def _ml_score_put_v2(rec: dict) -> float:
    """
    Score a put-calendar rec using the put LR v1 model (put ≈ straddle/2 at ATM).
    Uses fill_mid (put fill) and back_put_mid instead of straddle equivalents.
    Returns P(bigwin) in [0,1].
    """
    model = _load_ml_put_v2()
    if model is None:
        return 0.0

    # When any leg is unquoted (stale), live IVs/prices are unreliable — use Dolt fallbacks.
    _stale = rec.get("stale_data", False)
    ff     = (rec.get("ff_live") if (not _stale and rec.get("ff_live") is not None)
              else None) or rec.get("ff_pct")
    b_iv   = (None if _stale else rec.get("back_iv_live")) or rec.get("back_iv")
    bstrad = (None if _stale else rec.get("back_put_mid")) or (rec.get("back_straddle") / 2 if rec.get("back_straddle") else None)
    debit  = (None if _stale else rec.get("fill_mid"))     or (rec.get("entry_debit")   / 2 if rec.get("entry_debit")   else None)
    ticker = rec.get("ticker", "")

    fe_str = rec.get("front_expiry", "")
    be_str = rec.get("back_expiry", "")
    try:
        t_front = (date.fromisoformat(fe_str) - date.today()).days
    except (ValueError, TypeError):
        t_front = None
    try:
        t_back = (date.fromisoformat(be_str) - date.today()).days
    except (ValueError, TypeError):
        t_back = None

    if any(v is None for v in [ff, b_iv, bstrad, debit, t_front, t_back]):
        return 0.0
    if bstrad <= 0:
        return 0.0

    t_fwd = t_back - t_front
    max_p = bstrad * math.sqrt(t_fwd / max(t_back, 1)) - debit

    d     = date.today()
    month = d.month
    quarter = (month - 1) // 3 + 1

    _KNOWN_ETFS = {
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLP", "XLU", "XLB", "XLRE", "GLD", "SLV", "TLT", "IEF", "HYG",
        "LQD", "EEM", "EFA", "VNQ", "DBC", "USO", "GDX",
    }

    features = {
        "entry_ff":          ff,
        "entry_ff_sq":       ff ** 2,
        "entry_ff_pos":      max(ff, 0.0),
        "entry_debit":       debit,
        "log_debit":         math.log(max(debit, 0.01)),
        "back_straddle":     bstrad,
        "log_back_straddle": math.log(max(bstrad, 0.01)),
        "debit_ratio":       debit / bstrad,
        "t_front":           float(t_front),
        "t_back":            float(t_back),
        "t_fwd":             float(t_fwd),
        "t_fwd_ratio":       t_fwd / max(t_back, 1),
        "max_profit_ratio":  max_p / bstrad,
        "iv_proxy":          b_iv * math.sqrt(2.0 / math.pi) / 2,  # put IV proxy ≈ straddle/2
        "ff_x_tfwd":         ff * math.sqrt(max(t_fwd, 1)),
        "month_sin":         math.sin(2 * math.pi * month / 12),
        "month_cos":         math.cos(2 * math.pi * month / 12),
        "quarter":           float(quarter),
        "is_etf":            1.0 if ticker in _KNOWN_ETFS else 0.0,
    }

    coefs = model["coefs"]
    means = model["scaler_mean"]
    stds  = model["scaler_std"]
    logit = model["intercept"]
    for feat, val in features.items():
        std = stds.get(feat, 1.0)
        mu  = means.get(feat, 0.0)
        z   = (val - mu) / std if std > 0 else 0.0
        logit += coefs.get(feat, 0.0) * z

    return 1.0 / (1.0 + math.exp(-logit))


def _ml_score_for_rec(rec: dict) -> float:
    """ML win probability for index table (0–100 scale). Put model is primary."""
    return round(_ml_score_put_v2(rec) * 100, 1)


_INDEX_STYLE = """
  body  { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#0f1117;color:#e0e0e0;margin:0;padding:24px; }
  h1,h2 { color:#fff;margin-bottom:4px; }
  .sub  { color:#888;font-size:14px;margin-bottom:20px; }
  table { width:100%;border-collapse:collapse;font-size:14px; }
  th    { color:#9ca3af;font-weight:500;padding:8px 12px;text-align:left;
          border-bottom:2px solid #2d3148;white-space:nowrap; }
  td    { padding:8px 12px;border-bottom:1px solid #1e2130; }
  tr:hover td { background:#1e2230; }
  .note { color:#555;font-size:12px;margin-top:16px; }
"""

_INDEX_THEAD = """
  <thead>
    <tr>
      <th>Ticker</th><th>FF%</th><th>Put P(bigwin) ★</th><th>Straddle P(bigwin)</th><th>Status</th>
      <th>FF qual</th><th>Composite</th>
      <th>Put mid</th><th>Straddle mid</th><th>IV/HV</th>
      <th>IV Rank</th><th>Front DTE</th><th>Not-ready reason</th>
    </tr>
  </thead>"""


def _candidate_table_rows(results: list[dict], link_prefix: str = "") -> str:
    """<tr> rows for the candidate summary table, sorted ready→ML score.

    link_prefix: prepended to each TICKER_report.html href (e.g. "2026-06-12/" for master index).
    """
    rows_sorted = sorted(
        results,
        key=lambda r: (0 if r.get("entry_ready") else 1, -_ml_score_for_rec(r))
    )

    def _dte(expiry_str):
        try: return (date.fromisoformat(expiry_str) - date.today()).days
        except: return "?"

    html = ""
    for rec in rows_sorted:
        ticker  = rec.get("ticker", "?")
        ff      = rec.get("ff_pct") or 0
        ml      = _ml_score_for_rec(rec)
        ml_put  = round(_ml_score_put_v2(rec) * 100, 1)
        ready   = rec.get("entry_ready", False)
        str_mid = rec.get("straddle_fill_mid") or 0
        put_mid = rec.get("fill_mid") or 0
        ivhv    = rec.get("iv_hv_ratio")
        ivr     = rec.get("iv_rank_20d")
        dte     = _dte(rec.get("front_expiry", ""))
        reasons = "; ".join(rec.get("not_ready_reasons") or []) or "—"

        badge_bg  = "#1a4731" if ready else "#1c1f2b"
        badge_col = "#4ade80" if ready else "#f87171"
        badge_txt = "READY ✓" if ready else "NOT READY"
        ml_col     = "#4ade80" if ml     >= 45 else ("#f59e0b" if ml     >= 35 else "#f87171")
        ml_put_col = "#4ade80" if ml_put >= 45 else ("#f59e0b" if ml_put >= 35 else "#f87171")
        row_style  = 'style="background:#1a3520"' if ready else ""
        # Flag Dolt-only ML scores (stale or live chain not available)
        _live = (rec.get("back_iv_live") is not None
                 and not rec.get("stale_data", False)
                 and rec.get("fill_mid") is not None)
        ml_src_badge = "" if _live else ' <span style="color:#f59e0b;font-size:10px" title="Scored on Dolt EOD data — re-run intraday for live score">Dolt</span>'

        # ff_quality_pts: sweet spot is 5-8 (57.7% OOS win, +123% avg)
        ffq_raw = rec.get("ff_quality_pts")
        ffq     = float(ffq_raw) if ffq_raw is not None else None
        if ffq is not None:
            ffq_col  = "#4ade80" if 5 <= ffq <= 8 else ("#f59e0b" if ffq > 8 else "#9ca3af")
            ffq_cell = f'<td style="color:{ffq_col};font-weight:{"700" if 5<=ffq<=8 else "400"}">{ffq:.0f}{"★" if 5<=ffq<=8 else ""}</td>'
        else:
            ffq_cell = '<td style="color:#555">—</td>'

        # composite_score: 80+ tier underperforms (48% win); 0-60 outperforms (54% win)
        cs_raw = rec.get("composite_score")
        cs     = float(cs_raw) if cs_raw is not None else None
        if cs is not None:
            cs_col  = "#f59e0b" if cs >= 80 else ("#9ca3af" if cs >= 60 else "#4ade80")
            cs_tip  = "⚠" if cs >= 80 else ("" if cs >= 60 else "✓")
            cs_cell = f'<td style="color:{cs_col}">{cs:.0f}{cs_tip}</td>'
        else:
            cs_cell = '<td style="color:#555">—</td>'

        html += f"""
    <tr {row_style}>
      <td><a href="{link_prefix}{ticker}_report.html" style="color:#60a5fa;font-weight:600">{ticker}</a></td>
      <td style="color:#e0e0e0">{ff:+.1f}%</td>
      <td style="color:{ml_put_col};font-weight:700">{ml_put:.0f}%{ml_src_badge}</td>
      <td style="color:{ml_col};font-weight:700">{ml:.0f}%{ml_src_badge}</td>
      <td><span style="background:{badge_bg};color:{badge_col};padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">{badge_txt}</span></td>
      {ffq_cell}
      {cs_cell}
      <td>${put_mid:.2f}</td>
      <td>${str_mid:.2f}</td>
      <td>{f"{ivhv:.2f}×" if ivhv else "n/a"}</td>
      <td>{f"{ivr:.0f}%" if ivr else "n/a"}</td>
      <td>{dte}d</td>
      <td style="color:#6b7280;font-size:12px">{reasons[:60]}{"…" if len(reasons) > 60 else ""}</td>
    </tr>"""
    return html


def generate_index_html(results: list[dict], scan_date: str) -> str:
    """Per-date summary page: reports/YYYY-MM-DD/index.html."""
    row_html = _candidate_table_rows(results)
    n_ready  = sum(1 for r in results if r.get("entry_ready"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FF Scanner — {scan_date}</title>
<style>{_INDEX_STYLE}</style>
</head>
<body>
<h1>FF Scanner <span style="color:#888;font-size:18px">— {scan_date}</span></h1>
<div class="sub">{len(results)} candidates priced &nbsp;|&nbsp;
  <span style="color:#4ade80">{n_ready} entry-ready</span>
  &nbsp;|&nbsp; sorted by: ready ↓ then ML score ↓
</div>
<table>{_INDEX_THEAD}
  <tbody>{row_html}
  </tbody>
</table>
<p class="note">ML Win% = logistic regression (7 features, 3,813 OOS trades 2024-2025, transfer AUC 0.698 on 2020-2022). FF qual ★ = sweet spot 5-8 (57.7% win, +123% avg). Composite ⚠ = 80+ tier underperforms (48% win). ✓ = 0-60 tier outperforms (54% win). Click ticker to open full report.</p>
</body>
</html>"""


def generate_master_index_html(latest_results: list[dict], latest_date: str) -> str:
    """Top-level reports/index.html: latest scan at top, past date links below."""
    row_html = _candidate_table_rows(latest_results, link_prefix=f"{latest_date}/")
    n_ready  = sum(1 for r in latest_results if r.get("entry_ready"))

    # Discover past date directories
    past_rows = ""
    if REPORTS_DIR.exists():
        date_dirs = sorted(
            [d for d in REPORTS_DIR.iterdir()
             if d.is_dir() and re.match(r'^\d{4}-\d{2}-\d{2}$', d.name) and d.name != latest_date],
            key=lambda d: d.name,
            reverse=True,
        )
        for d in date_dirs:
            n_reports = len(list(d.glob("*_report.html")))
            # Flat new structure; graceful fallback for pre-migration dates
            if (d / "index.html").exists():
                link = f"{d.name}/index.html"
            elif (d / "ready" / "index.html").exists():
                link = f"{d.name}/ready/index.html"
                n_reports = len(list((d / "ready").glob("*_report.html")))
            elif (d / "ready").exists():
                link = f"{d.name}/ready/"
                n_reports = len(list((d / "ready").glob("*_report.html")))
            else:
                link = f"{d.name}/"
            past_rows += f"""
    <tr>
      <td><a href="{link}" style="color:#60a5fa;font-weight:600">{d.name}</a></td>
      <td style="color:#9ca3af;padding-left:24px">{n_reports} candidates</td>
    </tr>"""

    past_section = f"""
<h2 style="margin-top:48px">Past scans</h2>
<table style="width:auto">
  <thead>
    <tr>
      <th>Date</th>
      <th style="padding-left:24px">Candidates</th>
    </tr>
  </thead>
  <tbody>{past_rows}
  </tbody>
</table>""" if past_rows else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FF Scanner — Reports</title>
<style>{_INDEX_STYLE}</style>
</head>
<body>
<h1>FF Scanner</h1>
<h2 style="color:#60a5fa;margin-bottom:2px">Latest scan: {latest_date}</h2>
<div class="sub">{len(latest_results)} candidates priced &nbsp;|&nbsp;
  <span style="color:#4ade80">{n_ready} entry-ready</span>
  &nbsp;|&nbsp; sorted by: ready ↓ then ML score ↓
</div>
<table>{_INDEX_THEAD}
  <tbody>{row_html}
  </tbody>
</table>
<p class="note">ML Win% = logistic regression (7 features, 3,813 OOS trades 2024-2025, transfer AUC 0.698 on 2020-2022). FF qual ★ = sweet spot 5-8 (57.7% win, +123% avg). Composite ⚠ = 80+ tier underperforms (48% win). ✓ = 0-60 tier outperforms (54% win). Click ticker to open full report.</p>
{past_section}
</body>
</html>"""


def write_one_report(rec: dict, scan_date: str) -> Path:
    """Write a single ticker's report. Called as each candidate finishes scoring."""
    date_dir = REPORTS_DIR / scan_date
    date_dir.mkdir(parents=True, exist_ok=True)
    ticker = rec.get("ticker", "UNKNOWN")
    path   = date_dir / f"{ticker}_report.html"
    path.write_text(generate_html_report(rec, scan_date), encoding="utf-8")
    return path


def write_indexes(results: list[dict], scan_date: str):
    """(Re)write the per-date index and master dashboard from results so far.
    Safe to call repeatedly — the tables sort themselves from the list given."""
    if not results:
        return
    date_dir = REPORTS_DIR / scan_date
    date_dir.mkdir(parents=True, exist_ok=True)
    (date_dir / "index.html").write_text(
        generate_index_html(results, scan_date), encoding="utf-8"
    )
    (REPORTS_DIR / "index.html").write_text(
        generate_master_index_html(results, scan_date), encoding="utf-8"
    )


def write_html_reports(results: list[dict], scan_date: str):
    """Write HTML reports + per-date index to reports/YYYY-MM-DD/, then update master index."""
    date_dir = REPORTS_DIR / scan_date
    written = [write_one_report(rec, scan_date) for rec in results]
    write_indexes(results, scan_date)

    if written:
        ready_count = sum(1 for r in results if r.get("entry_ready"))
        print(f"\nHTML reports → {date_dir}")
        print(f"  {len(written)} reports ({ready_count} entry-ready, {len(written)-ready_count} priced/not-ready)")
        print(f"  {date_dir}/index.html  ← today's summary")
        print(f"  {REPORTS_DIR}/index.html  ← master dashboard")
    return written


# ── Output ────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "scan_date", "ticker", "ff_pct", "front_expiry", "back_expiry", "strike",
    "front_call_bid", "front_put_bid", "back_call_ask", "back_put_ask",
    "net_debit_unfilled", "net_debit_realistic",
    "max_loss", "max_profit", "entry_ready",
    "bid_ask_width_front", "bid_ask_width_back",
    "volume_30dte", "volume_60dte",
    "stale_data", "wide_spread_flag",
    # Trend fields (populated when --universe provided)
    "current_ff_dolt", "ff_5d_ago", "ff_10d_ago",
    "trend", "trend_slope_5d", "days_above_thresh", "consec_above_thresh",
]


def print_recommendations(recs: list[dict]):
    ready = [r for r in recs if r.get("entry_ready")]
    print(f"\nEntry-ready candidates: {len(ready)} / {len(recs)} priced")

    if ready:
        print("\n=== TRADE RECOMMENDATIONS ===")
        hdr = f"{'Ticker':<7} {'FF%':>7}  {'Strike':>8}  {'Front':>11}  {'Back':>11}  {'Debit':>8}  {'Real':>8}  {'MaxLoss':>8}"
        print(hdr)
        for r in ready:
            print(
                f"  {r['ticker']:<5}  {r['ff_pct']:>+6.1f}%"
                f"  {r['strike']:>8.2f}"
                f"  {r['front_expiry']:>11}  {r['back_expiry']:>11}"
                f"  ${r['net_debit_unfilled']:>6.2f}"
                f"  ${r['net_debit_realistic']:>6.2f}"
                f"  ${r['max_loss'] or 0:>6.0f}"
            )
        print()
        for r in ready:
            print(
                f"TICKET — {r['ticker']} | Strike {r['strike']}"
                f" | SELL {r['front_expiry']} straddle (call+put)"
                f" | BUY {r['back_expiry']} straddle (call+put)"
                f" | Limit ${r['net_debit_realistic']:.2f} debit (mid, work up)"
            )
    else:
        print("\nNo entry-ready candidates.")

        if recs:
            print("\n(All candidates priced but failed entry gate:)")
            for r in recs:
                reasons = []
                if r.get("stale_data"):
                    reasons.append("stale data")
                if (r.get("net_debit_unfilled") or 0) <= 0:
                    reasons.append("negative/zero debit (data issue)")
                if r.get("wide_spread_flag"):
                    reasons.append("wide spread")
                if (r.get("volume_30dte") or 0) < MIN_VOLUME_PER_LEG:
                    reasons.append(f"low front vol ({r.get('volume_30dte', 0)})")
                if (r.get("volume_60dte") or 0) < MIN_VOLUME_PER_LEG:
                    reasons.append(f"low back vol ({r.get('volume_60dte', 0)})")
                print(f"  {r['ticker']:5s} FF={r['ff_pct']:+.1f}%: {', '.join(reasons)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FF Trade Scanner — price calendar spread recommendations")
    parser.add_argument("--no-ib", action="store_true", help="Skip IB gateway, use yfinance only")
    parser.add_argument("--scan-file", metavar="PATH", help="Use existing FF scan CSV (skip re-scan)")
    parser.add_argument("--universe", metavar="FILE",
                        help="Candidates CSV or universe JSON from ff_universe_scan.py (latest_candidates.csv recommended)")
    parser.add_argument("--ff-min", type=float, default=FF_MIN_DEFAULT,
                        help=f"Minimum FF%% to price (default {FF_MIN_DEFAULT})")
    parser.add_argument("--date", default=date.today().isoformat(), help="Scan date for output filename")
    parser.add_argument("--good-only", action="store_true",
                        help=f"Price only tickers with a ready track record "
                             f"(>={GOOD_MIN_SEEN} scans, >={GOOD_MIN_RATE * 100:.0f}%% ready-rate). "
                             f"Fast path — does not let new tickers earn a record.")
    parser.add_argument("--no-stats", action="store_true",
                        help="Do not update ready_stats.json with this run's outcomes")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"FF Trade Scanner — {args.date}")
    print(f"FF threshold: {args.ff_min}% | Source: {'yfinance' if args.no_ib else 'IB (yfinance fallback)'}")
    if args.universe:
        print(f"Universe: {args.universe} (FF trend data enabled)")

    # Load universe trend data if provided.
    # If the universe file is a CSV (candidates file), also use it as the
    # candidate source — no need to re-run ff_scanner.py against the watchlist.
    universe = load_universe(args.universe) if args.universe else {}

    universe_is_csv = args.universe and (args.universe.endswith(".csv") or args.universe.endswith(".json"))
    if universe_is_csv and not args.scan_file:
        # Pull candidates directly from the universe CSV, with staleness guards
        from datetime import date as _today_cls
        today = _today_cls.today()
        min_dte = 20  # Minimum DTE from today — matches strategy front window floor
        skipped_dte = skipped_ff100 = 0
        candidates = []
        refreshed_expiry = 0
        for t, data in universe.items():
            ff = (data.get("stats", {}).get("current_ff") or 0)
            if ff < args.ff_min:
                continue
            if ff >= 99.9:  # fwd_vol=0 artifact
                skipped_ff100 += 1
                continue
            fe_str = data.get("front_expiry")
            if fe_str:
                try:
                    fe = _today_cls.fromisoformat(fe_str)
                    if (fe - today).days < min_dte:
                        # Stored expiry is stale — re-select from yfinance
                        new_front, new_back = pick_live_expiry(t)
                        if new_front and new_back:
                            data = dict(data)
                            data["front_expiry"] = new_front
                            data["back_expiry"]  = new_back
                            data["front_dte"]    = (_today_cls.fromisoformat(new_front) - today).days
                            data["back_dte"]     = (_today_cls.fromisoformat(new_back)  - today).days
                            refreshed_expiry += 1
                        else:
                            skipped_dte += 1
                            continue
                except ValueError:
                    pass
            candidates.append({"ticker": t, **data})
        if refreshed_expiry:
            print(f"  Refreshed expiry for {refreshed_expiry} candidates (stored expiry was stale)")
        if skipped_dte:
            print(f"  Skipped {skipped_dte} candidates: no valid expiry in {_FRONT_DTE_WINDOW} DTE window")
        if skipped_ff100:
            print(f"  Skipped {skipped_ff100} candidates: FF=100% artifact (fwd_vol=0)")
        # Flatten stats fields so price_candidate can read ff_pct, front_expiry, back_expiry
        for c in candidates:
            c.setdefault("ff_pct", c.get("stats", {}).get("current_ff"))
        print(f"Candidates from universe CSV: {len(candidates)} with FF≥{args.ff_min}%")
    else:
        candidates = load_or_run_scan(args.scan_file, args.no_ib, args.ff_min)
    if not candidates:
        print(f"\nNo FF >= {args.ff_min}% candidates to price.")
        return 0

    # Ready-state history: prioritise tickers that actually clear the entry gate.
    ready_stats = load_ready_stats()
    if args.good_only:
        good = [c for c in candidates if is_good_ticker(ready_stats, c["ticker"])]
        skipped = len(candidates) - len(good)
        if not good:
            print(f"\n--good-only: no candidate has a ready track record yet "
                  f"(needs ≥{GOOD_MIN_SEEN} scans at ≥{GOOD_MIN_RATE:.0%}). "
                  f"Run without --good-only to build one.")
            return 0
        print(f"  --good-only: {len(good)} proven candidates, {skipped} skipped")
        candidates = good

    # Best ready-rate first, FF as tie-break. Priority order flows through the IB
    # batch and the pricing pool, so good tickers get reports written first.
    candidates.sort(key=lambda c: (-ready_rate(ready_stats, c["ticker"]),
                                   -(c.get("ff_pct") or 0)))
    n_proven = sum(1 for c in candidates if is_good_ticker(ready_stats, c["ticker"]))
    if n_proven:
        head = ", ".join(c["ticker"] for c in candidates[:5])
        print(f"  Priority: {n_proven} proven-ready ticker(s) first — {head}")

    # IB batch pricing: one subprocess, one connection, all tickers at once.
    # Falls back to yfinance per-ticker if IB unavailable.
    ib_batch_data: dict[str, dict] = {}
    if not args.no_ib:
        print(f"Connecting to IB for batch pricing ({len(candidates)} tickers)...")
        ib_batch_data = price_candidates_via_ib(candidates)
        n_ok = sum(1 for v in ib_batch_data.values() if v.get("ok"))
        if n_ok:
            print(f"  IB batch: {n_ok}/{len(ib_batch_data)} tickers priced successfully")
        else:
            print("  IB unavailable — falling back to yfinance per-ticker")

    mode = "IB batch" if ib_batch_data else "yfinance per-ticker"
    print(f"\nPricing {len(candidates)} candidate(s) [{mode}]")

    def _price_one(cand: dict) -> dict | None:
        ticker = cand["ticker"]
        raw    = ib_batch_data.get(ticker)
        if raw and raw.get("ok"):
            rec = build_rec_from_ib(cand, raw)
            if rec is None:
                # Shouldn't happen after removing the return-None kill gate, but guard anyway
                return None
        else:
            if ib_batch_data and not raw:
                # IB ran but returned nothing for this ticker — log why
                ib_err = (ib_batch_data.get(ticker) or {}).get("error", "not returned by IB batch")
                print(f"  {ticker:5s} IB batch miss ({ib_err}) — falling back to yfinance")
            elif raw and not raw.get("ok"):
                print(f"  {ticker:5s} IB batch error ({raw.get('error','?')}) — falling back to yfinance")
            rec = price_candidate(
                ticker=ticker,
                front_expiry=cand["front_expiry"],
                back_expiry=cand["back_expiry"],
                ff_pct=float(cand["ff_pct"]),
                no_ib=args.no_ib,
            )
        if rec and universe:
            rec = enrich_with_trend(rec, universe)
        return rec

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fut_to_cand = {pool.submit(_price_one, c): c for c in candidates}
        for fut in concurrent.futures.as_completed(fut_to_cand):
            cand  = fut_to_cand[fut]
            ticker = cand["ticker"]
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  {ticker:5s} ERROR: {e}")
                continue
            if rec:
                results.append(rec)
                status = "READY" if rec["entry_ready"] else "not ready"
                reasons = rec.get("not_ready_reasons") or []
                reason_str = f" [{reasons[0]}]" if reasons and not rec["entry_ready"] else ""
                print(f"  {ticker:5s} FF={rec.get('ff_pct',0):+.1f}% debit=${rec['net_debit_unfilled']:.2f} → {status}{reason_str}")
                # Publish this ticker's report immediately so entry-ready names are
                # reviewable while the rest of the book is still pricing.
                if universe:
                    try:
                        write_one_report(rec, args.date)
                        write_indexes(results, args.date)
                    except Exception as e:
                        print(f"  {ticker:5s} report write failed: {e}", file=sys.stderr)
            else:
                print(f"  {ticker:5s} no data")

    # Rank by: entry_ready desc, FF% desc, max_profit desc, bid_ask_width_front asc
    results.sort(key=lambda r: (
        not r.get("entry_ready"),
        -(r.get("ff_pct") or 0),
        -(r.get("max_profit") or 0),
        r.get("bid_ask_width_front") or 999,
    ))

    out_path = OUTPUT_DIR / f"{args.date}_trade_recommendations.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\nSaved: {out_path}")
    print_recommendations(results)

    # Reports were streamed per ticker; refresh indexes once more so the final
    # ordering reflects the complete result set.
    if universe:
        write_indexes(results, args.date)
        date_dir    = REPORTS_DIR / args.date
        ready_count = sum(1 for r in results if r.get("entry_ready"))
        print(f"\nHTML reports → {date_dir}")
        print(f"  {len(results)} reports ({ready_count} entry-ready, {len(results)-ready_count} priced/not-ready)")
        print(f"  {date_dir}/index.html  ← today's summary")
        print(f"  {REPORTS_DIR}/index.html  ← master dashboard")

    # Record ready outcomes so tomorrow's run can prioritise. Skipped for
    # --good-only, which sees a biased subset and would inflate the record.
    if not args.no_stats and not args.good_only:
        update_ready_stats(ready_stats, results, args.date)
        save_ready_stats(ready_stats, args.date)
        proven = sorted(
            (t for t in ready_stats["tickers"] if is_good_ticker(ready_stats, t)),
            key=lambda t: -ready_stats["tickers"][t]["ready_rate"],
        )
        print(f"\nReady stats → {READY_STATS_FILE.name} "
              f"({len(proven)} proven-ready ticker(s) tracked)")
        if proven:
            print("  " + ", ".join(
                f"{t} {ready_stats['tickers'][t]['ready_rate']:.0%}" for t in proven[:10]
            ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
