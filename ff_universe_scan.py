#!/usr/bin/env python3
"""
FF Universe Scanner — overnight Dolt-based batch scan.

Scans ALL tickers in the Dolt options DB (2,274 names) using one batch query
per date. Computes FF over the last N trading days per ticker, then exports:
  - universe/YYYY-MM-DD_universe.json     Full FF time series per ticker
  - universe/YYYY-MM-DD_candidates.csv   FF > ff_min tickers (sorted by FF desc)
  - universe/latest_candidates.csv       Copy of latest candidates for intraday use

The candidates CSV feeds ff_scanner.py --universe for fast intraday scanning.

Runtime: ~1 query per date (batch), ~20 dates = ~20 queries total. Typical runtime
2–5 min for the full 2,274-ticker universe, depending on rows per date.

Dolt DB update timing: ~11:00 Paris time (05:00 ET). Run this scan after that window
so the previous day's data is fully available. Ideal: cron at 11:30 Paris / 05:30 ET.

Tickers with fewer than --min-iv-rows valid IV rows on a given date are skipped
(natural liquidity filter — untradeable chains don't produce FF signals anyway).

Usage:
    python ff_universe_scan.py                    # all Dolt tickers, ff_min=15
    python ff_universe_scan.py --date 2025-06-05  # specific date
    python ff_universe_scan.py --ff-min 12        # lower threshold
    python ff_universe_scan.py --days 30          # 30-day look-back
    python ff_universe_scan.py --min-iv-rows 5    # looser liquidity filter
    python ff_universe_scan.py --tickers AAPL NVDA AMD  # override (testing)

Dolt DB: localhost:3307
    options.option_chain  — IV data by (date, ticker, expiry, strike)
    stocks.ohlcv          — close prices + trading date calendar
"""

import argparse
import csv
import json
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

FRONT_DTE_WINDOW = (22, 45)
BACK_DTE_WINDOW  = (46, 80)

OUTPUT_DIR = Path(__file__).parent / "universe"

KNOWN_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLP", "XLU", "XLB", "XLRE", "GLD", "SLV", "TLT", "IEF", "HYG",
    "LQD", "EEM", "EFA", "VNQ", "DBC", "USO", "GDX",
}




def conn(db_name=None, **kwargs):
    """Backward-compatible connection factory — delegates to earningsvol_conn().

    The old pymysql API required a database name and read_timeout; both
    are now handled transparently by the PostgreSQL connection.
    """
    return earningsvol_conn()


# ── DB helpers ────────────────────────────────────────────────────────────────


OHLCV_CHUNK = 100   # tickers per OHLCV batch query (avoids huge IN clauses)


def find_latest_data_date(c, start_from: date, max_lookback_days: int = 10) -> date | None:
    """
    Find the most recent date in Dolt that has options data.
    Starts from start_from and walks backward up to max_lookback_days,
    skipping weekends.
    Returns the date or None if no data found in window.
    """
    for i in range(max_lookback_days):
        check_date = start_from - timedelta(days=i)
        if check_date.weekday() >= 5:
            continue
        with c.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM option_chain WHERE date=%s AND vol IS NOT NULL AND vol > 0 LIMIT 1",
                (check_date.isoformat(),),
            )
            count = cur.fetchone()[0]
        if count > 0:
            return check_date
    return None


def get_trading_dates(n: int, up_to: date) -> list[date]:
    """Return the last n business days up to and including up_to.
    Uses calendar arithmetic (Mon-Fri) — holidays will produce empty DB results
    in the main loop and are silently skipped there.  Fetches n+10 candidates
    so ~2 weeks of holidays still yield enough data points.
    """
    from datetime import timedelta
    candidates = []
    d = up_to
    needed = n + 10  # small buffer for market holidays
    while len(candidates) < needed:
        if d.weekday() < 5:  # Mon–Fri
            candidates.append(d)
        d -= timedelta(days=1)
    return sorted(candidates)  # ascending


def get_all_spots(c, d: date, max_lookback: int = 3) -> dict[str, float]:
    """Fetch all closing prices for a given date. Falls back up to max_lookback
    business days earlier if stocks.ohlcv lags behind options data."""
    for lag in range(max_lookback + 1):
        check = d - timedelta(days=lag)
        if check.weekday() >= 5:  # skip weekends
            continue
        with c.cursor() as cur:
            cur.execute("SELECT act_symbol, close FROM ohlcv WHERE date=%s", (check.isoformat(),))
            rows = {r[0]: float(r[1]) for r in cur.fetchall() if r[1] is not None}
        if rows:
            if lag > 0:
                print(f"  stocks.ohlcv: no data for {d}, using {check} (lag {lag}d)", flush=True)
            return rows
    return {}


def get_all_options_for_date(c, d: date, min_iv_rows: int = 10) -> dict[str, list[dict]]:
    """
    Batch-fetch all options for a given date in one query.
    Returns {ticker: [option_rows]} filtered to tickers with >= min_iv_rows valid IV rows.

    One query per date instead of one per ticker — critical for full-universe scanning.
    """
    import sys
    print(f"[fetching options for {d}]", end=" ", file=sys.stderr, flush=True)
    with c.cursor() as cur:
        cur.execute(
            "SELECT act_symbol, expiration, strike, call_put, vol, delta "
            "FROM option_chain "
            "WHERE date=%s AND vol IS NOT NULL AND vol > 0 "
            "ORDER BY act_symbol, expiration, strike",
            (d.isoformat(),),
        )
        rows = cur.fetchall()
    print(f"[{len(rows)} rows]", end=" ", file=sys.stderr, flush=True)

    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ticker[r[0]].append({
            "expiry":  r[1],
            "strike":  float(r[2]) if r[2] is not None else None,
            "right":   r[3],
            "iv":      float(r[4]) if r[4] is not None else None,
            "delta":   float(r[5]) if r[5] is not None else None,
        })

    # Apply minimum row filter — tickers with too few rows can't form a valid calendar
    return {t: rows for t, rows in by_ticker.items() if len(rows) >= min_iv_rows}


def fetch_ohlcv_batch(c, tickers: list[str], n_days: int, up_to: date) -> dict[str, list[dict]]:
    """
    Batch OHLCV fetch for a list of tickers.
    Returns {ticker: [{date, close, high, low}, ...]} ascending, up to n_days trading days.
    Chunked into OHLCV_CHUNK-ticker groups. Each chunk retries up to 3× with a fresh
    connection on timeout (pymysql kills the connection on read_timeout expiry).
    """
    start = up_to - timedelta(days=int(n_days * 1.6) + 10)
    by_ticker: dict[str, list] = defaultdict(list)

    def _exec_chunk(connection, chunk: list[str]) -> list:
        placeholders = ",".join(["%s"] * len(chunk))
        with connection.cursor() as cur:
            cur.execute(
                f"SELECT act_symbol, date, close, high, low FROM ohlcv "
                f"WHERE act_symbol IN ({placeholders}) AND date BETWEEN %s AND %s "
                f"ORDER BY act_symbol, date",
                chunk + [start.isoformat(), up_to.isoformat()],
            )
            return cur.fetchall()

    for i in range(0, len(tickers), OHLCV_CHUNK):
        chunk = tickers[i : i + OHLCV_CHUNK]
        rows = None
        last_exc = None
        for attempt in range(3):
            try:
                rows = _exec_chunk(c, chunk)
                break
            except Exception as exc:
                last_exc = exc
                print(f"  OHLCV chunk {i//OHLCV_CHUNK + 1} attempt {attempt+1} failed: {exc}; reconnecting…",
                      file=sys.stderr, flush=True)
                try:
                    c.close()
                except Exception:
                    pass
                c = earningsvol_conn()
        if rows is None:
            print(f"  OHLCV chunk {i//OHLCV_CHUNK + 1} skipped after 3 failures: {last_exc}",
                  file=sys.stderr, flush=True)
            continue
        for r in rows:
            by_ticker[r[0]].append({
                "date":  r[1],
                "close": float(r[2]) if r[2] is not None else None,
                "high":  float(r[3]) if r[3] is not None else None,
                "low":   float(r[4]) if r[4] is not None else None,
            })

    return {t: data[-n_days:] for t, data in by_ticker.items()}


def fetch_earnings_batch(tickers: list[str], from_date: date, to_date: date) -> dict[str, list]:
    """
    Returns {ticker: [earnings_date, ...]} for upcoming earnings within the window.
    Connects to the earnings.earnings_calendar Dolt table.
    """
    c = earningsvol_conn()
    by_ticker: dict[str, list] = defaultdict(list)
    for i in range(0, len(tickers), OHLCV_CHUNK):
        chunk = tickers[i : i + OHLCV_CHUNK]
        placeholders = ",".join(["%s"] * len(chunk))
        with c.cursor() as cur:
            cur.execute(
                f"SELECT act_symbol, date FROM earnings_calendar "
                f"WHERE act_symbol IN ({placeholders}) AND date BETWEEN %s AND %s "
                f"ORDER BY act_symbol, date",
                chunk + [from_date.isoformat(), to_date.isoformat()],
            )
            for r in cur.fetchall():
                by_ticker[r[0]].append(r[1])
    c.close()
    return dict(by_ticker)

# ── FF calculation (same formula as ff_scanner.py) ───────────────────────────

def forward_vol(front_iv: float, front_t: float, back_iv: float, back_t: float) -> float:
    var = back_t * back_iv ** 2 - front_t * front_iv ** 2
    return math.sqrt(max(0.0, var / (back_t - front_t)))


def compute_ff(front_iv: float, fwd: float) -> float | None:
    # Campasano (2018) formula: 1mIV/FV(1,1) - 1, expressed as %
    return (front_iv / fwd - 1) * 100 if fwd > 0 else None


def ff_for_ticker_date(ticker: str, d: date, spot: float, rows: list[dict]) -> dict | None:
    """Compute FF for a ticker on a given date. Returns None if data insufficient."""
    exps: dict[date, list[dict]] = defaultdict(list)
    for r in rows:
        if r["expiry"] and r["right"] in ("Call", "C") and r["iv"] and r["iv"] > 0:
            exps[r["expiry"]].append(r)
    if not exps:
        return None

    front_exp = back_exp = None
    front_dte = back_dte = None
    for exp in sorted(exps.keys()):
        dte = (exp - d).days
        if FRONT_DTE_WINDOW[0] <= dte <= FRONT_DTE_WINDOW[1] and front_exp is None:
            front_exp, front_dte = exp, dte
        elif BACK_DTE_WINDOW[0] <= dte <= BACK_DTE_WINDOW[1] and back_exp is None:
            back_exp, back_dte = exp, dte

    if not front_exp or not back_exp:
        return None

    def atm(exp_rows):
        by_d = [r for r in exp_rows if r["delta"] is not None]
        if by_d:
            return min(by_d, key=lambda r: abs(r["delta"] - 0.50))
        by_s = [r for r in exp_rows if r["strike"] is not None]
        return min(by_s, key=lambda r: abs(r["strike"] - spot)) if by_s else None

    f_atm = atm(exps[front_exp])
    b_atm = atm(exps[back_exp])
    if not f_atm or not b_atm or not f_atm["iv"] or not b_atm["iv"]:
        return None

    fwd = forward_vol(f_atm["iv"], front_dte / 365, b_atm["iv"], back_dte / 365)
    ff  = compute_ff(f_atm["iv"], fwd)

    # Approximate ATM straddle mid using Black-Scholes: straddle ≈ S×σ×√T×√(2/π)
    _sqrt2pi = math.sqrt(2 / math.pi)
    back_straddle  = spot * b_atm["iv"] * math.sqrt(back_dte  / 365) * _sqrt2pi
    front_straddle = spot * f_atm["iv"] * math.sqrt(front_dte / 365) * _sqrt2pi
    entry_debit    = back_straddle - front_straddle
    t_fwd          = back_dte - front_dte
    max_theo       = back_straddle * math.sqrt(t_fwd / back_dte) if back_dte > 0 else None
    max_profit     = (max_theo - entry_debit) if max_theo is not None else None
    atm_strike     = b_atm["strike"]

    return {
        "date":          d.isoformat(),
        "ticker":        ticker,
        "front_exp":     front_exp.isoformat(),
        "back_exp":      back_exp.isoformat(),
        "front_dte":     front_dte,
        "back_dte":      back_dte,
        "front_iv":      round(f_atm["iv"], 4),
        "back_iv":       round(b_atm["iv"], 4),
        "forward_vol":   round(fwd, 4),
        "ff_pct":        round(ff, 2) if ff is not None else None,
        # Pricing approximations (BS ATM straddle) — used for ML scoring
        "back_straddle": round(back_straddle, 4),
        "entry_debit":   round(entry_debit, 4),
        "max_profit":    round(max_profit, 4) if max_profit is not None else None,
        "atm_strike":    round(atm_strike, 2) if atm_strike else None,
        "spot":          round(spot, 2),
    }

# ── Trend analysis ────────────────────────────────────────────────────────────

def trend_stats(series: list[dict], ff_threshold: float) -> dict:
    """
    Given a list of daily FF records (ascending by date), compute trend metrics.
    """
    ff_vals = [r["ff_pct"] for r in series if r.get("ff_pct") is not None]
    if not ff_vals:
        return {}

    current = ff_vals[-1]
    above   = [v for v in ff_vals if v >= ff_threshold]
    days_above = len(above)

    # Consecutive days above threshold (counting from most recent)
    consec = 0
    for v in reversed(ff_vals):
        if v >= ff_threshold:
            consec += 1
        else:
            break

    # Trend: slope over last 5 points
    last5 = ff_vals[-5:]
    if len(last5) >= 2:
        slope = (last5[-1] - last5[0]) / max(len(last5) - 1, 1)
        trend = "rising" if slope > 1.0 else ("falling" if slope < -1.0 else "stable")
    else:
        slope = 0.0
        trend = "insufficient_data"

    n = len(ff_vals)
    return {
        "current_ff":         current,
        "ff_5d_ago":          ff_vals[-5] if n >= 5 else None,
        "ff_10d_ago":         ff_vals[-10] if n >= 10 else None,
        "days_above_thresh":  days_above,
        "consec_above_thresh": consec,
        "trend":              trend,
        "trend_slope_5d":     round(slope, 2),
        "n_observations":     n,
    }

# ── Ranking: compute helpers ──────────────────────────────────────────────────

def compute_hv20(closes: list[float]) -> float | None:
    """20-day annualized realized volatility from daily log returns."""
    if len(closes) < 21:
        return None
    log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(-20, 0)]
    mean_r = sum(log_rets) / 20
    var = sum((r - mean_r) ** 2 for r in log_rets) / 19
    return math.sqrt(var * 252)


def compute_trend_strength(closes: list[float], n: int = 20) -> float | None:
    """
    Returns |slope_pct_per_day| * R² over n days.
    Low = range-bound (good for calendars). High = clean directional trend (bad).
    Better than ADX for event-driven stocks that gapped but then stabilized.
    """
    if len(closes) < n:
        return None
    ys = closes[-n:]
    mean_x = (n - 1) / 2
    mean_y = sum(ys) / n
    ss_xy = sum((i - mean_x) * (ys[i] - mean_y) for i in range(n))
    ss_xx = sum((i - mean_x) ** 2 for i in range(n))
    ss_yy = sum((y - mean_y) ** 2 for y in ys)
    if ss_xx == 0 or ss_yy == 0:
        return 0.0
    slope = ss_xy / ss_xx
    r2 = (ss_xy ** 2) / (ss_xx * ss_yy)
    slope_pct_per_day = abs(slope) / mean_y * 100
    return slope_pct_per_day * r2


def compute_iv_rank_20d(series: list[dict]) -> float | None:
    """
    IV Rank over the 20-day scan window using front_iv.
    Not a full 52-week IVR but captures recent IV elevation relative to the scan period.
    """
    ivs = [r["front_iv"] for r in series if r.get("front_iv") is not None]
    if len(ivs) < 3:
        return None
    min_iv, max_iv = min(ivs), max(ivs)
    if max_iv == min_iv:
        return 50.0
    return (ivs[-1] - min_iv) / (max_iv - min_iv) * 100


# ── Ranking: scoring functions (each returns component points) ─────────────────

def _score_earnings(days_to_earn: int | None, front_dte: int) -> float:
    """
    0–25 pts. Earnings before front expiry = 0 (hard disqualify).
    Earnings just after front = partial penalty (IV crush risk lingers).
    """
    if days_to_earn is None:
        return 20.0  # No earnings data — assume clean, slight penalty for uncertainty
    if days_to_earn <= front_dte:
        return 0.0   # Earnings inside front window — IV crush will destroy calendar
    if days_to_earn <= front_dte + 14:
        return 8.0   # Earnings close after front — IV still partially priced in
    return 25.0      # Earnings well outside front window — clean


def _score_iv_rank(ivr: float | None) -> float:
    """
    0–20 pts. Sweet spot 40–70. Below 30 = noise. Above 85 = event risk.
    Uses 20-day rank from scan series (not 52-week — noted limitation).
    """
    if ivr is None:
        return 8.0   # Neutral if insufficient history
    if ivr < 10:
        return 3.0
    if ivr < 30:
        return 8.0 + (ivr - 10) * 0.2   # 8→12 linearly
    if ivr <= 70:
        return 20.0                       # Full score
    if ivr <= 85:
        return 20.0 - (ivr - 70) * 0.8   # Decay 20→8
    return 5.0                            # Extreme = likely event-distorted


def _score_iv_hv(ratio: float | None) -> float:
    """
    0–25 pts. IV/HV20 sweet spot 1.2–2.5.
    Below 1.0 = IV < realized = no premium to sell on front.
    Above 4.0 = event IV — may crush before expiry.
    """
    if ratio is None:
        return 8.0
    if ratio < 0.9:
        return 0.0
    if ratio < 1.2:
        return 8.0
    if ratio <= 2.5:
        return 25.0
    if ratio <= 4.0:
        return 25.0 - (ratio - 2.5) * 8.0
    return 5.0


def _score_trend(ts: float | None) -> float:
    """
    0–20 pts. Low slope×R² = range-bound = good for calendars.
    Does not penalize post-event gaps that have stabilized (unlike ADX).
    """
    if ts is None:
        return 10.0  # Neutral
    if ts < 0.2:
        return 20.0  # Range-bound
    if ts < 0.5:
        return 15.0
    if ts < 1.0:
        return 8.0
    return 0.0       # Clean directional trend = bad


def _score_ff_quality(series: list[dict], stats: dict) -> float:
    """
    0–10 pts. Which leg is driving FF, and is timing good?
    Stabilizing/declining FF = better entry timing.
    Back IV rising faster than front = sustainable inversion.
    Front exploding alone = unstable, short-lived.
    """
    trend = stats.get("trend", "")
    if trend == "falling":
        return 10.0   # Post-peak — classic entry window

    if trend == "stable":
        return 8.0

    if trend == "rising" and len(series) >= 5:
        prev, curr = series[-5], series[-1]
        d_front = (curr.get("front_iv") or 0) - (prev.get("front_iv") or 0)
        d_back  = (curr.get("back_iv")  or 0) - (prev.get("back_iv")  or 0)
        if d_back > d_front:
            return 6.0   # Back rising faster — sustainable inversion
        return 3.0       # Front exploding relative to back — unstable

    return 5.0  # insufficient_data or other


def round_to_strike(price: float) -> float:
    """Round to nearest tradeable strike increment based on price level."""
    if price < 10:
        inc = 0.5
    elif price < 50:
        inc = 1.0
    elif price < 100:
        inc = 2.0
    elif price < 250:
        inc = 5.0
    elif price < 500:
        inc = 5.0
    else:
        inc = 10.0
    return round(round(price / inc) * inc, 2)


def suggest_strikes(
    spot: float | None,
    front_iv: float | None,
    front_dte: int | None,
    structure: str,
) -> dict:
    """
    Compute suggested strikes for the calendar.
    Single: ATM (one tent). Double: spot ± 1σ expected move (two tents).
    Returns dict with atm_strike, dc_lower_strike, dc_upper_strike, expected_move_1sd.
    """
    if not spot or not front_iv or not front_dte:
        return {"atm_strike": None, "dc_lower_strike": None,
                "dc_upper_strike": None, "expected_move_1sd": None}

    exp_move = spot * front_iv * math.sqrt(front_dte / 365)
    atm = round_to_strike(spot)

    if structure == "double_calendar":
        lower = round_to_strike(spot - exp_move)
        upper = round_to_strike(spot + exp_move)
    else:
        lower = upper = None

    return {
        "atm_strike":       atm,
        "dc_lower_strike":  lower,
        "dc_upper_strike":  upper,
        "expected_move_1sd": round(exp_move, 2),
    }


def suggest_structure(ticker: str, trend_str: float | None, earnings_in_front: bool) -> str:
    """
    Double calendar: mean-reverting underlyings (ETFs + range-bound stocks).
    Single calendar: stocks with directional lean. Skip: earnings in front window.
    """
    if earnings_in_front:
        return "skip"
    if ticker in KNOWN_ETFS:
        return "double_calendar"
    if trend_str is not None and trend_str < 0.3:
        return "double_calendar"
    return "single_calendar"


# ── ML scoring (LR bin_bigwin model) ─────────────────────────────────────────

_ML_MODEL: dict | None = None  # module-level cache


def load_ml_scorer() -> dict | None:
    """Load LR coefficients from ff_ml_lr_coefs_v2.json (same directory as this script)."""
    path = Path(__file__).parent / "ff_ml_lr_coefs_v2.json"
    if not path.exists():
        print(f"WARNING: ML model not found at {path} — ML scoring disabled", file=sys.stderr)
        return None
    with open(path) as fh:
        return json.load(fh)


def _get_ml_model() -> dict | None:
    global _ML_MODEL
    if _ML_MODEL is None:
        _ML_MODEL = load_ml_scorer()
    return _ML_MODEL


def ml_score_candidate(c: dict) -> float | None:
    """
    Score a candidate using the LR bin_bigwin model (P(hold-return > +50%)).
    Returns probability ∈ [0,1], or None if required fields are missing.

    Reads: ff_pct, back_iv, entry_debit, back_straddle, max_profit,
           front_dte, back_dte, ticker, data_date (ISO string).
    """
    model = _get_ml_model()
    if model is None:
        return None

    ff      = c.get("ff_pct")
    b_iv    = c.get("back_iv")
    debit   = c.get("entry_debit")
    bstrad  = c.get("back_straddle")
    max_p   = c.get("max_profit")
    t_front = c.get("front_dte")
    t_back  = c.get("back_dte")
    ticker  = c.get("ticker", "")
    ds      = c.get("data_date") or c.get("front_expiry", "")

    if any(v is None for v in [ff, b_iv, debit, bstrad, t_front, t_back]):
        return None
    if bstrad <= 0:
        return None

    t_fwd = t_back - t_front

    try:
        d = date.fromisoformat(str(ds)[:10])
    except (ValueError, TypeError):
        d = date.today()
    month = d.month
    quarter = (month - 1) // 3 + 1

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
        "max_profit_ratio":  (max_p / bstrad) if max_p is not None else 0.0,
        "iv_proxy":          b_iv * math.sqrt(2.0 / math.pi),  # ≈ straddle/(S×√(T/365))
        "ff_x_tfwd":         ff * math.sqrt(max(t_fwd, 1)),
        "month_sin":         math.sin(2 * math.pi * month / 12),
        "month_cos":         math.cos(2 * math.pi * month / 12),
        "quarter":           float(quarter),
        "is_etf":            1.0 if ticker in KNOWN_ETFS else 0.0,
    }

    coefs     = model["coefs"]
    means     = model["scaler_mean"]
    stds      = model["scaler_std"]
    logit     = model["intercept"]
    for feat, val in features.items():
        std = stds.get(feat, 1.0)
        mu  = means.get(feat, 0.0)
        z   = (val - mu) / std if std > 0 else 0.0
        logit += coefs.get(feat, 0.0) * z

    prob = 1.0 / (1.0 + math.exp(-logit))
    return round(prob, 4)


# ── Ranking: main enrichment ──────────────────────────────────────────────────

def rank_candidates(
    candidates: list[dict],
    scan_result: dict,
    target_date: date,
) -> list[dict]:
    """
    Enrich candidates with ranking scores. Fetches OHLCV + earnings from Dolt.
    Mutates candidates in place; returns sorted by composite_score (earnings risk last).
    """
    if not candidates:
        return candidates

    tickers = [c["ticker"] for c in candidates]
    print(f"\nRanking {len(tickers)} candidates (OHLCV + earnings) ...", flush=True)

    stk = earningsvol_conn()
    ohlcv_data = fetch_ohlcv_batch(stk, tickers, 60, target_date)
    stk.close()

    earn_data = fetch_earnings_batch(
        tickers, target_date, target_date + timedelta(days=90)
    )

    print(f"  OHLCV: {len(ohlcv_data)} tickers | Earnings: {sum(len(v) for v in earn_data.values())} events")

    from datetime import datetime as _dt

    for c in candidates:
        ticker = c["ticker"]
        front_dte = c.get("front_dte") or 30

        front_expiry = None
        fe_str = c.get("front_expiry")
        if fe_str:
            try:
                front_expiry = _dt.strptime(fe_str, "%Y-%m-%d").date()
            except ValueError:
                pass

        # OHLCV-based metrics
        ohlcv = ohlcv_data.get(ticker, [])
        closes = [r["close"] for r in ohlcv if r["close"]]
        hv = compute_hv20(closes)
        trend_str = compute_trend_strength(closes)

        # IV metrics from existing scan series
        series = scan_result.get(ticker, {}).get("series", [])
        stats  = scan_result.get(ticker, {}).get("stats", {})
        iv_rank_20d = compute_iv_rank_20d(series)
        current_iv = c.get("front_iv")
        iv_hv = (current_iv / hv) if (current_iv and hv and hv > 0) else None

        # Earnings
        earn_dates = earn_data.get(ticker, [])
        next_earn = None
        days_to_earn = None
        if earn_dates:
            future = [d for d in earn_dates if d >= target_date]
            if future:
                next_earn = min(future)
                days_to_earn = (next_earn - target_date).days

        earnings_in_front = (
            next_earn is not None
            and front_expiry is not None
            and next_earn <= front_expiry
        )

        # Component scores
        earn_pts  = _score_earnings(days_to_earn, front_dte)
        ivr_pts   = _score_iv_rank(iv_rank_20d)
        ivhv_pts  = _score_iv_hv(iv_hv)
        trend_pts = _score_trend(trend_str)
        ff_pts    = _score_ff_quality(series, stats)
        composite = earn_pts + ivr_pts + ivhv_pts + trend_pts + ff_pts

        structure = suggest_structure(ticker, trend_str, earnings_in_front)
        spot = closes[-1] if closes else None
        strikes = suggest_strikes(spot, current_iv, front_dte, structure)

        ml_score = ml_score_candidate(c)

        c.update({
            "ml_score":            ml_score,
            "composite_score":     round(composite, 1),
            "earnings_risk":       earnings_in_front,
            "days_to_earnings":    days_to_earn,
            "next_earnings":       next_earn.isoformat() if next_earn else None,
            "iv_rank_20d":         round(iv_rank_20d, 1) if iv_rank_20d is not None else None,
            "hv20":                round(hv, 4) if hv is not None else None,
            "iv_hv_ratio":         round(iv_hv, 2) if iv_hv is not None else None,
            "trend_strength":      round(trend_str, 3) if trend_str is not None else None,
            "earn_pts":            round(earn_pts, 1),
            "ivr_pts":             round(ivr_pts, 1),
            "ivhv_pts":            round(ivhv_pts, 1),
            "trend_pts":           round(trend_pts, 1),
            "ff_quality_pts":      round(ff_pts, 1),
            "suggested_structure": structure,
            "spot":                round(spot, 2) if spot else None,
            **strikes,
        })

    # Compute ml_pct: percentile rank within today's candidate list (100 = best)
    from bisect import bisect_right as _bsr
    _scores = sorted(c.get("ml_score") or 0.0 for c in candidates)
    _n = len(_scores)
    for c in candidates:
        s = c.get("ml_score") or 0.0
        c["ml_pct"] = round(100.0 * _bsr(_scores, s) / _n, 1) if _n > 0 else None

    # Sort: earnings-risk tickers last, then by ML score descending
    candidates.sort(
        key=lambda r: (r.get("earnings_risk") or False, -(r.get("ml_score") or 0.0))
    )
    return candidates

# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan(
    target_date: date,
    n_days: int,
    ff_min: float,
    ticker_override: list[str] | None = None,
    min_iv_rows: int = 10,
) -> dict:
    """
    Optimized batch scan: filter to FF >= ff_min on latest date, then backfill trend for those only.
    Returns: {ticker: {"series": [...], "stats": {...}}}

    ticker_override: if set, only these tickers are processed (testing/override mode).
    min_iv_rows: minimum number of valid IV rows a ticker must have to be included.
    """
    opt_conn = earningsvol_conn()
    stk_conn = earningsvol_conn()

    # Find latest date with actual data (starts from today, walks back up to 10 days)
    print(f"Finding latest options data starting from {target_date} ...", flush=True)
    latest = find_latest_data_date(opt_conn, target_date)
    if not latest:
        print(f"ERROR: No options data found in last 10 days from {target_date}", file=sys.stderr)
        opt_conn.close()
        stk_conn.close()
        return {}

    if latest < target_date:
        days_old = (target_date - latest).days
        print(f"⚠ Latest data is {days_old} day(s) old: {latest}", file=sys.stderr)

    print(f"Getting last {n_days} trading dates up to {latest} ...", flush=True)
    trading_dates = get_trading_dates(n_days, latest)
    if not trading_dates:
        print("No trading dates found in Dolt.", file=sys.stderr)
        return {}
    print(f"  Dates: {trading_dates[0]} → {trading_dates[-1]} ({len(trading_dates)} days)")

    # Phase 1: Get candidates from latest date only (fast filter)
    print(f"Phase 1: Filtering to FF >= {ff_min}% on {latest} ...", flush=True)
    spots_latest = get_all_spots(stk_conn, latest)
    print(f"  Spots: {len(spots_latest)} tickers", flush=True)
    options_latest = get_all_options_for_date(opt_conn, latest, min_iv_rows)
    print(f"  Options: {len(options_latest)} tickers", flush=True)

    candidates = []
    candidate_recs = {}  # {ticker: ff_record}
    no_spot = 0
    no_ff = 0
    for ticker, rows in options_latest.items():
        spot = spots_latest.get(ticker)
        if spot is None:
            no_spot += 1
            continue
        rec = ff_for_ticker_date(ticker, latest, spot, rows)
        if rec:
            ff = rec.get("ff_pct")
            if ff is not None and ff >= ff_min:
                candidates.append(ticker)
                candidate_recs[ticker] = rec  # Store for ranking
            else:
                no_ff += 1

    if ticker_override:
        candidates = [t for t in candidates if t in ticker_override]

    print(f"  {len(candidates)} tickers pass FF >= {ff_min}%")
    if not candidates:
        if len(options_latest) == 0:
            print(f"  ERROR: No options data in options.option_chain for {latest}", file=sys.stderr)
        elif no_spot > 0:
            print(f"  ERROR: {no_spot} tickers have options but no spot price in stocks.ohlcv for {latest}", file=sys.stderr)
        else:
            print(f"  ERROR: All {no_ff} tickers have FF < {ff_min}% on {latest} — no candidates.", file=sys.stderr)
        opt_conn.close()
        stk_conn.close()
        return {}

    # Build minimal scan_result for Phase 1 candidates (latest date only, for deterministic filters)
    phase1_results: dict[str, dict] = {}
    for ticker, rec in candidate_recs.items():
        phase1_results[ticker] = {"series": [rec], "stats": {}}

    # Phase 1.5: Apply deterministic filters before expensive backfill
    print(f"\nApplying deterministic filters to {len(candidates)} candidates ...", flush=True)
    # Pass the full rec so rank_candidates has front_expiry for earnings_in_front check
    candidate_list = []
    for t in candidates:
        rec = candidate_recs[t]
        candidate_list.append({
            "ticker":       t,
            "front_expiry": rec.get("front_exp"),
            "back_expiry":  rec.get("back_exp"),
            "front_dte":    rec.get("front_dte"),
            "front_iv":     rec.get("front_iv"),
        })
    filtered = rank_candidates(candidate_list, phase1_results, latest)

    # Filter by deterministic gates: no earnings_risk, good spreads/volume implied by entry_ready
    filtered_final = []
    for c in filtered:
        ticker = c["ticker"]
        earnings_risk = c.get("earnings_risk", False)
        if earnings_risk:
            print(f"  {ticker}: skip (earnings during hold)", file=sys.stderr)
            continue
        filtered_final.append(ticker)

    print(f"  {len(filtered_final)} pass deterministic filters")
    candidates = filtered_final

    # Phase 2: Backfill trend for candidates only (expensive operation)
    print(f"Phase 2: Backfilling {len(candidates)} candidates over {len(trading_dates)} dates ...", flush=True)
    results: dict[str, list[dict]] = defaultdict(list)
    # Pre-populate with Phase 1 (latest) records — only tickers that passed Phase 1.5
    for ticker in candidates:
        results[ticker].append(candidate_recs[ticker])
    skipped_empty = skipped_no_hits = 0

    for d in trading_dates:
        print(f"  {d} ... ", end="", flush=True)

        spots = get_all_spots(stk_conn, d)

        options_by_ticker = None
        for _attempt in range(3):
            try:
                options_by_ticker = get_all_options_for_date(opt_conn, d, min_iv_rows)
                break
            except Exception as _exc:
                print(f"[retry {_attempt+1}: {_exc}] ", end="", file=sys.stderr, flush=True)
                try:
                    opt_conn.close()
                except Exception:
                    pass
                opt_conn = earningsvol_conn()

        # Skip dates with no options data at all
        if not options_by_ticker:
            print("[no data] SKIP", file=sys.stderr)
            skipped_empty += 1
            continue

        # Filter to only candidates that passed latest-date filter
        options_by_ticker = {t: v for t, v in options_by_ticker.items() if t in candidates}

        hits = 0
        for ticker, rows in options_by_ticker.items():
            spot = spots.get(ticker)
            if spot is None:
                continue
            rec = ff_for_ticker_date(ticker, d, spot, rows)
            if rec:
                results[ticker].append(rec)
                hits += 1

        n_found = len(options_by_ticker)
        if n_found > 0 and hits == 0:
            print(f"{hits}/{n_found} (0 hits — exit date too far back?) SKIP", file=sys.stderr)
            skipped_no_hits += 1
        else:
            print(f"{hits}/{n_found} candidates with data")

    if skipped_empty or skipped_no_hits:
        print(f"Phase 2 summary: skipped {skipped_empty} empty dates, {skipped_no_hits} dates with 0 hits", file=sys.stderr)

    opt_conn.close()
    stk_conn.close()

    # Build output: series + trend stats per ticker
    output = {}
    for ticker, series in results.items():
        if not series:
            continue
        stats = trend_stats(series, ff_min)
        output[ticker] = {"series": series, "stats": stats}

    return output

# ── Output writers ────────────────────────────────────────────────────────────

CANDIDATE_FIELDS = [
    # Primary ranking
    "ticker", "ml_score", "ml_pct", "ff_pct", "suggested_structure",
    # FF trend
    "ff_5d_ago", "ff_10d_ago", "trend", "trend_slope_5d",
    "days_above_thresh", "consec_above_thresh", "n_observations",
    # Expiry info
    "front_expiry", "back_expiry", "front_dte", "back_dte", "front_iv", "back_iv",
    # Pricing (BS ATM straddle approximation — ML inputs)
    "back_straddle", "entry_debit", "max_profit",
    # Ranking inputs
    "iv_rank_20d", "hv20", "iv_hv_ratio", "trend_strength",
    "earnings_risk", "days_to_earnings", "next_earnings",
    # Suggested trade setup
    "spot", "expected_move_1sd", "atm_strike",
    "dc_lower_strike", "dc_upper_strike",
    # Component scores (for diagnostics)
    "composite_score", "earn_pts", "ivr_pts", "ivhv_pts", "trend_pts", "ff_quality_pts",
]


def write_outputs(scan_result: dict, target_date: date, ff_min: float):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.isoformat()

    # Full universe JSON (complete time series)
    json_path = OUTPUT_DIR / f"{date_str}_universe.json"
    with open(json_path, "w") as f:
        json.dump(scan_result, f, indent=2, default=str)
    print(f"\nFull universe: {json_path}")

    # Candidates CSV (FF >= ff_min on latest date)
    candidates = []
    for ticker, data in scan_result.items():
        stats = data.get("stats", {})
        current_ff = stats.get("current_ff")
        if current_ff is None or current_ff < ff_min:
            continue
        if current_ff >= 99.9:  # fwd_vol=0 artifact (back IV < front IV)
            continue
        series = data.get("series", [])
        latest = series[-1] if series else {}

        # Re-derive DTE from target_date (today) — the stored front_exp was selected
        # relative to the last Dolt data date, which may be days behind today.
        fe_str = latest.get("front_exp")
        be_str = latest.get("back_exp")
        front_dte_today = (date.fromisoformat(fe_str) - target_date).days if fe_str else None
        back_dte_today  = (date.fromisoformat(be_str) - target_date).days if be_str else None

        # Skip if front expiry has rolled below the entry window from today.
        if front_dte_today is None or front_dte_today < FRONT_DTE_WINDOW[0]:
            continue

        candidates.append({
            "ticker":              ticker,
            "ff_pct":              current_ff,
            "ff_5d_ago":           stats.get("ff_5d_ago"),
            "ff_10d_ago":          stats.get("ff_10d_ago"),
            "trend":               stats.get("trend"),
            "trend_slope_5d":      stats.get("trend_slope_5d"),
            "days_above_thresh":   stats.get("days_above_thresh"),
            "consec_above_thresh": stats.get("consec_above_thresh"),
            "n_observations":      stats.get("n_observations"),
            "front_expiry":        fe_str,
            "back_expiry":         be_str,
            "front_dte":           front_dte_today,
            "back_dte":            back_dte_today,
            "front_iv":            latest.get("front_iv"),
            # ML scoring fields (from BS straddle approximation in ff_for_ticker_date)
            "back_iv":             latest.get("back_iv"),
            "back_straddle":       latest.get("back_straddle"),
            "entry_debit":         latest.get("entry_debit"),
            "max_profit":          latest.get("max_profit"),
            "data_date":           latest.get("date"),
        })

    # Enrich with ranking scores (OHLCV + earnings — batch Dolt queries)
    candidates = rank_candidates(candidates, scan_result, target_date)
    # rank_candidates already sorts; no secondary sort needed

    cand_path = OUTPUT_DIR / f"{date_str}_candidates.csv"
    with open(cand_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(candidates)

    # "latest" copies for intraday scanner
    import shutil
    latest_csv  = OUTPUT_DIR / "latest_candidates.csv"
    latest_json = OUTPUT_DIR / "latest_universe.json"
    shutil.copy(cand_path, latest_csv)
    shutil.copy(json_path, latest_json)

    print(f"Candidates (FF≥{ff_min}%): {cand_path}  [{len(candidates)} tickers]")
    print(f"Latest CSV : {latest_csv}")
    print(f"Latest JSON: {latest_json}")

    # Console summary — sorted by ML score, earnings-risk flagged
    if candidates:
        hdr = (f"{'Ticker':<7} {'MLscore':>7}  {'MLpct':>5}  {'FF%':>7}  {'IVR20':>5}  {'IV/HV':>5}"
               f"  {'Earn':>6}  {'Structure':<16}  {'Spot':>7}  {'1σ':>5}  {'Strikes'}")
        print(f"\n{hdr}")
        print("  " + "-" * (len(hdr) + 2))
        for r in candidates:
            earn_flag = "⚠ EARN" if r.get("earnings_risk") else ""
            structure = r.get("suggested_structure") or ""
            if structure == "double_calendar":
                lo = r.get("dc_lower_strike")
                hi = r.get("dc_upper_strike")
                strikes_str = f"{lo}/{hi}" if lo and hi else ""
            else:
                atm = r.get("atm_strike")
                strikes_str = f"ATM {atm}" if atm else ""
            ml_s = r.get("ml_score")
            ml_p = r.get("ml_pct")
            ml_str = f"{ml_s:>7.4f}" if ml_s is not None else f"{'N/A':>7}"
            print(
                f"  {r['ticker']:<6}"
                f"  {ml_str}"
                f"  {(ml_p or 0):>4.0f}%"
                f"  {(r.get('ff_pct') or 0):>+6.1f}%"
                f"  {(r.get('iv_rank_20d') or 0):>4.0f}%"
                f"  {(r.get('iv_hv_ratio') or 0):>4.2f}x"
                f"  {earn_flag:<6}"
                f"  {structure:<16}"
                f"  {(r.get('spot') or 0):>7.2f}"
                f"  {(r.get('expected_move_1sd') or 0):>5.2f}"
                f"  {strikes_str}"
            )
    else:
        print(f"\nNo tickers with FF >= {ff_min}% on {target_date}.")

    return candidates


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FF Universe Scanner — overnight Dolt batch scan")
    parser.add_argument("--date", default=None,
                        help="Target date YYYY-MM-DD (default: latest in Dolt)")
    parser.add_argument("--ff-min", type=float, default=15.0,
                        help="FF%% threshold for candidates output (default 15)")
    parser.add_argument("--days", type=int, default=20,
                        help="Trading days of look-back history (default 20)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override: scan only these tickers (default: all Dolt tickers)")
    parser.add_argument("--min-iv-rows", type=int, default=10,
                        help="Minimum valid IV rows per ticker per date to include (default 10)")
    args = parser.parse_args()

    # ensure_dolt_server() — removed, PostgreSQL connection is always available

    if args.date:
        from datetime import datetime as dt
        target = dt.strptime(args.date, "%Y-%m-%d").date()
    else:
        target = date.today()

    universe_desc = (
        f"{len(args.tickers)} tickers (override)" if args.tickers
        else f"all Dolt tickers (min_iv_rows≥{args.min_iv_rows})"
    )
    print(f"FF Universe Scanner")
    print(f"Target date: {target} | Look-back: {args.days} days | FF min: {args.ff_min}%")
    print(f"Universe: {universe_desc}")
    print()

    scan_result = run_scan(
        target, args.days, args.ff_min,
        ticker_override=args.tickers,
        min_iv_rows=args.min_iv_rows,
    )
    write_outputs(scan_result, target, args.ff_min)

    return 0


if __name__ == "__main__":
    sys.exit(main())
