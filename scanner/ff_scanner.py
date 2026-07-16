#!/usr/bin/env python3
"""
Forward Factor daily scanner for S3 calendar spread strategy (T028).

Scans the watchlist for FF > 16% signals indicating inverted term structure.
Designed to run at 15:00 ET (close) when IB data is live.

Usage:
    python ff_scanner.py                   # IB + yfinance fallback
    python ff_scanner.py --no-ib           # yfinance only (offline testing)
    python ff_scanner.py --tickers SPY QQQ IWM
    python ff_scanner.py --date 2026-06-05 --no-ib

Output: daily_scans/YYYY-MM-DD_ff_scan.csv

Universe modes:
  Default:      hardcoded WATCHLIST (8 ETFs)
  --tickers:    explicit list
  --universe:   load pre-built candidates from ff_universe_scan.py output
                e.g. universe/latest_candidates.csv  (fast intraday mode)

Forward Factor definition (Campasano / Sean Ryan):
    sigma_fwd = sqrt(max(0, (T2*sigma2^2 - T1*sigma1^2) / (T2-T1)))
    FF = (front_iv - sigma_fwd) / front_iv * 100
    FF > 16% = decile 9/10 = calendar spread entry signal
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

OPTTRADER_DIR = Path(os.environ.get("OPTTRADER_DIR", Path.home() / "Documents" / "OptionTrader"))
OPTCHAIN_SCRIPT = OPTTRADER_DIR / "tools" / "option_chain.py"
OUTPUT_DIR = Path(__file__).parent / "daily_scans"

WATCHLIST = ["SPY", "QQQ", "IWM", "XLK", "XLF", "GLD", "TLT", "EFA"]

FRONT_DTE_TARGET = 30
BACK_DTE_TARGET = 60
FRONT_DTE_WINDOW = (20, 45)
BACK_DTE_WINDOW = (45, 80)

FF_SIGNAL_THRESHOLD = 16.0
MIN_VOLUME = 1000


def forward_vol(front_iv: float, front_t: float, back_iv: float, back_t: float) -> float:
    var = back_t * back_iv ** 2 - front_t * front_iv ** 2
    return math.sqrt(max(0.0, var / (back_t - front_t)))


def compute_ff(front_iv: float, fwd_vol: float) -> float | None:
    if fwd_vol <= 0:
        return None
    return (front_iv / fwd_vol - 1) * 100


def ff_to_decile(ff: float | None) -> int | None:
    if ff is None:
        return None
    if ff < 2:
        return 1
    if ff < 5:
        return 3
    if ff < 8:
        return 5
    if ff < 12:
        return 7
    if ff < 16:
        return 8
    if ff < 20:
        return 9
    return 10


def fetch_chain(ticker: str, dte: int, no_ib: bool) -> dict | None:
    cmd = [
        sys.executable, str(OPTCHAIN_SCRIPT),
        "--ticker", ticker,
        "--dte", str(dte),
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
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    rows = data.get("rows", [])
    spot = data.get("spot") or 0
    if not rows:
        return None

    calls = [
        r for r in rows
        if r.get("right") in ("C", "Call")
        and r.get("iv") is not None
        and r.get("iv") > 0
    ]
    if not calls:
        return None

    def atm_key(r):
        d = r.get("delta")
        if d is not None:
            return abs(d - 0.50)
        return abs((r.get("strike") or 0) - spot)

    atm = min(calls, key=atm_key)

    expiry = atm.get("expiry")
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").date() if expiry else None
    actual_dte = (expiry_dt - date.today()).days if expiry_dt else None

    total_volume = sum((r.get("volume") or 0) for r in rows if r.get("expiry") == expiry)

    bid = atm.get("bid") or 0
    ask = atm.get("ask") or 0
    spread = round(ask - bid, 3) if ask > bid else None

    return {
        "spot": spot,
        "expiry": expiry,
        "dte": actual_dte,
        "iv": atm.get("iv"),
        "strike": atm.get("strike"),
        "delta": atm.get("delta"),
        "volume": total_volume,
        "bid_ask_spread": spread,
    }


def scan_ticker(ticker: str, no_ib: bool) -> dict | None:
    front = fetch_chain(ticker, FRONT_DTE_TARGET, no_ib)
    back = fetch_chain(ticker, BACK_DTE_TARGET, no_ib)

    if not front or not back:
        return None
    if not front.get("iv") or not back.get("iv"):
        return None
    if front.get("expiry") == back.get("expiry"):
        return None

    front_dte = front["dte"] or 0
    back_dte = back["dte"] or 0

    if not (FRONT_DTE_WINDOW[0] <= front_dte <= FRONT_DTE_WINDOW[1]):
        return None
    if not (BACK_DTE_WINDOW[0] <= back_dte <= BACK_DTE_WINDOW[1]):
        return None

    front_iv = front["iv"]
    back_iv = back["iv"]
    front_t = front_dte / 365.0
    back_t = back_dte / 365.0

    fwd = forward_vol(front_iv, front_t, back_iv, back_t)
    ff = compute_ff(front_iv, fwd)

    volume_ok = (front.get("volume") or 0) >= MIN_VOLUME and (back.get("volume") or 0) >= MIN_VOLUME
    ff_signal = ff is not None and ff >= FF_SIGNAL_THRESHOLD and volume_ok

    return {
        "scan_date": date.today().isoformat(),
        "ticker": ticker,
        "front_expiry": front["expiry"],
        "back_expiry": back["expiry"],
        "front_dte": front_dte,
        "back_dte": back_dte,
        "front_iv": round(front_iv, 4),
        "forward_vol": round(fwd, 4),
        "ff_pct": round(ff, 2) if ff is not None else None,
        "ff_decile": ff_to_decile(ff),
        "ff_signal": ff_signal,
        "volume_1000plus": volume_ok,
        "bid_ask_spread": front.get("bid_ask_spread"),
        "rank_by_ff": None,
    }


CSV_FIELDS = [
    "scan_date", "ticker", "front_expiry", "back_expiry",
    "front_dte", "back_dte", "front_iv", "forward_vol",
    "ff_pct", "ff_decile", "ff_signal", "volume_1000plus",
    "bid_ask_spread", "rank_by_ff",
]


def load_universe_from_file(path: str) -> list[str]:
    import csv as _csv
    with open(path, newline="") as f:
        rows = list(_csv.DictReader(f))
    return [r["ticker"] for r in rows if r.get("ticker")]


def acquire_lock(lock_name: str) -> int:
    import fcntl
    lock_dir = os.environ.get("CSFF_LOCK", "/tmp")
    lock_path = os.path.join(lock_dir, lock_name)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print(f"Lock {lock_path} already held — another {lock_name} is running", file=sys.stderr)
        sys.exit(1)
    return fd


def main():
    parser = argparse.ArgumentParser(description="Forward Factor daily scanner (S3/T028)")
    parser.add_argument("--no-ib", action="store_true", help="Skip IB gateway, use yfinance only")
    parser.add_argument("--tickers", nargs="+", default=None, help="Explicit ticker list")
    parser.add_argument("--universe", metavar="FILE",
                        help="Load ticker list from universe candidates CSV (fast intraday mode)")
    parser.add_argument("--date", default=date.today().isoformat(), help="Scan date for filename (YYYY-MM-DD)")
    args = parser.parse_args()

    acquire_lock("lock_intraday_scan")

    if args.universe:
        tickers = load_universe_from_file(args.universe)
        universe_src = f"universe file ({args.universe}, {len(tickers)} tickers)"
    elif args.tickers:
        tickers = args.tickers
        universe_src = f"explicit ({len(tickers)} tickers)"
    else:
        tickers = WATCHLIST
        universe_src = f"default watchlist ({len(tickers)} tickers)"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Forward Factor Scanner — {args.date}")
    print(f"Universe: {universe_src}")
    print(f"Source: {'yfinance' if args.no_ib else 'IB (yfinance fallback)'}")
    print()

    results = []
    for ticker in tickers:
        print(f"  {ticker:5s} ... ", end="", flush=True)
        row = scan_ticker(ticker, args.no_ib)
        if row:
            results.append(row)
            status = f"FF={row['ff_pct']:+6.1f}%  {'SIGNAL' if row['ff_signal'] else '      '}"
            print(status)
        else:
            print("no data")

    results.sort(key=lambda r: (r.get("ff_pct") or -9999), reverse=True)
    for i, r in enumerate(results):
        r["rank_by_ff"] = i + 1

    out_path = OUTPUT_DIR / f"{args.date}_ff_scan.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(results)

    print()
    signals = [r for r in results if r.get("ff_signal")]
    print(f"Saved: {out_path}")
    print(f"FF > 16% signals: {len(signals)} / {len(results)}")

    if signals:
        print("\n=== ENTRY CANDIDATES ===")
        print(f"{'Rank':<5} {'Ticker':<6} {'FF%':>7}  {'Front':>11} {'Back':>11}  {'Front IV':>9}  {'Fwd Vol':>9}")
        for r in signals:
            print(
                f"  {r['rank_by_ff']:<3}  {r['ticker']:<6}  {r['ff_pct']:>+6.1f}%"
                f"  {r['front_expiry']:>11}  {r['back_expiry']:>11}"
                f"  {r['front_iv']:>9.4f}  {r['forward_vol']:>9.4f}"
            )
    else:
        print("\nNo FF > 16% signals today — no calendar entries recommended.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
