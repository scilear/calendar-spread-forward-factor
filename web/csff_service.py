"""
CSFF business logic — scan orchestration, job tracking, ticker management,
PG queries for report metadata.

Runs scanner scripts as subprocesses with progress tracking via in-memory
job state. Three independent operations with separate lock files:
  - Universe scan (ff_universe_scan.py)
  - Intraday scan / price (ff_trade_scanner.py)
  - Single-ticker refresh (ff_trade_scanner.py --ticker X)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import date
from pathlib import Path

SCANNER_DIR = Path(__file__).resolve().parent.parent / "scanner"
DATA_DIR = Path(os.environ.get("CSFF_DATA_DIR", str(SCANNER_DIR.parent / "csff_data")))
REPORTS_DIR = Path(os.environ.get("CSFF_REPORTS_DIR", str(SCANNER_DIR.parent / "static" / "csff" / "reports")))

TICKER_FILE = DATA_DIR / "tickers.json"
READY_STATS_FILE = DATA_DIR / "ready_stats.json"

TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


class JobState:
    def __init__(self, job_id: str, scan_type: str):
        self.job_id = job_id
        self.scan_type = scan_type
        self.status = "queued"
        self.progress = ""
        self.result: dict | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self.completed_at: float | None = None


class CsffService:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobState] = {}
        self._scan_lock = threading.Lock()

    # ── Ticker management ────────────────────────────────────────────────────

    def get_tickers(self) -> list[str]:
        if TICKER_FILE.exists():
            try:
                data = json.loads(TICKER_FILE.read_text())
                return data if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError):
                pass
        return ["SPY", "QQQ", "IWM", "XLK", "XLF", "GLD", "TLT", "EFA"]

    def save_tickers(self, tickers_str: str) -> list[str]:
        parts = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        validated = [t for t in parts if TICKER_RE.match(t)]
        TICKER_FILE.write_text(json.dumps(validated, indent=2))
        return validated

    # ── Report listing ───────────────────────────────────────────────────────

    def list_reports(self, max_dates: int = 30) -> dict:
        if not REPORTS_DIR.exists():
            return {"dates": [], "latest": None, "total_reports": 0}
        date_dirs = sorted(
            (d for d in REPORTS_DIR.iterdir() if d.is_dir() and d.name[:4].isdigit()),
            reverse=True,
        )[:max_dates]
        dates = []
        total = 0
        for d in date_dirs:
            index_file = d / "index.html"
            if not index_file.exists():
                continue
            ticker_reports = sorted(f.stem.replace("_report", "") for f in d.glob("*_report.html"))
            dates.append({
                "date": d.name,
                "tickers": ticker_reports,
                "count": len(ticker_reports),
            })
            total += len(ticker_reports)
        latest = dates[0]["date"] if dates else None
        return {"dates": dates, "latest": latest, "total_reports": total}

    def get_date_index(self, date_str: str) -> dict | None:
        index_file = REPORTS_DIR / date_str / "index.html"
        if not index_file.exists():
            return None
        html = index_file.read_text(encoding="utf-8")
        ticker_reports = sorted(
            f.stem.replace("_report", "")
            for f in (REPORTS_DIR / date_str).glob("*_report.html")
        )
        return {"date": date_str, "tickers": ticker_reports, "html": html}

    def get_ticker_report(self, date_str: str, ticker: str) -> dict | None:
        report_file = REPORTS_DIR / date_str / f"{ticker}_report.html"
        if not report_file.exists():
            return None
        html = report_file.read_text(encoding="utf-8")
        return {"date": date_str, "ticker": ticker, "html": html}

    # ── Scan execution ───────────────────────────────────────────────────────

    def start_scan(self, scan_type: str = "universe", tickers_str: str | None = None) -> dict:
        if not self._scan_lock.acquire(blocking=False):
            return {"error": "scan already running", "status": "in_progress"}

        try:
            job_id = str(uuid.uuid4())
            job = JobState(job_id, scan_type)
            self._jobs[job_id] = job

            if scan_type == "universe":
                thread = threading.Thread(
                    target=self._run_universe_scan,
                    args=(job, tickers_str),
                    daemon=True,
                )
            elif scan_type == "intraday":
                thread = threading.Thread(
                    target=self._run_intraday_scan,
                    args=(job, tickers_str),
                    daemon=True,
                )
            else:
                self._scan_lock.release()
                return {"error": f"unknown scan type: {scan_type}"}

            thread.start()
            return {"job_id": job_id, "status": "queued"}
        except Exception as exc:
            self._scan_lock.release()
            return {"error": str(exc)}

    def _run_universe_scan(self, job: JobState, tickers_str: str | None):
        try:
            job.status = "running"
            cmd = [sys.executable or "python3", str(SCANNER_DIR / "ff_universe_scan.py")]
            env = os.environ.copy()
            if tickers_str:
                tickers = [t.strip() for t in tickers_str.split(",") if t.strip() and TICKER_RE.match(t.strip())]
                if tickers:
                    cmd.extend(["--tickers"] + tickers)
            cmd.extend(["--output-dir", str(REPORTS_DIR / "universe")])
            job.progress = "starting universe scan"
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800, env=env,
            )
            if result.returncode != 0:
                job.status = "failed"
                job.error = result.stderr.strip() or f"exit code {result.returncode}"
            else:
                job.status = "done"
                job.result = {"stdout": result.stdout.strip(), "exit_code": 0}
        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error = "timed out after 1800s"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.completed_at = time.time()
            self._scan_lock.release()

    def _run_intraday_scan(self, job: JobState, tickers_str: str | None):
        try:
            job.status = "running"
            universe_dir = REPORTS_DIR / "universe"
            latest_csv = universe_dir / "latest_candidates.csv"
            cmd = [sys.executable or "python3", str(SCANNER_DIR / "ff_trade_scanner.py")]
            env = os.environ.copy()
            if latest_csv.exists():
                cmd.extend(["--universe", str(latest_csv)])
            if tickers_str:
                tickers = [t.strip() for t in tickers_str.split(",") if t.strip() and TICKER_RE.match(t.strip())]
                if tickers:
                    cmd.extend(["--tickers"] + tickers)
            cmd.extend([
                "--reports-dir", str(REPORTS_DIR),
            ])
            job.progress = "starting intraday scan"
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env,
            )
            if result.returncode != 0:
                job.status = "failed"
                job.error = result.stderr.strip() or f"exit code {result.returncode}"
            else:
                job.status = "done"
                job.result = {"stdout": result.stdout.strip(), "exit_code": 0}
        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error = "timed out after 600s"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.completed_at = time.time()
            self._scan_lock.release()

    def refresh_ticker(self, ticker: str) -> dict:
        try:
            cmd = [
                sys.executable or "python3",
                str(SCANNER_DIR / "ff_trade_scanner.py"),
                "--ticker", ticker,
                "--reports-dir", str(REPORTS_DIR),
            ]
            env = os.environ.copy()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, env=env,
            )
            if result.returncode != 0:
                return {"error": result.stderr.strip() or f"exit code {result.returncode}"}
            report_data = self.get_ticker_report(date.today().isoformat(), ticker)
            if report_data:
                return report_data
            return {"ticker": ticker, "status": "refreshed"}
        except subprocess.TimeoutExpired:
            return {"error": "timed out after 120s"}
        except Exception as exc:
            return {"error": str(exc)}

    # ── Job status ──────────────────────────────────────────────────────────

    def get_job_status(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "scan_type": job.scan_type,
            "progress": job.progress,
            "error": job.error,
            "result": job.result,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }



