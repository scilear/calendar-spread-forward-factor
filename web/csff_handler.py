"""
CSFF HTTP handler — routes for the Calendar Spread Forward Factor web UI.

Integrates into the optionstrat server via a route table pattern.
All CSFF endpoints are prefixed with /csff/.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .csff_service import CsffService

STATIC_DIR = Path(__file__).resolve().parent.parent / "static" / "csff"
REPORTS_DIR = Path(os.environ.get("CSFF_REPORTS_DIR", str(STATIC_DIR / "reports")))


class CsffHandler:
    ROUTES = [
        (r"^/csff/api/reports$",       "list_reports"),
        (r"^/csff/api/report$",        "get_report"),
        (r"^/csff/api/tickers$",       "handle_tickers"),
        (r"^/csff/api/scan$",          "run_scan"),
        (r"^/csff/api/refresh$",       "refresh_ticker"),
        (r"^/csff/api/status$",        "job_status"),
        (r"^/csff/vendor/",            "serve_vendor"),
        (r"^/csff/",                   "serve_static"),
    ]

    CONTENT_TYPES = {
        ".html": "text/html; charset=utf-8",
        ".css":  "text/css; charset=utf-8",
        ".js":   "application/javascript; charset=utf-8",
        ".json": "application/json",
        ".png":  "image/png",
        ".svg":  "image/svg+xml",
    }

    def __init__(self):
        try:
            self.service = CsffService()
            self.available = True
        except Exception as exc:
            print(f"[csff] Service init failed: {exc}")
            self.service = None
            self.available = False

    def dispatch(self, method: str, path: str, body: bytes | None = None) -> tuple[int, dict | str, str]:
        if not self.available:
            return (503, {"error": "csff_unavailable", "detail": "CSFF service not initialized"}, "application/json")

        parsed = urlparse(path)
        query = parse_qs(parsed.query)

        for pattern, handler_name in self.ROUTES:
            if re.match(pattern, parsed.path):
                handler = getattr(self, f"_handle_{handler_name}", None)
                if handler:
                    try:
                        return handler(method, path, query, body)
                    except Exception as exc:
                        traceback.print_exc()
                        return (500, {"error": "internal_error", "detail": str(exc)}, "application/json")

        return (404, {"error": "not_found"}, "application/json")

    def _json_response(self, status: int, data: dict) -> tuple[int, dict, str]:
        return (status, data, "application/json")

    def _html_response(self, status: int, html: str) -> tuple[int, str, str]:
        return (status, html, "text/html; charset=utf-8")

    def _handle_list_reports(self, method, path, query, body):
        if method != "GET":
            return self._json_response(405, {"error": "method_not_allowed"})
        max_dates = int(query.get("max", ["30"])[0])
        reports = self.service.list_reports(max_dates=max_dates)
        return self._json_response(200, reports)

    def _handle_get_report(self, method, path, query, body):
        if method != "GET":
            return self._json_response(405, {"error": "method_not_allowed"})
        date_str = query.get("date", [None])[0]
        ticker = query.get("ticker", [None])[0]

        if date_str and ticker:
            data = self.service.get_ticker_report(date_str, ticker)
            if data:
                return self._json_response(200, data)
            return self._json_response(404, {"error": "report_not_found", "detail": f"No report for {ticker} on {date_str}"})

        if date_str:
            data = self.service.get_date_index(date_str)
            if data:
                return self._json_response(200, data)
            return self._json_response(404, {"error": "date_not_found", "detail": f"No reports for {date_str}"})

        return self._json_response(400, {"error": "missing_params", "detail": "Provide ?date=Y-m-d or ?date=Y-m-d&ticker=X"})

    def _handle_handle_tickers(self, method, path, query, body):
        if method == "GET":
            tickers = self.service.get_tickers()
            return self._json_response(200, {"tickers": tickers})

        if method == "POST":
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return self._json_response(400, {"error": "invalid_json"})
            tickers_raw = data.get("tickers", "")
            if not isinstance(tickers_raw, str):
                return self._json_response(400, {"error": "invalid_format", "detail": "tickers must be a comma-separated string"})
            validated = self.service.save_tickers(tickers_raw)
            return self._json_response(200, {"tickers": validated})

        return self._json_response(405, {"error": "method_not_allowed"})

    def _handle_run_scan(self, method, path, query, body):
        if method != "POST":
            return self._json_response(405, {"error": "method_not_allowed"})
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            data = {}
        tickers_str = data.get("tickers", query.get("tickers", [None])[0])
        scan_type = data.get("type", query.get("type", ["universe"])[0])
        if isinstance(tickers_str, list):
            tickers_str = ",".join(tickers_str)
        result = self.service.start_scan(scan_type=scan_type, tickers_str=tickers_str)
        if "error" in result:
            status = 409 if "already running" in result.get("error", "") else 400
            return self._json_response(status, result)
        return self._json_response(202, result)

    def _handle_refresh_ticker(self, method, path, query, body):
        if method != "POST":
            return self._json_response(405, {"error": "method_not_allowed"})
        ticker = query.get("ticker", [None])[0]
        if not ticker:
            return self._json_response(400, {"error": "missing_ticker", "detail": "Provide ?ticker=X"})
        if not re.match(r"^[A-Z]{1,5}$", ticker):
            return self._json_response(400, {"error": "invalid_ticker", "detail": f"Invalid ticker: {ticker}"})
        result = self.service.refresh_ticker(ticker)
        if "error" in result:
            return self._json_response(500, result)
        return self._json_response(200, result)

    def _handle_job_status(self, method, path, query, body):
        if method != "GET":
            return self._json_response(405, {"error": "method_not_allowed"})
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            return self._json_response(400, {"error": "missing_job_id"})
        status = self.service.get_job_status(job_id)
        if not status:
            return self._json_response(404, {"error": "job_not_found"})
        return self._json_response(200, status)

    def _serve_static_file(self, file_path: Path) -> tuple[int, bytes | str, str]:
        if not file_path.exists() or not file_path.is_file():
            return (404, "Not found", "text/plain")
        suffix = file_path.suffix
        content_type = self.CONTENT_TYPES.get(suffix, "application/octet-stream")
        if suffix in (".html", ".css", ".js"):
            content = file_path.read_text(encoding="utf-8")
        else:
            content = file_path.read_bytes()
        return (200, content, content_type)

    def _handle_serve_static(self, method, path, query, body):
        if method != "GET":
            return self._json_response(405, {"error": "method_not_allowed"})
        if path == "/csff/" or path == "/csff":
            file_path = STATIC_DIR / "index.html"
        else:
            relative = path[len("/csff/"):]
            file_path = STATIC_DIR / relative
        return self._serve_static_file(file_path)

    def _handle_serve_vendor(self, method, path, query, body):
        if method != "GET":
            return self._json_response(405, {"error": "method_not_allowed"})
        relative = path[len("/csff/"):]
        file_path = STATIC_DIR / relative
        return self._serve_static_file(file_path)
