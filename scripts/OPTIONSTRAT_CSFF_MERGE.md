# Merging CSFF into Optionstrat

## Architecture

CSFF lives at `/opt/euro_optionstrat/csff/` (git clone of
`scilear/calendar-spread-forward-factor`). The optionstrat HTTP server
imports the web module and serves static assets from the same process.

### Integration Points

1. **`backend/csff_handler.py`** — symlinked from `web/csff_handler.py` into
   `backend/`. Imported by `backend/http_handler.py` with graceful fallback.
2. **`backend/csff_service.py`** — symlinked from `web/csff_service.py`.
3. **`static/csff/`** — symlinked to repo's `static/csff/` directory.
4. **`/opt/euro_optionstrat/csff_data/`** — persistent state directory
   (tickers.json, ready_stats.json) outside of repo checkout.

## Optionstrat Server Integration

In `backend/http_handler.py`, add:

```python
try:
    from .csff_handler import CsffHandler
    _csff = CsffHandler()
    print("[csff] CSFF module loaded")
except ImportError as e:
    _csff = None
    print(f"[csff] CSFF module unavailable (optionstrat continues without CSFF): {e}")
```

In `do_GET()` and `do_POST()`, add CSFF routing before static file fallback:

```python
def do_GET(self):
    parsed = urlparse(self.path)
    if parsed.path.startswith("/csff/"):
        status, data, ct = _csff.dispatch("GET", self.path)
        # ... respond with status, data, ct
        return
    # ... existing routing ...

def do_POST(self):
    parsed = urlparse(self.path)
    if parsed.path.startswith("/csff/"):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        status, data, ct = _csff.dispatch("POST", self.path, body)
        # ... respond ...
        return
```

## Env File

Create `/etc/portfoliomonitor/csff.env` (mode 600):

```bash
CSFF_PG_HOST=<hal-ip>
CSFF_PG_PORT=5432
CSFF_PG_DB=earningsvol
CSFF_PG_USER=fabien
CSFF_PG_PASSWORD=<password>
CSFF_REPORTS_DIR=/opt/euro_optionstrat/static/csff/reports
CSFF_UNIVERSE_DIR=/opt/euro_optionstrat/static/csff/reports/universe
CSFF_DATA_DIR=/opt/euro_optionstrat/csff_data
CSFF_LOCK=/tmp
OPTTRADER_DIR=/opt/OptionTrader
```

Add `EnvironmentFile=-/etc/portfoliomonitor/csff.env` to
`optionstrat.service`.

## URL Structure

| Path | Purpose |
|------|---------|
| `/csff/` | Report browser frontend (SPA) |
| `/csff/api/reports` | JSON: list available report dates |
| `/csff/api/report?date=X&ticker=Y` | JSON: per-ticker report data |
| `/csff/api/report?date=X` | JSON: date index |
| `/csff/api/tickers` | GET/POST: manage tracked tickers |
| `/csff/api/scan` | POST: trigger universe or intraday scan |
| `/csff/api/refresh?ticker=X` | POST: single-ticker live refresh |
| `/csff/api/status?job_id=UUID` | GET: scan job progress |
