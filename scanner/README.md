# CSFF Scanner — CLI Usage

## Scripts

| Script | Purpose | Typical Runtime |
|--------|---------|-----------------|
| `ff_universe_scan.py` | Overnight batch: scan all PG tickers for FF > threshold | 2-5 min |
| `ff_scanner.py` | Intraday: scan watchlist via IB/yfinance for FF > 16% | 30-60s |
| `ff_trade_scanner.py` | Price candidates: generate HTML reports via OptionTrader | 2-5 min |

## Connection

Scanner scripts connect to PostgreSQL on hal. Set env vars:

```bash
export CSFF_PGHOST=<hal-ip>         # default: Unix socket
export CSFF_PGPORT=5432
export CSFF_PGDATABASE=earningsvol
export CSFF_PGUSER=fabien
export CSFF_PGPASSWORD=<password>
```

Or pass `--pg-host` / `--pg-port` / `--pg-db` / `--pg-user` / `--pg-password`
directly (takes precedence over env vars).

## Lock Files

Each script acquires a lock before running. Lock directory defaults to
`/tmp` and can be overridden via `CSFF_LOCK` env var:

```bash
export CSFF_LOCK=/opt/euro_optionstrat/csff_data/locks
```

| Operation | Lock Name |
|-----------|-----------|
| Universe scan | `lock_universe_scan` |
| Intraday scan | `lock_intraday_scan` |
| Single-ticker refresh | `lock_realtime` |

## Dependencies

Install via pip:

```bash
pip install -r scanner/requirements.txt
```

Requires: `psycopg2-binary`, `numpy`, `scipy`, `pandas`, `yfinance`

## OptionTrader

`ff_scanner.py` and `ff_trade_scanner.py` call `option_chain.py` from the
OptionTrader project. Set `OPTTRADER_DIR` env var if not at default location:

```bash
export OPTTRADER_DIR=/opt/OptionTrader
```
