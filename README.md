# Calendar Spread Forward Factor (CSFF / T028)

> **Note:** The functional code in this repo (`web/`, `scanner/`, `static/csff/`,
> the systemd units under `scripts/`) has been merged into the
> [euro_optionstrat](https://github.com/scilear/euro-optionstrat) repository.
> This repo remains the historical/research source for the original CSFF
> scanner and the web integration design.

Forward Factor scanner and trade recommendation engine for S3 calendar
spread strategies.  Computes the Campasano (2018) Forward Factor from
option IV term structure to identify inverted vol curves suitable for
long calendar spreads.

## Where the functional code now lives

After the merge:

| Original CSFF path | New location in euro_optionstrat |
|---|---|
| `web/csff_handler.py` | `backend/csff/csff_handler.py` |
| `web/csff_service.py` | `backend/csff/csff_service.py` |
| `scanner/` | `scanner/` |
| `static/csff/` | `static/csff/` |
| `scripts/csff-*.service` / `*.timer` | `scripts/csff/` |

Deployment is managed by `portfoliomonitor/scripts/deploy/setup_csff.sh`.

## Structure

```
calendar-spread-forward-factor/
├── scanner/          CLI scanner scripts (universe scan, IV refresh, pricing)
│   ├── db.py                 Standalone PG connection module
│   ├── ff_universe_scan.py   Overnight batch: scan all tickers from PG
│   ├── ff_trade_scanner.py   Live pricing + HTML report generation
│   ├── ff_scanner.py         Intraday IV refresh via OptionTrader
│   ├── ff_ml_lr_coefs_v2.json        Straddle LR model coefficients
│   └── ff_ml_lr_coefs_put_v1.json    Put LR model coefficients
├── web/              Web integration module for optionstrat server
│   ├── csff_handler.py       HTTP route handler
│   └── csff_service.py       Scan orchestration + PG queries
├── static/csff/      Frontend (vanilla JS, no build tools)
│   ├── index.html            Report browser shell
│   ├── styles.css            Dark theme
│   ├── app.js                UI logic + API client
│   └── vendor/               Vendored Chart.js + noUiSlider
└── scripts/          Systemd units + deployment
    ├── setup_csff.sh                 Self-sufficient server setup
    ├── csff-universe-scan.service    Overnight batch scan
    ├── csff-universe-scan.timer      Mon–Fri 11:30 Paris
    ├── csff-trade-scanner.service    Intraday pricing + reports
    └── csff-trade-scanner.timer      Mon–Fri 21:00 Paris
```

## Usage

See `scanner/README.md` for CLI usage.
See `scripts/OPTIONSTRAT_CSFF_MERGE.md` for web deployment.
