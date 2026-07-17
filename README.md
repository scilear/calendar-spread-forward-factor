# Calendar Spread — Forward Factor (CSFF / T028)

**Source:** Campasano & Simon, SSRN 3240028 · Predicting Alpha / Sean Ryan, "The Calendar Strategy"  
**Status:** LIVE — Scanner operational as of 2026-06-07 | ML scoring v2 as of 2026-06-18  
**Registry:** [[data/strategy-registry/CSFF.md]] (T028)

---

## Core Idea

Buy long calendar spreads when the **Forward Factor (FF) > 16%** — front-month IV is
disproportionately elevated relative to back-month IV, indicating inverted term structure.

```
FF = (Front IV − σ_fwd) / Front IV × 100
σ_fwd = sqrt(max(0, (T2·σ²₂ − T1·σ²₁) / (T2−T1)))
```

Campasano validated this signal in deciles 9–10 (2000–2018). Internal backtest on 177k
trades (2020–2025) confirms FF > 20% produces mean EV +19.8% vs all-trades baseline +3.9%.

---

## Trade Structures

| Parameter | Value |
|-----------|-------|
| Single calendar | Sell front ATM straddle + buy back ATM straddle (same strike) |
| Double calendar | Two strikes: spot ± 1σ expected move |
| Front expiry | 22–45 DTE |
| Back expiry | 46–80 DTE |
| Entry | Net debit — defined max loss, no tail exposure |

**Single vs Double:**
- **Single** — Stocks with detectable directional drift (slope×R² ≥ 0.3). One tent at ATM.
- **Double** — ETFs + range-bound stocks (slope×R² < 0.3). Two tents at spot ± 1σ. Exploits IV contraction + mean reversion.
- **Skip** — Any ticker where earnings fall before front expiry. IV crush destroys the position.

---

## Scanner Toolkit

**Venv:** `source /home/fabien/Documents/EarningsVolAnalysis/.venv/bin/activate`  
**Working dir:** `research/strategies/calendar-spread-forward-factor/`

| File | Purpose |
|------|---------|
| `ff_universe_scan.py` | Overnight Dolt batch scan → ML-ranked candidates CSV |
| `ff_trade_scanner.py` | Live chain pricing + dual-ML HTML reports per candidate |
| `ff_scanner.py` | Optional intraday live IV refresh of candidates |
| `ff_full_backtest.py` | Full 177k-trade backtest engine (2020–2025, 227 tickers) |
| `ff_exit_sweep.py` | Exit strategy optimizer (tests 15 strategies vs baseline) |
| `ff_ml_v2.py` | ML training harness — straddle + put models |
| `ff_ml_lr_coefs_v2.json` | Deployed straddle LR model (19 features) |
| `ff_ml_lr_coefs_put_v1.json` | Deployed put-calendar LR model (19 features) |

---

## Daily Workflow

```bash
source /home/fabien/Documents/EarningsVolAnalysis/.venv/bin/activate
cd research/strategies/calendar-spread-forward-factor

# Step 1 — After Dolt updates (~11:30 Paris / 05:30 ET)
python3 ff_universe_scan.py
# → universe/latest_candidates.csv  (sorted by ml_score descending)

# Step 2 — Intraday (~3:00–3:45 PM ET, live market data)
python3 ff_trade_scanner.py --universe universe/latest_candidates.csv
# → trade_recommendations/YYYY-MM-DD_trade_recommendations.csv
# → reports/TICKER_YYYY-MM-DD_report.html  (one per candidate)
```

---

## Ranking: ML Score (Primary)

Candidates are ranked by `ml_score` — P(hold return > +50%) from the straddle LR v2 model.

### Why ML-first

The full backtest (177k trades) revealed that **DTE structure dominates raw FF level** as a predictor.
The composite score (heuristic) was replaced as the primary sort key.

| Metric | Value |
|--------|-------|
| Model | LR bin_bigwin — P(straddle return > +50%) |
| AUC (OOF) | 0.795 |
| EV top-20% | +18.9% (vs baseline +3.9%) |
| EV top-10% | +31.6% |
| EV top-5% | +53% (~4 deduped new positions/day) |
| Year-by-year AUC | 0.76–0.82 (2021–2025) |

### Top features (by |coef|)

1. `t_fwd_ratio` (+7.53) — forward DTE / back DTE. **DTE structure is the primary edge, not raw FF.**
2. `t_fwd` (−4.18), `t_front` (+4.03) — DTE pair
3. `ff_x_tfwd` (−1.99) — FF × √t_fwd interaction (penalizes extreme FF with long forward period)
4. `t_back` (−1.74)
5. `entry_ff` (+1.43) — raw FF is 6th, real but not dominant
6. `entry_debit` (+0.65), `debit_ratio` (−0.62) — sizing features

**Key implication:** A calendar with FF=25% and optimal DTE ratio (t_fwd/t_back ≈ 0.5) outperforms
one with FF=35% and poor DTE structure. Do not filter purely on FF level.

### CSV columns

| Column | Meaning |
|--------|---------|
| `ml_score` | P(bigwin) from straddle LR v2 — primary sort key |
| `ml_pct` | Percentile rank within today's candidates (100 = best) |
| `composite_score` | Heuristic 5-component score (diagnostic, not primary) |
| `ff_pct` | Current Forward Factor % |
| `back_straddle` / `entry_debit` / `max_profit` | BS ATM straddle approximation (ML inputs) |

---

## Dual ML Scoring (Trade Scanner)

The trade scanner produces two ML scores per candidate — both shown in HTML reports and index table.

| Score | Model | Inputs |
|-------|-------|--------|
| Straddle P(bigwin) | `ff_ml_lr_coefs_v2.json` | `back_straddle_mid`, `straddle_fill_mid` |
| Put P(bigwin) | `ff_ml_lr_coefs_put_v1.json` | `back_put_mid`, `fill_mid` |

The **straddle score is primary**. The put score uses the same features but trained on put-sized inputs
(back_put ≈ back_straddle/2 at ATM). Coefficients for `log_debit` and `log_back_straddle` flip sign
between the two models — put pricing carries different skew information.

Both scores display as colored badges in the HTML report (green ≥45%, yellow ≥35%, red <35%).

---

## Backtest Summary (ff_full_backtest.py)

**Dataset:** 227 tickers × 1,045 Dolt dates × 2020-01-04 to 2025-12-31  
**Trades:** 177,399 entered | 121,048 with exits  
**Exit mechanics:**
- *Chain-roll:* front or back leg disappears from Dolt 3-slot chain → mark at last observable mid
- *Expiry:* front has expired → value = back_straddle − |spot − strike|

**FF distribution (n=176k):** median −1.5%, FF>16% = 21.9%, FF>20% = 18.2%

| FF tier | Trades | Mean EV |
|---------|--------|---------|
| All trades | 177,399 | +3.9% |
| FF > 0% | 82k | +5.8% |
| FF > 16% | 38k | +15.6% |
| FF > 20% | 32k | +19.8% |

**Best exit:** TP40 (take profit at 40% of max_profit) adds +3.5pp EV over hold-to-chain-roll for FF>20% entries.

**Min-debit gate DESTROYS EV** — cheap debits carry the big-win tail. Do not filter on minimum debit.

---

## ML Models (ff_ml_v2.py)

### Training

```bash
python3 ff_ml_v2.py --model lr --targets bin_bigwin
```

- Loads `ff_all_trades.csv` (177k trades) + `ff_exit_sim.csv` (TP40 strategy)
- 5-fold purged/embargoed walk-forward (14-day embargo, year-by-year folds)
- Exports both models automatically

### Model files

| File | Description |
|------|-------------|
| `ff_ml_lr_coefs_v2.json` | Straddle LR — intercept, 19 coefs, scaler mean/std |
| `ff_ml_lr_coefs_put_v1.json` | Put LR — same structure, inputs halved (put ≈ straddle/2) |
| `ff_all_trades.csv` | 177k trades from full backtest |
| `ff_exit_sim.csv` | TP40 exit simulation results |

### Put model rationale

At ATM, put ≈ straddle/2 by put-call parity. Return percentages are approximately equal between
straddle and put calendars (both numerator and denominator scale by 1/2). The put model is trained
on the same `bin_bigwin` targets with halved `entry_debit` and `back_straddle_entry` features.
OOF AUC = 0.7952 (identical to straddle model). The divergence in `log_debit` / `log_back_straddle`
coefficients captures the different skew information embedded in put vs. straddle pricing.

---

## Universe

**Source:** All tickers in Dolt `options.option_chain` with ≥ 10 valid IV rows on the scan date.  
**Typical size:** ~1,512 tickers (2024–2025 Dolt data).  
**ETFs present:** SPY, DIA, MDY, XLK, XLF, XLE, XLV, XLI, XLP, XLU, XLB, XLRE, XLC, XLY, XBI, XRT, XHB, XME, XOP, KRE, XSD, XAR (plus others starting with X that are stocks, not ETFs).  
**ETFs absent:** QQQ, IWM, GLD, TLT, HYG, EEM, and other non-sector ETFs — not in Dolt.

ETFs present are routed to `double_calendar` structure. `is_etf = 1.0` is one of the 19 ML features (coefficient +0.055 — small positive).

---

## Entry Gates (trade scanner)

| Gate | Threshold | Source |
|------|-----------|--------|
| FF ≥ 16% | Hard kill (returns `None`) | Live chain |
| Volume | ≥ 1,000 both legs | Live chain |
| Bid-ask width | ≤ $0.50 per leg | Live chain |
| Earnings risk | Flagged, not blocked | Dolt earnings DB |

---

## File Structure

```
calendar-spread-forward-factor/
├── ff_universe_scan.py          # overnight scan → ML-ranked candidates
├── ff_trade_scanner.py          # live pricing → dual-ML HTML reports
├── ff_scanner.py                # optional intraday IV refresh
├── ff_full_backtest.py          # 177k-trade backtest engine
├── ff_exit_sweep.py             # exit strategy optimizer
├── ff_ml_v2.py                  # ML training (straddle + put models)
├── ff_ml_lr_coefs_v2.json       # straddle LR deployment artifact
├── ff_ml_lr_coefs_put_v1.json   # put LR deployment artifact
├── ff_all_trades.csv            # backtest output (177k trades)
├── ff_exit_sim.csv              # TP40 exit simulation
├── universe/
│   ├── YYYY-MM-DD_universe.json     # full FF time series per ticker
│   ├── YYYY-MM-DD_candidates.csv    # ML-ranked candidates
│   └── latest_candidates.csv        # copy for intraday use
├── daily_scans/                 # intraday IV scan outputs
├── trade_recommendations/       # priced spread CSVs
└── reports/                     # HTML trade reports (Chart.js FF history)
```

---

## Key Nuances

**DTE structure > FF level:**  
t_fwd_ratio (= (back_dte − front_dte) / back_dte) is the #1 ML feature. Targeting ~0.5 ratio
(28-day forward period in a 56-day back) is more important than maximising FF%.

**FF trend:**  
FF *declining* from a high level is better entry timing than peak FF.  
FF rising because *back IV* is rising = sustainable (back vol is cheap).  
FF rising because *front IV* alone is spiking = unstable (event priced in, skip).

**IV/HV ratio:**  
IV/HV > 4.0 at entry = worst failure case. Front IV stays elevated and the spread doesn't compress.

**Min-debit gate:**  
Do not add a minimum debit filter. Cheap debits carry the big-win tail. Removing low-debit trades
degrades EV significantly (confirmed in backtest).

---

## Status Log

| Date | Milestone |
|------|-----------|
| 2026-05-04 | Research started, concept validated |
| 2026-06-05 | FF calculator (`ff_scanner.py`) |
| 2026-06-06 | Trade pricing + HTML reports (`ff_trade_scanner.py`) |
| 2026-06-07 | Universe scan live — 1,517 tickers, batch Dolt queries |
| 2026-06-08 | Composite ranking: earnings gate, IV/HV, trend, FF driver, suggested strikes |
| 2026-06-17 | Full backtest: 177k trades, 227 tickers, 2020–2025 (`ff_full_backtest.py`) |
| 2026-06-17 | Exit sweep: TP40 best exit (+3.5pp EV for FF>20%) (`ff_exit_sweep.py`) |
| 2026-06-17 | ML v2: LR bin_bigwin AUC 0.795, 19 features, honest OOF (`ff_ml_v2.py`) |
| 2026-06-18 | Universe scan: ML score replaces composite as primary sort; dual BS straddle fields added |
| 2026-06-18 | Trade scanner: dual ML scoring (straddle + put), HTML reports updated |
| 2026-06-18 | Put model trained + deployed (`ff_ml_lr_coefs_put_v1.json`) |

---

## Backlog

- [ ] TP40 exit: integrate into live trade scanner monitoring (currently backtest-only)
- [ ] 52-week IV Rank (scanner uses 20-day window)
- [ ] Double calendar live pricing in trade scanner (currently prices single straddle leg only)
- [ ] Earnings hard gate (currently flags `earnings_risk`, doesn't block)
- [ ] Walk-forward ML retraining schedule (model trained on 2020–2025, refresh cadence TBD)

---

*Research started: 2026-05-04 · Scanner live: 2026-06-07 · ML scoring live: 2026-06-18*

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
