# CSFF Practical Review — live funnel audit (2026-07-07, Kris)

Requested by Fabien: "theoretically working and good, but produces very
little actionable trades now that we are live — to be reviewed in practical
terms." Data: all trade_recommendations + universe files since go-live
2026-06-07 (14 scanner days), `ff_trade_scanner.py` source, registry T028.

## The funnel (2026-06-07 → 2026-07-07)

| stage | count | rate |
| --- | ---: | --- |
| Universe scanned (Dolt) | ~1,500 tickers/day | |
| Morning candidates (FF>16 + ML rank) | 31–127/day | |
| Intraday scanner rows | 433 over 14 days | ~31/day |
| `entry_ready` = True | 28 | 6%, ~2/day; **5 of 14 days = zero** |
| entry-ready AND min-leg volume ≥ 100 | **1 in the entire month** | |
| — at registry's documented gate (vol≥1000, width≤$0.50) | **0 in the entire month** | |

Fabien's lived experience ("very little actionable") is exactly right — and
it's ~1/month, not ~2/day.

## Root cause 1 — signal–liquidity anti-correlation (STRUCTURAL)

FF>16% (inverted single-name term structure) lives almost exclusively in
illiquid names. Median min-leg option volume across all 433 recommendations:
**4 contracts**; 95th percentile: 112. ETFs — the liquid names — appeared in
only 10/433 rows (1 ready). Earnings doesn't explain the ready set either
(2/16 matched rows flagged; median 30d to earnings). The inverted-term-
structure state and fillable option markets barely intersect.

**Implication for the backtest:** the 177k-trade EV (+19.8% at FF>20) is
mid-priced/no-slippage across ~1,500 tickers — i.e., the EV is concentrated
in exactly the names that cannot be filled. (Consistent with the known ML-
eval caveat: mid-priced no-slippage bottleneck.)

## Root cause 2 — `entry_ready` doesn't check volume (CODE ≠ DOCS)

`ff_trade_scanner.py:281`: `entry_ready = debit_positive and not stale and
not wide_spread` (+ live-FF≥16 kill at line 347). **No volume condition
exists in the code.** The registry documents "Volume ≥ 1,000 both legs" as
an entry gate — that gate was never implemented. So the daily report's
"ready" flag advertises 10-lot names as tradeable. Ready-row min-leg
volumes: 1,1,2,3,3,3,6,7,8,9,9,10,10,11,14,14,16,18,19,22,34,34,45,48,70,
79,81,212.

## Root cause 3 — operational cadence

Intraday scanner produced output on 14 of ~21 trading days since go-live
(missing 6/24–26, 7/01, 7/04, …). Secondary issue; fix only if the strategy
survives R1.

## Capacity math (the practical-terms answer)

Median ready debit $0.74 → $74 risk per 1-lot; backtest EV ≈ +20% of debit
≈ **$15/lot expected**. In names quoting 10–100 contracts/day, 10-lots IS
the market. Honest expected value as currently scoped: **~$50–150/month**
before the operational cost of a daily 3 PM manual session. As scoped
(broad single-name universe), CSFF cannot absorb meaningful capital — the
capacity constraint that protects the edge also caps it at pocket change.

## Recommendations (ranked)

- **R1 — kill-or-keep gate (next work item, ~half-day):** join
  `ff_all_trades.csv` (has no liquidity fields) to Dolt option volume;
  recompute backtest EV restricted to a FILLABLE subset (min-leg vol ≥ 250).
  If EV dies → CSFF single-name demoted to monitor-only. If EV survives →
  accept the honest cadence (~1 good trade/week-to-month), size it, cron it.
  Also answers the liquid-universe question in the same query: how often
  does FF>16 EVER fire on top-100-liquidity names (2024-25 Dolt history)?
- **R2 — truth in reporting (1h, after R1 verdict):** add a `fillable` tier
  (min-leg vol + width) to the scanner output; align registry docs with
  code. Non-breaking, additive column.
- **R3 — ops:** cron the intraday scanner — only if R1 keeps it.

## Verdict

The strategy's SIGNAL is not in question here (Campasano + 177k internal
backtest); its practical scope is. Live evidence says: as a broad
single-name scan it is a **paper tiger at real fills — ~1 fillable trade
per month at tiny size**. R1 decides whether a fillable core exists worth
keeping, or whether CSFF's real value was the validated forward-factor
MACHINERY (reusable for liquid-underlying variants, e.g., the double-
calendar/event work) rather than the standalone scan.

---

## R1 — pre-registered spec (2026-07-07, before the query)

Fillable subset = trades where BOTH legs' entry-date option volume ≥ 250
contracts (Dolt `option_chain`, exact date+ticker+expiry+strike match).
**KEEP bar (frozen):** fillable subset has ≥ 500 trades AND FF>20 mean
hold-EV ≥ +10% (half the headline +19.8%). Anything less → demote scan to
monitor-only; FF machinery remains the asset. Also reported: EV by FF tier
in the fillable subset, and FF>16 firing frequency on the top-100 tickers
by total 2024-25 option volume.

---

## ⚠️ ADDENDUM (same day, post-R1 investigation) — REVISED CONCLUSIONS

R1 as pre-registered is DATA-BLOCKED: Dolt `option_chain` has NO volume
column (live volumes come from IB, which keeps no history), and width was
measured INVALID as a depth proxy on the live month (best precision 26%).
The investigation then overturned parts of the review itself:

1. **Root Cause 1 is WEAKENED — the live "unfillable" evidence is
   CONTAMINATED by scan run-times.** File timestamps: 2026-07-02 ran at
   09:57 ET (27 min after open — thin early-session quotes), 2026-07-07 at
   16:58 ET (AFTER the close — stale/wide by construction; single-digit
   "volume" on ETF options was the tell). June files were batch-touched
   2026-06-30, original run times unknown. None verifiably hit the designed
   3:00–3:45 PM ET window. Widths/volumes in the rec files do not measure
   3:45 PM reality. Root Cause 3 (ops) is PROMOTED to primary.
2. **The backtest says the liquid core is REAL and RECENT** (my earlier
   in-session scaling error corrected; hold_mid_ret is already in %):
   liquid-list FF>20, mid-priced: 2024 n=1240 meanEV **+24.3%**, 2025
   n=2042 **+22.0%**, win ~54-55%; fired on 233/258 days in 2025
   (NFLX/CSCO/NKE/ORCL/TGT/WMT/AMZN...). Today's morning candidates include
   10/31 liquid names — the Dolt scan is not the filter.
3. **The open question collapses to one number: does +22% mid-EV survive
   real spread-crossing in the liquid subset?**

### Revised actions
- **R0 (ops, FIRST, no sign-off needed):** run `ff_trade_scanner.py` inside
  3:00–3:45 PM ET (cron) for 5–10 sessions → uncontaminated live
  width/volume on the liquid subset. Until then, all live-funnel liquidity
  claims are suspect.
- **R1b (amended spec — REQUIRES Fabien sign-off, data-substitution rule):**
  historical realistic-fill EV on the LIQUID-LIST subset (2024-25, FF>20)
  using Dolt bid/ask: entry/exit cross 50% of quoted spread per leg
  (matches scanner SLIPPAGE_FACTOR). Substitution implications: (a)
  liquidity = external name list, not per-contract volume (coarser); (b)
  EOD quoted spreads may be wider than 3:45 PM touch (conservative bias).
  KEEP bar unchanged in spirit: liquid FF>20 realistic-fill EV ≥ +10%.
- Verdict language "paper tiger" is WITHDRAWN pending R0+R1b. Capacity
  ceiling conclusion also suspended (it was computed off contaminated
  quotes).

---

## R1b — RESULT (2026-07-07): ❌ FAIL at every fill model — the edge lives inside the spread

`ff_r1b_liquid_fill_ev.py` / `ff_r1b_results.csv`. Liquid-list FF>20
2024-25, 3,282 trades, 2,746 priced (16% missing quotes, excluded honestly),
straddle-calendar 4-leg costing from Dolt EOD bid/ask. **Median spread cost
= 72.8% of the debit.**

| fill model | mean EV | median EV | win |
| --- | ---: | ---: | ---: |
| cross 50% of width (base spec) | −366% | −75% | 1% |
| cross 25% (patient combo) | −172% | −37% | 5% |
| cross 10% (near-perfect) | −56% | −13% | 21% |
| mid fills (= the backtest) | +21.8% | +1.8% | 54% |

Debit ≥ $2 subset is proportionally better but still fails everywhere
(−15% mean even at 10%). KEEP bar (≥ +10%): **FAIL by ~66-380pp.**

Notes: (a) live scanner trades 2-leg PUT calendars vs the backtest's 4-leg
straddle calendar — both debit and crossing cost roughly halve, so the
cost-to-debit ratio and the verdict are unchanged; (b) EOD widths are wider
than 3:45 PM touch (conservative), but the sensitivity shows even 5×
tighter fills don't save it.

## FINAL VERDICT

**The FF single-name calendar edge is real at mid and uncapturable through
the spread — in liquid names too.** This resolves the whole review: the
live scan produced few actionable trades because, at real fills, there were
never any. RECOMMENDATION (Fabien's call, registry stays `live` until he
rules): **demote T028 to monitor-only, stop the daily 3:45 PM session.**

Keepers:
- FF signal machinery = validated component (predicts mid-price IV-surface
  convergence; usable as an INPUT/timing filter for other structures, e.g.
  the DC/fast-events program — not as a standalone spread-crossing trade).
- The R1b fill-cost harness (`ff_r1b_liquid_fill_ev.py`) = REUSABLE: this is
  the fill-mechanics test every future options strategy (DC, timeflies)
  must pass BEFORE any mid-price backtest is believed. Fabien's thesis
  confirmed quantitatively: fill mechanics is the binding constraint of the
  options book.
- R0 cron REPURPOSED: 1-2 weeks of in-window (3:15 PM ET) scans to collect
  REAL intraday widths vs Dolt EOD widths → calibrates the fill model for
  all future options research. Then stop.

---

## NUANCE ADDENDUM (2026-07-08, Fabien's challenge: "tested and implemented by many people — there must be a more nuanced version")

Correct. What R1b PROVED is narrow: **scan-and-cross is dead** — taking every
FF>20 signal and paying the quoted leg spreads loses at any crossing
fraction. What it did NOT prove: that the premium is uncapturable. Three
specific nuances:

1. **Leg-sum costing overstates the real market.** Calendars trade as COMBO
   orders (IB BAG); market makers net the offsetting leg risk and quote the
   spread far tighter than the sum of four leg widths — often 2-5× tighter
   on liquid names. My harness priced leg-sum EOD widths = an UPPER BOUND
   on cost. The true combo width is unmeasured (Dolt has no combo book).
2. **It is a liquidity-provision premium.** The +22% mid EV is earned by
   whoever RESTS near mid, not whoever crosses. Capturing it means working
   patient combo limit orders and paying instead adverse selection +
   non-fill opportunity cost — quantities EOD data cannot price. Mid-price
   backtests (Campasano, Quantpedia — the "many people" evidence is mostly
   this same mid-price genre) implicitly assume perfect liquidity provision.
3. **Practitioners are selective, not systematic.** Real implementations
   take a few best-quality combos per week (tight markets, event-anchored),
   not every signal on 85 names. The average trade being uncapturable does
   not preclude the best-executed decile being positive.

**What survives the nuance:** the ALERT-AND-CROSS system (what T028's daily
session de facto was) stays dead; the capturable version, if it exists, is
a low-capacity ORDER-WORKING CRAFT — selective, combo-quoted, patient —
which cannot be validated from EOD data at all.

**R0 EXTENDED (the resolution):** capture IB calendar COMBO (BAG) bid/ask
at 3:15 PM ET for the day's top ~10 candidates alongside the leg quotes.
1-2 weeks of that measures the TRUE tradable width → recalibrate the
harness → the demote/keep decision is then made on the real number, not on
a leg-sum upper bound.

---

## ETF slice + Q&A addendum (2026-07-08)

**ETFs: the effect DOES appear, and it's the best recent subset.** FF>16 on
the 19 sector/index ETFs: 2025 = 461 trades over 172 distinct days (~2/3 of
days), mid EV +59% mean / +10% median, 60% win; 2024 = +71%/+15%, 60% win
(XLY/KRE/XLV/XLC/XLE lead). Pre-2024 counts not comparable (Dolt universe
break). **The R1b ETF costing (median 749% of debit) is the MODEL breaking,
not the market**: EOD closing quotes on penny-class ETF options are
fiction-wide and ETF calendar debits are tiny; real sector-ETF combo
markets are the tightest on the board. → The candidate SURVIVING version of
CSFF: **FF on liquid sector ETFs, executed as combo limit orders** —
untested pending real width data; R0 snapshots prioritize ETF candidates.

**Timing (why 3:15 PM):** (a) research is priced on EOD chains → near-close
execution minimizes live-vs-research divergence; (b) intraday option
liquidity is U-shaped — last hour tightest outside the noisy open; (c) one
desk session with EPSB. Honest counterpoint: the Dolt signal is ~20h stale
by then; scanner's live-FF recheck mitigates. EMPIRICAL SETTLEMENT: sample
twice (10:00 + 15:15 ET) for the first week and compare widths.

**QC combo backtest:** QC has minute OPRA leg NBBO (fixes the EOD-width
artifact) but NOBODY has historical complex-order-book data — naive QC
multi-leg fills just re-create leg-sum crossing with better quotes. Credible
design = TWO-STEP: (1) R0 live BAG snapshots measure combo-width/leg-sum
ratio k (~2 weeks); (2) QC minute backtest on the ETF subset with fills at
mid ± k·legsum/2. Scoped to ETFs to keep the run cheap. Not credible
without step 1.

---

## QC ETF combo-fill backtest — pre-registration (2026-07-08, Fabien's proposal; BEFORE code)

**Design (one QC run + debug allowance):**
- Universe: 9 liquid sector ETFs by FF firing frequency + liquidity: XLY,
  KRE, XLV, XLC, XLE, XLI, XLU, XLK, SPY. Period 2024-01 → present (matches
  the Dolt-era evidence window). Minute resolution.
- Signal: FF>20 computed in-algo from ATM IVs (BS inversion on mid quotes —
  NOT QC's lagged Greeks, per gotchas #). Entry 15:15 ET, one position per
  ticker, PUT calendar (2 legs, = live structure), ~30/60 DTE pairing as in
  the scanner.
- Exit: TP40 (best exit per Dolt sweep) + time stop at front expiry −1d.
- **Fill model = the bracketing trick:** execute at MID in QC, but RECORD
  quoted leg widths at entry & exit minutes per trade. Post-analysis applies
  the cost spectrum k ∈ {0, 0.25, 0.5, 1.0} × leg-sum → one run yields the
  whole table + **k\*** = the breakeven crossing fraction.
- Commissions $0.65/contract/way included in all arms.
- **Decision rule (frozen):** the R0 live BAG snapshots measure ACTUAL
  combo k. KEEP the ETF variant iff measured k < k\* with margin (k ≤
  0.8·k\*) AND EV at measured k ≥ +10% of debit. The backtest alone cannot
  keep it (fill-probability/adverse-selection unmodeled); it CAN kill it
  (if even k=0.25 intraday EV < 0, dead — no combo market beats quarter
  leg-sum systematically).
- Prereqs before code: load QC gotchas list (expiration filter bug,
  AskPrice/BidPrice, strike width, assignment gating).

---

## QC ETF combo-fill backtest — RESULT (2026-07-13): ❌ KILL per pre-registered rule

103 trades (XLC 20, XLU 17, XLY 16, XLK 12, XLV 11, XLI 9, XLE 9, KRE 7,
SPY 2), 2024-01→2026-07, minute-resolution real intraday quotes (fixes the
EOD-artifact that inflated ETF costs in R1b: median entry leg-sum width now
**229% of debit**, not R1b's 749% — still very wide, just no longer
fictional).

| k | mean EV | median EV | win |
| --- | ---: | ---: | ---: |
| 0.00 (mid) | +125.3% | +59.4% | 98% |
| **0.25 (kill test)** | **−72.9%** | **−21.6%** | **36%** |
| 0.50 (market order) | −271.0% | −121.0% | 5% |
| 1.00 (full width) | −667.3% | −326.7% | 3% |

**k\* (breakeven crossing fraction) = 0.158.** Per-ticker at k=0.25: only
XLE (+16.2%, 80% win, n=9), KRE (+6.2%, 70% win, n=7), SPY (+22.5%, n=2)
positive; XLC/XLU/XLV/XLI/XLY all negative, XLC worst (−247%, 20% win,
n=20).

**PRE-REGISTERED KILL TRIGGERED:** EV(k=0.25) < 0 → per the frozen decision
rule this backtest alone is sufficient to kill; live BAG combo-width
measurement (R0) is now MOOT for the general ETF scan — k\*=0.158 is far
tighter than any real combo market reasonably prices (real combos net
offsetting leg risk but do not typically quote inside 16% of summed leg
width on 30-60 DTE calendars). Two names (XLE, KRE) show a positive
per-ticker signal at n=9 and n=7 — sample too thin to act on, noted as a
loose end only.

## FINAL VERDICT — ENTIRE CSFF PRACTICAL REVIEW (2026-07-07 → 2026-07-13)

Every fill-realism test, at every step of increasing data fidelity, gives
the same answer: **the FF calendar edge is a mid-price artifact.** It does
not survive spread-crossing in single names (R1b), it does not survive
spread-crossing in liquid ETFs even with real intraday minute quotes and a
best-case combo-tightening assumption (QC backtest). Nuance stands
(liquidity-provision premium, resting-order craft, unmeasured true combo
depth) but the burden now sits entirely on an UNTESTABLE claim (adverse
selection cost of resting orders is favorable) rather than a testable one.

**RECOMMENDATION (final, awaiting Fabien ruling): demote T028 to
monitor-only; stop the daily 3:15 PM session; do not pursue the live BAG
snapshot workstream for the general scan.** XLE/KRE loose end: revisit only
if either shows up independently in the DC/fast-events or timeflies work.

**What survives and compounds:**
1. FF machinery (signal only) — reusable as a timing input elsewhere.
2. `ff_r1b_liquid_fill_ev.py` + `ff_etf_calendar_qc.py`/`ff_qc_post_analysis.py`
   — the k-spectrum bracketing method is now the STANDARD pre-flight test
   for any multi-leg options strategy before it is trusted at mid-price.
   Apply to DB2 multi-ticker and timeflies BEFORE further backtesting.
3. Methodological keeper: mid-price backtests of multi-leg option strategies
   need a fill-realism gate as a matter of course — this is now house
   process, not a one-off audit.
