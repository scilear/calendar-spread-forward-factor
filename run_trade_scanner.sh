#!/bin/bash
# CSFF intraday trade scanner — MUST run inside 3:00-3:45 PM ET (R0, 2026-07-07)
# 21:15 Paris = 15:15 ET in summer. ⚠️ DST: US/EU switch dates differ (Mar/Nov);
# re-check this offset the weeks EU/US clocks diverge.
source /home/fabien/Documents/EarningsVolAnalysis/.venv/bin/activate
cd /home/fabien/Documents/InvestmentDeskAgents/research/strategies/calendar-spread-forward-factor
echo "=== scanner run $(date '+%Y-%m-%d %H:%M %Z') ==="
python3 ff_trade_scanner.py --universe universe/latest_candidates.csv
