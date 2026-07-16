#!/usr/bin/env bash
#
# setup_csff.sh — Self-sufficient deployment script for CSFF online tool.
#
# Run this ON THE PRODUCTION SERVER (107.175.67.48) as the portfolio user
# after the repo has been cloned.  Configures:
#   1.  Python venv + dependencies
#   2.  Data directories (csff_data, reports/universe)
#   3.  Optionstrat merge (copies web/ modules into optionstrat backend)
#   4.  Systemd timers for scheduled scans
#   5.  Vendor assets (Chart.js, noUiSlider — already vendored in repo)
#
# Usage:
#   sudo -u portfolio bash scripts/setup_csff.sh
#
# Prerequisites:
#   - Git clone exists at /opt/euro_optionstrat/csff
#   - optionstrat service is running (will be restarted)
#   - /etc/portfoliomonitor/csff.env exists with PG credentials
#

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OPTSTRAT_DIR="/opt/euro_optionstrat"
VENV_DIR="$OPTSTRAT_DIR/.venv"
CSFF_DATA_DIR="$OPTSTRAT_DIR/csff_data"
REPORTS_DIR="$OPTSTRAT_DIR/static/csff/reports"
ENV_FILE="/etc/portfoliomonitor/csff.env"

echo "=== CSFF Setup ==="
echo "Repo:      $REPO_DIR"
echo "Target:    $OPTSTRAT_DIR"
echo "Ven:   $VENV_DIR"
echo "Data dir:  $CSFF_DATA_DIR"
echo "Reports:   $REPORTS_DIR"
echo "Env:       $ENV_FILE"
echo ""

# ── 1. Verify env file ──────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    echo "WARNING: $ENV_FILE not found."
    echo "Create it with:"
    cat <<'EOF'
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
EOF
    echo ""
    echo "Continuing without env file (scanner scripts will use defaults)..."
fi

# ── 2. Create data directories ──────────────────────────────────────
mkdir -p "$CSFF_DATA_DIR"
mkdir -p "$REPORTS_DIR/universe"
echo "[OK] Data directories created"

# ── 3. Install Python dependencies ──────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --quiet -r "$REPO_DIR/scanner/requirements.txt"
echo "[OK] Python dependencies installed"

# ── 4. Symlink CSFF into optionstrat ────────────────────────────────
# The repo is already cloned at $OPTSTRAT_DIR/csff/
# We need to make the web/ module importable from optionstrat backend.
# Option A: symlink individual files
# Option B: symlink web/ into backend/
BACKEND_DIR="$OPTSTRAT_DIR/backend"
if [ -d "$BACKEND_DIR" ]; then
    ln -sf "$REPO_DIR/web/csff_handler.py" "$BACKEND_DIR/csff_handler.py"
    ln -sf "$REPO_DIR/web/csff_service.py" "$BACKEND_DIR/csff_service.py"
    echo "[OK] Web module symlinked into optionstrat backend"
else
    echo "WARNING: $BACKEND_DIR not found — skipping web module symlink"
fi

# ── 5. Symlink static assets ────────────────────────────────────────
STATIC_DIR="$OPTSTRAT_DIR/static/csff"
mkdir -p "$(dirname "$STATIC_DIR")"
if [ ! -L "$STATIC_DIR" ] && [ ! -d "$STATIC_DIR" ]; then
    ln -sf "$REPO_DIR/static/csff" "$STATIC_DIR"
    echo "[OK] Static assets symlinked"
elif [ -L "$STATIC_DIR" ]; then
    echo "[OK] Static assets symlink already exists"
else
    echo "WARNING: $STATIC_DIR exists and is not a symlink — skipping"
fi

# ── 6. Restart optionstrat service ──────────────────────────────────
echo "Restarting optionstrat service..."
sudo systemctl daemon-reload
sudo systemctl restart optionstrat
echo "[OK] optionstrat restarted"

# ── 7. Install systemd timers ───────────────────────────────────────
echo "Installing systemd timers..."
sudo cp "$REPO_DIR/scripts/csff-universe-scan.service" /etc/systemd/system/
sudo cp "$REPO_DIR/scripts/csff-universe-scan.timer"   /etc/systemd/system/
sudo cp "$REPO_DIR/scripts/csff-trade-scanner.service" /etc/systemd/system/
sudo cp "$REPO_DIR/scripts/csff-trade-scanner.timer"    /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now csff-universe-scan.timer 2>/dev/null || true
sudo systemctl enable --now csff-trade-scanner.timer 2>/dev/null || true
echo "[OK] Systemd timers installed"

# ── 8. Verify ───────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
sleep 2
if curl -sf "http://127.0.0.1:8765/csff/" > /dev/null 2>&1; then
    echo "[OK] /csff/ is reachable"
else
    echo "WARNING: /csff/ is not yet reachable — check optionstrat logs"
fi

if curl -sf "http://127.0.0.1:8765/csff/api/reports" > /dev/null 2>&1; then
    echo "[OK] /csff/api/reports responds"
else
    echo "WARNING: /csff/api/reports failed — check CSFF module import"
fi

echo ""
echo "=== Setup complete ==="
echo "CSFF is available at: https://optionstrat.alphangel.com/csff/"
