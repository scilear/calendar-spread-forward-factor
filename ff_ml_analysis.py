#!/usr/bin/env python3
"""
ff_ml_analysis.py — ML-based analysis of ff_backtest results.

Runs classification (win/loss) and regression (P&L%) for both
straddle calendar and ATM put calendar structures.

Models: Logistic Regression, Random Forest, XGBoost + SHAP.
Validation: time-series split (train on earlier dates, test on later).

Output:
  backtest/results/ml_report.html   — visual report
  backtest/results/ml_findings.txt  — text summary
"""

import csv
import json
import math
import os
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "backtest" / "results"
TRADES_CSV  = RESULTS_DIR / "trades.csv"
ML_HTML     = RESULTS_DIR / "ml_report.html"
ML_TEXT     = RESULTS_DIR / "ml_findings.txt"

# Cap extremes so one outlier doesn't dominate regression
PNL_PCT_CLIP = 500.0  # clip pnl_pct at ±500%
MIN_DEBIT    = 0.10   # skip tiny-debit trades where pct is meaningless
MIN_DAYS     = 1      # minimum days_held (exit-date bug guard)


# ── Data Loading ──────────────────────────────────────────────────────────────

def _flt(v):
    try: return float(v)
    except: return None


def load_data():
    rows = []
    with open(TRADES_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    records = []
    for r in rows:
        # Only valid, non-earnings, priced trades for both structures
        if r.get("earnings_risk", "").lower() in ("true", "1"):
            continue
        dh = _flt(r.get("days_held"))
        if dh is None or dh < MIN_DAYS:
            continue

        sd = r.get("scan_date", "")
        try:
            scan_dt = datetime.strptime(sd, "%Y-%m-%d").date()
        except ValueError:
            continue

        fi = _flt(r.get("front_iv_entry"))
        bi = _flt(r.get("back_iv_entry"))

        spot         = _flt(r.get("atm_strike"))
        entry_debit  = _flt(r.get("entry_mid_debit"))
        put_debit    = _flt(r.get("put_entry_mid_debit"))

        # Normalize debit by stock price so $40 stock ≠ $400 stock in features
        debit_pct_spot     = (entry_debit / spot * 100) if (entry_debit and spot) else None
        put_debit_pct_spot = (put_debit   / spot * 100) if (put_debit   and spot) else None
        # IB commission ($0.052/share) as % of debit — directly measures how much commission
        # eats into the trade. High value (cheap debit) = commission drag; low value = negligible.
        IB_COMM_PER_SHARE = 5.20 / 100.0
        comm_to_debit_pct     = (IB_COMM_PER_SHARE / entry_debit * 100) if entry_debit else None
        comm_to_debit_pct_put = (IB_COMM_PER_SHARE / put_debit   * 100) if put_debit   else None

        rec = {
            "scan_date":          scan_dt,
            "period":             r.get("period", "OOS"),
            "ticker":             r.get("ticker", ""),
            "atm_strike":         spot,
            # raw signal features
            "ff_pct":             _flt(r.get("ff_pct")),
            "ff_quality_pts":     _flt(r.get("ff_quality_pts")),
            "front_dte":          _flt(r.get("front_dte")),
            "back_dte":           _flt(r.get("back_dte")),
            "front_iv":           fi,
            "back_iv":            bi,
            "iv_spread":          (bi - fi) if (bi is not None and fi is not None) else None,
            "iv_hv_ratio":        _flt(r.get("iv_hv_ratio")),
            "iv_rank_20d":        _flt(r.get("iv_rank_20d")),
            "trend_strength":     _flt(r.get("trend_strength")),
            "days_to_earnings":   _flt(r.get("days_to_earnings")),
            # score components
            "earn_pts":           _flt(r.get("earn_pts")),
            "ivr_pts":            _flt(r.get("ivr_pts")),
            "ivhv_pts":           _flt(r.get("ivhv_pts")),
            "trend_pts":          _flt(r.get("trend_pts")),
            "composite_score":    _flt(r.get("composite_score")),
            # normalized cost (% of stock price — the real measure)
            "debit_pct_spot":     debit_pct_spot,
            "put_debit_pct_spot": put_debit_pct_spot,
            "entry_mid_debit":    entry_debit,   # kept for reference, not used as feature
            "comm_to_debit_pct":  comm_to_debit_pct,
            "comm_to_debit_pct_put": comm_to_debit_pct_put,
            "days_held":          dh,
            # targets — straddle
            "pnl_pct_mid":        _flt(r.get("pnl_pct_mid")),
            "pnl_mid":            _flt(r.get("pnl_mid")),
            # targets — put calendar
            "put_pnl_pct_mid":    _flt(r.get("put_pnl_pct_mid")),
            "put_pnl_mid":        _flt(r.get("put_pnl_mid")),
            "put_entry_mid":      put_debit,
        }
        records.append(rec)

    print(f"Loaded {len(records)} valid trades (after earnings/DH filter)")
    return records


# ── Feature Engineering ───────────────────────────────────────────────────────

FEATURE_COLS = [
    "ff_pct", "ff_quality_pts",
    "front_dte", "back_dte",
    "front_iv", "back_iv", "iv_spread",
    "iv_hv_ratio", "iv_rank_20d", "trend_strength",
    "days_to_earnings",
    "earn_pts", "ivr_pts", "ivhv_pts", "trend_pts",
    "debit_pct_spot",        # calendar cost as % of stock price
    "comm_to_debit_pct",     # IB commission as % of debit ($0.052/share ÷ debit): penalises cheap debits
    # log_price removed: near-zero coef in the retrained model; comm_to_debit_pct is the direct driver
    # days_held removed — leaky feature (unknown at entry time)
]

# Put calendar uses its own debit normalisation
FEATURE_COLS_PUT = [
    "ff_pct", "ff_quality_pts",
    "front_dte", "back_dte",
    "front_iv", "back_iv", "iv_spread",
    "iv_hv_ratio", "iv_rank_20d", "trend_strength",
    "days_to_earnings",
    "earn_pts", "ivr_pts", "ivhv_pts", "trend_pts",
    "put_debit_pct_spot",
    "comm_to_debit_pct_put",
]

FEATURE_MEDIANS = {}


def build_matrix(records, fit_medians=True, cols=None):
    """Convert records to numpy feature matrix with median imputation."""
    global FEATURE_MEDIANS
    if cols is None:
        cols = FEATURE_COLS

    # Special handling: days_to_earnings null → 365 (no upcoming earnings)
    for r in records:
        if r.get("days_to_earnings") is None:
            r["days_to_earnings"] = 365.0

    if fit_medians:
        for col in cols:
            vals = [r[col] for r in records if r.get(col) is not None]
            FEATURE_MEDIANS[col] = float(np.median(vals)) if vals else 0.0

    X = np.array([
        [r.get(c) if r.get(c) is not None else FEATURE_MEDIANS.get(c, 0.0) for c in cols]
        for r in records
    ], dtype=float)

    return X


def prepare_targets(records, structure="straddle"):
    """Return (y_bin, y_pct, mask) where mask selects rows with valid pricing."""
    if structure == "straddle":
        pnl_key = "pnl_pct_mid"
        debit_key = "entry_mid_debit"
    else:
        pnl_key = "put_pnl_pct_mid"
        debit_key = "put_entry_mid"

    # IB commission per roundtrip: $0.65 × 8 contracts (4 legs × entry + exit) = $5.20
    # pnl_pct_mid is expressed as % of entry debit (e.g. 10 = 10% gain on debit paid).
    # Commission as % of debit = ($5.20/100) / entry_debit * 100 = 5.20 / entry_debit
    # (5.20/100 converts 8-contract to per-share; dividing by debit normalises to same scale as pnl)
    IB_COMM_DOLLARS_PER_SHARE = 5.20 / 100.0  # $0.052/share roundtrip

    mask = []
    y_bin, y_pct = [], []
    for r in records:
        pnl = r.get(pnl_key)
        dbt = r.get(debit_key)
        if pnl is None or dbt is None or dbt < MIN_DEBIT:
            mask.append(False)
            y_bin.append(0)
            y_pct.append(0.0)
        else:
            mask.append(True)
            # Convert commission to same % scale as pnl: ($/share) / (debit $/share) * 100
            comm_pct = IB_COMM_DOLLARS_PER_SHARE / dbt * 100.0
            pnl_after_comm = pnl - comm_pct
            y_bin.append(1 if pnl_after_comm > 0 else 0)
            y_pct.append(float(np.clip(pnl, -PNL_PCT_CLIP, PNL_PCT_CLIP)))

    return np.array(y_bin), np.array(y_pct), np.array(mask, dtype=bool)


# ── Time-series CV ────────────────────────────────────────────────────────────

def ts_split(records, n_folds=4):
    """
    Split into n_folds chronological folds.
    Each fold: train on all prior data, test on this fold.
    Uses OOS period only (sufficient data).
    """
    oos = [r for r in records if r["period"] == "OOS"]
    dates = sorted(set(r["scan_date"] for r in oos))
    fold_size = len(dates) // n_folds
    folds = []
    for i in range(1, n_folds):
        cutoff = dates[i * fold_size]
        train = [r for r in oos if r["scan_date"] <  cutoff]
        test  = [r for r in oos if r["scan_date"] >= cutoff]
        if len(train) >= 50 and len(test) >= 50:
            folds.append((train, test, cutoff))
    return folds


# ── Model Training & Evaluation ───────────────────────────────────────────────

def run_models(records, structure="straddle"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
    from sklearn.calibration import calibration_curve
    import xgboost as xgb

    print(f"\n{'='*60}")
    print(f"  {structure.upper()} CALENDAR")
    print(f"{'='*60}")

    # Choose feature set by structure
    feat_cols = FEATURE_COLS if structure == "straddle" else FEATURE_COLS_PUT

    # Build full dataset
    X_all = build_matrix(records, fit_medians=True, cols=feat_cols)
    y_bin_all, y_pct_all, mask_all = prepare_targets(records, structure)
    X_p  = X_all[mask_all]
    y_b  = y_bin_all[mask_all]
    y_p  = y_pct_all[mask_all]
    recs = [r for r, m in zip(records, mask_all) if m]

    print(f"  Priced trades: {len(y_b)} | Win rate: {y_b.mean()*100:.1f}%")
    print(f"  Avg P&L: {y_p.mean():+.1f}%  Median: {float(np.median(y_p)):+.1f}%")

    # IS/OOS split
    is_mask  = np.array([r["period"] == "IS"  for r in recs])
    oos_mask = np.array([r["period"] == "OOS" for r in recs])

    print(f"  IS: {is_mask.sum()} | OOS: {oos_mask.sum()}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_p[oos_mask])  # fit on OOS (IS too small)
    X_full  = scaler.transform(X_p)
    X_is    = scaler.transform(X_p[is_mask]) if is_mask.sum() > 0 else None

    # ── Logistic Regression ──
    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_b[oos_mask])
    lr_probs = lr.predict_proba(X_full)[:, 1]
    lr_acc   = accuracy_score(y_b[oos_mask], lr.predict(X_train))
    lr_auc   = roc_auc_score(y_b[oos_mask], lr.predict_proba(X_train)[:, 1])
    print(f"\n  LogReg (OOS): acc={lr_acc:.3f}  AUC={lr_auc:.3f}")
    # Print coefficients for embedding in ff_trade_scanner.py
    _med = {c: float(np.median(X_p[oos_mask, i])) for i, c in enumerate(feat_cols)}
    _std = {c: float(np.std(X_p[oos_mask, i]) or 1.0) for i, c in enumerate(feat_cols)}
    _coef = dict(zip(feat_cols, lr.coef_[0]))
    print(f"\n  --- LR SCANNER COEFFICIENTS ({structure.upper()}) ---")
    print(f"  _ML_MED  = {_med}")
    print(f"  _ML_STD  = {_std}")
    print(f"  _ML_COEF = {_coef}")
    print(f"  intercept = {lr.intercept_[0]:.4f}")

    # ── Random Forest ──
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_b[oos_mask])
    rf_probs = rf.predict_proba(X_full)[:, 1]
    rf_acc   = accuracy_score(y_b[oos_mask], rf.predict(X_train))
    rf_auc   = roc_auc_score(y_b[oos_mask], rf.predict_proba(X_train)[:, 1])
    rf_imp   = dict(zip(feat_cols, rf.feature_importances_))
    print(f"  RF        (OOS): acc={rf_acc:.3f}  AUC={rf_auc:.3f}")

    # ── XGBoost ──
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(1 - y_b[oos_mask].mean()) / y_b[oos_mask].mean(),
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_b[oos_mask], verbose=False)
    xgb_probs = xgb_model.predict_proba(X_full)[:, 1]
    xgb_acc   = accuracy_score(y_b[oos_mask], xgb_model.predict(X_train))
    xgb_auc   = roc_auc_score(y_b[oos_mask], xgb_model.predict_proba(X_train)[:, 1])
    print(f"  XGBoost   (OOS): acc={xgb_acc:.3f}  AUC={xgb_auc:.3f}")

    # ── SHAP analysis on XGBoost ──
    try:
        import shap
        explainer  = shap.TreeExplainer(xgb_model)
        shap_vals  = explainer.shap_values(X_train)
        shap_mean  = np.abs(shap_vals).mean(axis=0)
        shap_imp   = dict(zip(feat_cols, shap_mean))
        top_shap   = sorted(shap_imp.items(), key=lambda x: -x[1])[:10]
        print(f"\n  SHAP top features (XGBoost):")
        for feat, imp in top_shap:
            bar = "█" * int(imp / top_shap[0][1] * 20)
            print(f"    {feat:<22} {imp:.4f}  {bar}")
    except Exception as e:
        print(f"  SHAP failed: {e}")
        shap_imp = rf_imp
        top_shap = sorted(shap_imp.items(), key=lambda x: -x[1])[:10]

    # ── Time-series cross-validation ──
    print(f"\n  Time-series CV (chronological folds):")
    folds = ts_split(records)
    cv_aucs, cv_wrs, cv_gains = [], [], []
    for train_r, test_r, cutoff in folds:
        X_cv_train = build_matrix(train_r, fit_medians=True,  cols=feat_cols)
        y_tr_b, _,  tm = prepare_targets(train_r, structure)
        X_cv_test  = build_matrix(test_r,  fit_medians=False, cols=feat_cols)
        y_bt, y_pt, te = prepare_targets(test_r, structure)

        if tm.sum() < 30 or te.sum() < 30:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_cv_train[tm])
        Xte = sc.transform(X_cv_test[te])

        m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                               subsample=0.8, random_state=42, n_jobs=-1)
        m.fit(Xtr, y_tr_b[tm], verbose=False)
        probs = m.predict_proba(Xte)[:, 1]

        try:
            auc = roc_auc_score(y_bt[te], probs)
        except Exception:
            auc = float("nan")

        # Top 30% predicted probability → check actual win rate
        thresh = np.percentile(probs, 70)
        top_mask = probs >= thresh
        top_wr = y_bt[te][top_mask].mean() if top_mask.sum() > 0 else float("nan")
        top_pnl = y_pt[te][top_mask].mean() if top_mask.sum() > 0 else float("nan")

        cv_aucs.append(auc)
        cv_wrs.append(top_wr)
        cv_gains.append(top_pnl)
        print(f"    Before {cutoff}: AUC={auc:.3f}  Top-30% win={top_wr*100:.1f}%  avg P&L={top_pnl:+.1f}%")

    mean_auc = float(np.nanmean(cv_aucs)) if cv_aucs else float("nan")
    mean_wr  = float(np.nanmean(cv_wrs))  if cv_wrs  else float("nan")
    mean_gain= float(np.nanmean(cv_gains))if cv_gains else float("nan")
    print(f"  Mean CV: AUC={mean_auc:.3f}  Top-30% win={mean_wr*100:.1f}%  avg P&L={mean_gain:+.1f}%")

    # ── Feature importance table ──
    rf_sorted  = sorted(rf_imp.items(),  key=lambda x: -x[1])
    shap_sorted = sorted(shap_imp.items(), key=lambda x: -x[1])

    # ── P&L by predicted probability decile (XGBoost, OOS) ──
    oos_probs = xgb_probs[oos_mask]
    oos_y_b   = y_b[oos_mask]
    oos_y_p   = y_p[oos_mask]
    decile_data = []
    for q in range(10):
        lo, hi = np.percentile(oos_probs, q*10), np.percentile(oos_probs, (q+1)*10)
        sel = (oos_probs >= lo) & (oos_probs < hi)
        if sel.sum() == 0:
            continue
        decile_data.append({
            "decile": f"D{q+1}",
            "n":      int(sel.sum()),
            "win_pct": float(oos_y_b[sel].mean() * 100),
            "avg_pnl": float(oos_y_p[sel].mean()),
            "pred_prob": float(oos_probs[sel].mean()),
        })

    return {
        "structure":    structure,
        "n":            len(y_b),
        "win_rate":     float(y_b.mean() * 100),
        "avg_pnl":      float(y_p.mean()),
        "lr_auc":       lr_auc,
        "rf_auc":       rf_auc,
        "xgb_auc":      xgb_auc,
        "mean_cv_auc":  mean_auc,
        "mean_cv_wr_top30": mean_wr,
        "mean_cv_pnl_top30": mean_gain,
        "rf_importance": rf_sorted,
        "shap_importance": shap_sorted,
        "decile_data":  decile_data,
        "xgb_probs_oos": oos_probs.tolist(),
        "y_win_oos":    oos_y_b.tolist(),
        "y_pnl_oos":    oos_y_p.tolist(),
    }


# ── Feature correlation / unconditional analysis ─────────────────────────────

def univariate_analysis(records, structure="straddle"):
    """Spearman-rank each feature against win outcome."""
    from scipy.stats import spearmanr

    pnl_key = "pnl_pct_mid" if structure == "straddle" else "put_pnl_pct_mid"
    priced = [r for r in records if r.get(pnl_key) is not None
              and (r.get("entry_mid_debit") or 0) >= MIN_DEBIT]

    results = []
    feat_cols = FEATURE_COLS if structure == "straddle" else FEATURE_COLS_PUT
    for col in feat_cols:
        vals = [(r[col], 1 if r[pnl_key] > 0 else 0)
                for r in priced if r[col] is not None]
        if len(vals) < 50:
            results.append({"feature": col, "corr": None, "pval": None, "n": len(vals)})
            continue
        xs, ys = zip(*vals)
        corr, pval = spearmanr(xs, ys)
        results.append({"feature": col, "corr": round(corr, 4), "pval": round(pval, 4), "n": len(vals)})

    results.sort(key=lambda x: abs(x["corr"] or 0), reverse=True)
    return results


# ── HTML Report ───────────────────────────────────────────────────────────────

def build_html(straddle_res, put_res, straddle_uni, put_uni):
    def js(x):
        return json.dumps(x)

    def imp_rows(imp_list, limit=12):
        rows = ""
        max_val = imp_list[0][1] if imp_list else 1
        for feat, val in imp_list[:limit]:
            pct = val / max_val * 100
            rows += (f"<tr><td style='text-align:left'>{feat}</td>"
                     f"<td><div style='width:{pct:.0f}%;background:#00d4ff;height:12px;border-radius:3px'></div></td>"
                     f"<td>{val:.4f}</td></tr>\n")
        return rows

    def uni_rows(uni_list):
        rows = ""
        for r in uni_list[:12]:
            corr = r["corr"]
            if corr is None:
                continue
            color = "#00d4ff" if corr > 0 else "#ff6060"
            sig = "✓" if (r["pval"] or 1) < 0.05 else ""
            rows += (f"<tr><td style='text-align:left'>{r['feature']}</td>"
                     f"<td style='color:{color}'>{corr:+.4f}</td>"
                     f"<td>{r['pval']:.3f}</td>"
                     f"<td>{r['n']}</td>"
                     f"<td>{sig}</td></tr>\n")
        return rows

    def decile_rows(dec_data):
        rows = ""
        for d in dec_data:
            color = "#00ff88" if d["avg_pnl"] > 0 else "#ff6060"
            rows += (f"<tr><td>{d['decile']}</td><td>{d['n']}</td>"
                     f"<td>{d['pred_prob']*100:.0f}%</td>"
                     f"<td>{d['win_pct']:.1f}%</td>"
                     f"<td style='color:{color}'>{d['avg_pnl']:+.1f}%</td></tr>\n")
        return rows

    str_dec = straddle_res["decile_data"]
    put_dec = put_res["decile_data"]

    str_dec_labels = [d["decile"] for d in str_dec]
    str_dec_wr     = [d["win_pct"] for d in str_dec]
    str_dec_pnl    = [d["avg_pnl"] for d in str_dec]
    put_dec_labels = [d["decile"] for d in put_dec]
    put_dec_wr     = [d["win_pct"] for d in put_dec]
    put_dec_pnl    = [d["avg_pnl"] for d in put_dec]

    shap_str_labels = [x[0] for x in straddle_res["shap_importance"][:10]]
    shap_str_vals   = [x[1] for x in straddle_res["shap_importance"][:10]]
    shap_put_labels = [x[0] for x in put_res["shap_importance"][:10]]
    shap_put_vals   = [x[1] for x in put_res["shap_importance"][:10]]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>CSFF ML Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body{{font-family:monospace;background:#1a1a2e;color:#e0e0e0;margin:24px;line-height:1.5}}
  h1{{color:#00d4ff}} h2{{color:#aaa;border-bottom:1px solid #333;padding-bottom:4px}}
  .meta{{color:#888;font-size:.9em;margin-bottom:16px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:24px}}
  .grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:24px}}
  .box{{background:#16213e;border-radius:8px;padding:16px}}
  .stat{{font-size:1.6em;color:#00d4ff;font-weight:bold}}
  .sublabel{{color:#888;font-size:.8em}}
  canvas{{max-height:240px}}
  table{{border-collapse:collapse;width:100%;font-size:.83em;margin-top:8px}}
  th,td{{padding:4px 10px;text-align:right;border:1px solid #333}}
  th{{background:#2a3a5e;color:#aaa;text-align:center}}
  tr:nth-child(even){{background:#1c2a45}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.8em}}
  .green{{background:#1a3a2a;color:#00ff88}} .red{{background:#3a1a1a;color:#ff6060}}
  .warn{{color:#ffaa00;font-size:.9em}}
</style></head><body>

<h1>CSFF — ML Analysis</h1>
<div class="meta">Generated {date.today()} &nbsp;|&nbsp; OOS 2024-2025 &nbsp;|&nbsp; Earnings-risk filtered</div>

<div class="grid3">
  <div class="box">
    <div class="sublabel">Straddle Cal — Priced Trades</div>
    <div class="stat">{straddle_res['n']:,}</div>
    <div class="sublabel">Win {straddle_res['win_rate']:.1f}% &nbsp; Avg P&amp;L {straddle_res['avg_pnl']:+.1f}%</div>
  </div>
  <div class="box">
    <div class="sublabel">Put Cal — Priced Trades</div>
    <div class="stat">{put_res['n']:,}</div>
    <div class="sublabel">Win {put_res['win_rate']:.1f}% &nbsp; Avg P&amp;L {put_res['avg_pnl']:+.1f}%</div>
  </div>
  <div class="box">
    <div class="sublabel">XGBoost CV AUC (time-series)</div>
    <div class="stat">{straddle_res['mean_cv_auc']:.3f}</div>
    <div class="sublabel">Straddle &nbsp;|&nbsp; Put: {put_res['mean_cv_auc']:.3f}</div>
  </div>
</div>

<h2>Model Performance Summary</h2>
<div class="grid">
<div class="box">
  <h2 style="color:#aaa;font-size:1em">Straddle Calendar</h2>
  <table>
  <tr><th>Model</th><th>AUC (in-sample train)</th></tr>
  <tr><td>Logistic Regression</td><td>{straddle_res['lr_auc']:.3f}</td></tr>
  <tr><td>Random Forest</td><td>{straddle_res['rf_auc']:.3f}</td></tr>
  <tr><td>XGBoost</td><td>{straddle_res['xgb_auc']:.3f}</td></tr>
  <tr><td><b>CV Mean AUC (time-series)</b></td><td><b>{straddle_res['mean_cv_auc']:.3f}</b></td></tr>
  <tr><td>CV Top-30% Win Rate</td><td>{straddle_res['mean_cv_wr_top30']*100:.1f}%</td></tr>
  <tr><td>CV Top-30% Avg P&amp;L</td><td>{straddle_res['mean_cv_pnl_top30']:+.1f}%</td></tr>
  </table>
</div>
<div class="box">
  <h2 style="color:#aaa;font-size:1em">Put Calendar</h2>
  <table>
  <tr><th>Model</th><th>AUC (in-sample train)</th></tr>
  <tr><td>Logistic Regression</td><td>{put_res['lr_auc']:.3f}</td></tr>
  <tr><td>Random Forest</td><td>{put_res['rf_auc']:.3f}</td></tr>
  <tr><td>XGBoost</td><td>{put_res['xgb_auc']:.3f}</td></tr>
  <tr><td><b>CV Mean AUC (time-series)</b></td><td><b>{put_res['mean_cv_auc']:.3f}</b></td></tr>
  <tr><td>CV Top-30% Win Rate</td><td>{put_res['mean_cv_wr_top30']*100:.1f}%</td></tr>
  <tr><td>CV Top-30% Avg P&amp;L</td><td>{put_res['mean_cv_pnl_top30']:+.1f}%</td></tr>
  </table>
</div>
</div>

<div class="grid">
  <div class="box"><h2>Straddle — Win Rate by Predicted Decile (OOS)</h2><canvas id="c_str_dec"></canvas></div>
  <div class="box"><h2>Put Cal — Win Rate by Predicted Decile (OOS)</h2><canvas id="c_put_dec"></canvas></div>
</div>

<div class="grid">
  <div class="box"><h2>SHAP Feature Importance — Straddle</h2>
    <table><tr><th style="text-align:left">Feature</th><th>Bar</th><th>SHAP</th></tr>
    {imp_rows(straddle_res['shap_importance'])}</table>
  </div>
  <div class="box"><h2>SHAP Feature Importance — Put Calendar</h2>
    <table><tr><th style="text-align:left">Feature</th><th>Bar</th><th>SHAP</th></tr>
    {imp_rows(put_res['shap_importance'])}</table>
  </div>
</div>

<div class="grid">
  <div class="box"><h2>Univariate Spearman Correlation (win vs feature) — Straddle</h2>
    <table><tr><th>Feature</th><th>ρ</th><th>p-val</th><th>n</th><th>sig?</th></tr>
    {uni_rows(straddle_uni)}</table>
  </div>
  <div class="box"><h2>Univariate Spearman Correlation (win vs feature) — Put Cal</h2>
    <table><tr><th>Feature</th><th>ρ</th><th>p-val</th><th>n</th><th>sig?</th></tr>
    {uni_rows(put_uni)}</table>
  </div>
</div>

<div class="grid">
  <div class="box"><h2>Straddle — P&amp;L by Decile (OOS)</h2>
    <table><tr><th>Decile</th><th>N</th><th>Pred%</th><th>Win%</th><th>Avg P&amp;L</th></tr>
    {decile_rows(str_dec)}</table>
  </div>
  <div class="box"><h2>Put Cal — P&amp;L by Decile (OOS)</h2>
    <table><tr><th>Decile</th><th>N</th><th>Pred%</th><th>Win%</th><th>Avg P&amp;L</th></tr>
    {decile_rows(put_dec)}</table>
  </div>
</div>

<script>
const chartOpts = (yLabel) => ({{
  plugins:{{legend:{{labels:{{color:'#aaa'}}}}}},
  scales:{{
    y:{{title:{{display:true,text:yLabel,color:'#888'}},ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}},
    x:{{ticks:{{color:'#aaa'}},grid:{{color:'#333'}}}}
  }}
}});

new Chart('c_str_dec',{{type:'bar',data:{{
  labels:{js(str_dec_labels)},
  datasets:[
    {{label:'Win %',data:{js(str_dec_wr)},backgroundColor:'rgba(0,212,255,.7)',yAxisID:'y'}},
    {{label:'Avg P&L %',data:{js(str_dec_pnl)},backgroundColor:'rgba(255,180,0,.7)',type:'line',yAxisID:'y2'}}
  ]
}},options:{{...chartOpts('Win %'),scales:{{
  y:{{title:{{display:true,text:'Win %',color:'#888'}},ticks:{{color:'#aaa'}},grid:{{color:'#333'}},min:0,max:100}},
  y2:{{position:'right',title:{{display:true,text:'Avg P&L %',color:'#888'}},ticks:{{color:'#aaa'}},grid:{{drawOnChartArea:false}}}}
}}}}}});

new Chart('c_put_dec',{{type:'bar',data:{{
  labels:{js(put_dec_labels)},
  datasets:[
    {{label:'Win %',data:{js(put_dec_wr)},backgroundColor:'rgba(160,100,255,.7)',yAxisID:'y'}},
    {{label:'Avg P&L %',data:{js(put_dec_pnl)},backgroundColor:'rgba(255,180,0,.7)',type:'line',yAxisID:'y2'}}
  ]
}},options:{{...chartOpts('Win %'),scales:{{
  y:{{title:{{display:true,text:'Win %',color:'#888'}},ticks:{{color:'#aaa'}},grid:{{color:'#333'}},min:0,max:100}},
  y2:{{position:'right',title:{{display:true,text:'Avg P&L %',color:'#888'}},ticks:{{color:'#aaa'}},grid:{{drawOnChartArea:false}}}}
}}}}}});
</script>
</body></html>"""
    return html


# ── Text Summary ──────────────────────────────────────────────────────────────

def write_text_summary(straddle_res, put_res, straddle_uni, put_uni):
    lines = [
        "=" * 70,
        "CSFF ML ANALYSIS — FINDINGS",
        f"Generated {date.today()}",
        "=" * 70,
        "",
        "DATA",
        f"  Straddle: {straddle_res['n']} priced trades | Win {straddle_res['win_rate']:.1f}% | Avg P&L {straddle_res['avg_pnl']:+.1f}%",
        f"  Put Cal:  {put_res['n']} priced trades | Win {put_res['win_rate']:.1f}% | Avg P&L {put_res['avg_pnl']:+.1f}%",
        "",
        "ML PREDICTABILITY (time-series CV AUC)",
        f"  Straddle: {straddle_res['mean_cv_auc']:.3f}  | Put Cal: {put_res['mean_cv_auc']:.3f}",
        f"  AUC ~0.50 = no edge | ~0.55+ = some signal | ~0.60+ = meaningful",
        "",
        "TOP-30% PREDICTED BUCKET (OOS CV)",
        f"  Straddle: Win {straddle_res['mean_cv_wr_top30']*100:.1f}% | Avg P&L {straddle_res['mean_cv_pnl_top30']:+.1f}%",
        f"  Put Cal:  Win {put_res['mean_cv_wr_top30']*100:.1f}% | Avg P&L {put_res['mean_cv_pnl_top30']:+.1f}%",
        "",
        "TOP SHAP FEATURES — STRADDLE (by importance)",
    ]
    for feat, val in straddle_res["shap_importance"][:8]:
        lines.append(f"  {feat:<22} {val:.4f}")
    lines += [
        "",
        "TOP SHAP FEATURES — PUT CALENDAR (by importance)",
    ]
    for feat, val in put_res["shap_importance"][:8]:
        lines.append(f"  {feat:<22} {val:.4f}")
    lines += [
        "",
        "UNIVARIATE CORRELATIONS (Spearman ρ with win, significant p<0.05)",
    ]
    for r in straddle_uni[:8]:
        if r["corr"] and abs(r["corr"]) > 0.02:
            sig = "*" if (r["pval"] or 1) < 0.05 else ""
            lines.append(f"  {r['feature']:<22} ρ={r['corr']:+.4f} p={r['pval']:.3f} {sig}")
    lines += ["", "=" * 70]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("scipy not found — install with: pip install scipy")
        return 1

    records = load_data()
    if not records:
        print("No data loaded.")
        return 1

    # Run ML for both structures
    straddle_res = run_models(records, structure="straddle")
    put_res      = run_models(records, structure="put")

    # Univariate
    print("\nComputing univariate correlations ...")
    straddle_uni = univariate_analysis(records, "straddle")
    put_uni      = univariate_analysis(records, "put")

    # Output
    html = build_html(straddle_res, put_res, straddle_uni, put_uni)
    ML_HTML.write_text(html)
    print(f"\nHTML report: {ML_HTML}")

    summary = write_text_summary(straddle_res, put_res, straddle_uni, put_uni)
    ML_TEXT.write_text(summary)
    print(f"Text summary: {ML_TEXT}")
    print()
    print(summary)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
