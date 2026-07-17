#!/usr/bin/env python3
"""
ff_ml_eval.py — rigorous, honest evaluation harness for CSFF calendar ML.

Replaces the misleading in-sample numbers in ff_ml_analysis.py (which trained and
scored on the same rows). Everything here is out-of-fold.

Methodology
-----------
1. TARGET = return over DEBIT (= return on max loss for a long calendar), post-commission,
   winsorized to [WINSOR_LO, WINSOR_HI] so the data-corrupt < -100% tails and the
   cheap-debit +2000% spikes don't dominate. (See target diagnostics: cheap-debit trades
   carry std 213% and 13% |return|>200%; cap them.)

2. EV IS TAIL-DRIVEN. Median post-comm trade loses; mean is positive only because of the
   ~17% "big win" tail. So the deployment metric is TOP-QUINTILE REALIZED EV (winsorized
   mean net return of the top 20% by model score), NOT win rate.

3. EVALUATION = purged, embargoed walk-forward. A training trade entered at t with holding
   h occupies [t, t+h]; it is PURGED from a fold's train set if t+h >= test_start (its
   outcome leaks into the test window). Multiple expanding folds; pooled out-of-fold (OOF)
   predictions feed bootstrap CIs and calibration.

4. TARGET COMPARISON: we empirically compare candidate targets (binary-profit,
   binary-big-win, winsorized-regression) by held-out top-quintile EV and pick the winner.

5. MIN-DEBIT SWEEP: quantifies the single biggest risk lever (a hard min-debit gate),
   which is independent of and stronger than the ML score.

6. FEATURES are trimmed: the deterministic *_pts score components are dropped (they are
   1:1 transforms of the raw features they sit beside, inflating variance with no new info).

7. Deployment model = regularized logistic regression (embeddable as coefficients in
   ff_trade_scanner.py). XGBoost is reported as a ceiling check only — its honest OOF AUC
   is ~equal to LR, so its extra capacity buys nothing out-of-sample.

Output: backtest/results/ml_eval_findings.txt + console, and LR coefficients for the scanner.
"""

import math
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from ff_ml_analysis import load_data  # reuse the loader (same filters)

RESULTS_DIR = Path(__file__).parent / "backtest" / "results"
OUT_TEXT    = RESULTS_DIR / "ml_eval_findings.txt"

# ── Tunables ──────────────────────────────────────────────────────────────────
WINSOR_LO, WINSOR_HI = -100.0, 300.0   # cap net return-over-debit (%)
BIG_WIN_THRESH       = 50.0            # "big win" = net return > +50% (the EV tail)
MIN_DEBIT            = 0.10            # base filter (matches load_data)
N_FOLDS              = 5              # expanding walk-forward test segments = N_FOLDS-1
EMBARGO_DAYS         = 5              # extra calendar-day gap before each test start
TOP_Q                = 0.20           # top-quintile for EV metric
N_BOOT               = 1000           # bootstrap resamples
IB_COMM_PER_SHARE    = 5.20 / 100.0   # $0.052/share roundtrip

# Trimmed feature set — raw signal only, no deterministic point-transforms
TRIM_FEATURES = [
    "ff_pct", "front_dte", "back_dte",
    "front_iv", "back_iv", "iv_spread",
    "iv_hv_ratio", "iv_rank_20d", "trend_strength",
    "days_to_earnings",
    "debit_pct_spot", "comm_to_debit_pct",
]


# ── Target / data prep ──────────────────────────────────────────────────────────

def prep(records, structure="straddle"):
    """Return dict of aligned arrays for priced trades of the given structure."""
    if structure == "straddle":
        pnl_key, debit_key = "pnl_pct_mid", "entry_mid_debit"
    else:
        pnl_key, debit_key = "put_pnl_pct_mid", "put_entry_mid"

    feats = list(TRIM_FEATURES)
    if structure == "put":
        # swap straddle-specific cost features for put equivalents
        feats = [("put_debit_pct_spot" if f == "debit_pct_spot" else
                  "comm_to_debit_pct_put" if f == "comm_to_debit_pct" else f) for f in feats]

    pnl, debit, net, scan, held, tick = [], [], [], [], [], []
    feat_rows = []
    for r in records:
        p = r.get(pnl_key)
        d = r.get(debit_key)
        if p is None or d is None or d < MIN_DEBIT:
            continue
        comm_pct = IB_COMM_PER_SHARE / d * 100.0
        net_ret  = float(np.clip(p - comm_pct, WINSOR_LO, WINSOR_HI))
        pnl.append(p); debit.append(d); net.append(net_ret)
        scan.append(r["scan_date"]); held.append(r.get("days_held") or 1.0)
        tick.append(r.get("ticker", ""))
        feat_rows.append([r.get(f) for f in feats])

    X = np.array(feat_rows, dtype=object)
    # median-impute per column (fit on full set is fine; medians recomputed per fold below)
    return {
        "feats": feats,
        "X_raw": X,
        "net":   np.array(net),
        "debit": np.array(debit),
        "scan":  np.array(scan),
        "held":  np.array(held, dtype=float),
        "tick":  np.array(tick),
        "n":     len(net),
    }


def impute(X_raw, medians):
    X = np.empty(X_raw.shape, dtype=float)
    for j in range(X_raw.shape[1]):
        col = X_raw[:, j]
        for i, v in enumerate(col):
            X[i, j] = float(v) if v is not None else medians[j]
    return X


def col_medians(X_raw):
    med = []
    for j in range(X_raw.shape[1]):
        vals = [float(v) for v in X_raw[:, j] if v is not None]
        med.append(float(np.median(vals)) if vals else 0.0)
    return med


# ── Purged walk-forward folds ────────────────────────────────────────────────────

def purged_folds(scan, held, n_folds=N_FOLDS, embargo_days=EMBARGO_DAYS):
    """Yield (train_idx, test_idx) for expanding walk-forward with leakage purge.

    A training trade i with exit ~ scan_i + held_i is dropped if its outcome window
    reaches into (test_start - embargo).
    """
    uniq = sorted(set(scan))
    if len(uniq) < n_folds + 1:
        return
    # contiguous date chunks (clamp last edge to final date + 1 day sentinel)
    sentinel = uniq[-1] + timedelta(days=1)
    chunk_edges = []
    for k in range(n_folds + 1):
        idx = int(round(k * len(uniq) / n_folds))
        chunk_edges.append(sentinel if idx >= len(uniq) else uniq[idx])
    for k in range(1, n_folds):
        test_start = chunk_edges[k]
        test_end   = chunk_edges[k + 1]
        purge_before = test_start - timedelta(days=embargo_days)
        test_idx, train_idx = [], []
        for i in range(len(scan)):
            sd = scan[i]
            if test_start <= sd < test_end:
                test_idx.append(i)
            elif sd < test_start:
                exit_dt = sd + timedelta(days=float(held[i]))
                if exit_dt < purge_before:          # outcome fully resolved before test
                    train_idx.append(i)
        if len(train_idx) >= 50 and len(test_idx) >= 30:
            yield np.array(train_idx), np.array(test_idx), test_start


# ── Metrics ──────────────────────────────────────────────────────────────────────

def topq_ev(scores, net, q=TOP_Q):
    """Winsorized mean net return of the top-q fraction by score."""
    if len(scores) == 0:
        return float("nan")
    thr = np.quantile(scores, 1 - q)
    sel = scores >= thr
    return float(net[sel].mean()) if sel.sum() else float("nan")


def auc(y_bin, scores):
    from sklearn.metrics import roc_auc_score
    if len(set(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, scores))


# ── Out-of-fold evaluation for one target ────────────────────────────────────────

def oof_eval(data, target="bin_profit"):
    """Run purged WF with LR for a target; return pooled OOF scores + realized net."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    net = data["net"]
    oof_score, oof_net, oof_ybin = [], [], []
    for tr, te, _ in purged_folds(data["scan"], data["held"]):
        med = col_medians(data["X_raw"][tr])
        Xtr = impute(data["X_raw"][tr], med)
        Xte = impute(data["X_raw"][te], med)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        ntr, nte = net[tr], net[te]

        if target == "bin_profit":
            ytr = (ntr > 0).astype(int)
            if len(set(ytr)) < 2: continue
            m = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
            s = m.predict_proba(Xte)[:, 1]
        elif target == "bin_bigwin":
            ytr = (ntr > BIG_WIN_THRESH).astype(int)
            if len(set(ytr)) < 2: continue
            m = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
            s = m.predict_proba(Xte)[:, 1]
        elif target == "reg_winsor":
            m = Ridge(alpha=10.0).fit(Xtr, ntr)
            s = m.predict(Xte)
        else:
            raise ValueError(target)

        oof_score.extend(s.tolist())
        oof_net.extend(nte.tolist())
        oof_ybin.extend((nte > 0).astype(int).tolist())

    return (np.array(oof_score), np.array(oof_net), np.array(oof_ybin))


def bootstrap_metric(scores, net, ybin, fn, n=N_BOOT):
    """Bootstrap CI for a metric fn(scores, net, ybin)."""
    rng = np.random.default_rng(42)
    N = len(scores)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            vals.append(fn(scores[idx], net[idx], ybin[idx]))
        except Exception:
            pass
    vals = np.array([v for v in vals if not np.isnan(v)])
    return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95))) if len(vals) else (float("nan"),)*2


# ── Min-debit sweep ────────────────────────────────────────────────────────────

def min_debit_sweep(data, grid=(0.10, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00)):
    rows = []
    for thr in grid:
        m = data["debit"] >= thr
        if m.sum() < 30: continue
        net = data["net"][m]
        rows.append({
            "thr": thr, "n": int(m.sum()), "pct_kept": m.mean() * 100,
            "win_rate": (net > 0).mean() * 100,
            "ev_mean": net.mean(), "ev_median": float(np.median(net)),
        })
    return rows


# ── XGBoost ceiling (honest OOF) ─────────────────────────────────────────────────

def xgb_oof_auc(data):
    try:
        import xgboost as xgb
    except ImportError:
        return float("nan")
    from sklearn.preprocessing import StandardScaler
    oof_s, oof_y = [], []
    for tr, te, _ in purged_folds(data["scan"], data["held"]):
        med = col_medians(data["X_raw"][tr])
        Xtr = impute(data["X_raw"][tr], med); Xte = impute(data["X_raw"][te], med)
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        ytr = (data["net"][tr] > 0).astype(int)
        if len(set(ytr)) < 2: continue
        m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=42, n_jobs=-1).fit(Xtr, ytr)
        oof_s.extend(m.predict_proba(Xte)[:, 1].tolist())
        oof_y.extend((data["net"][te] > 0).astype(int).tolist())
    return auc(np.array(oof_y), np.array(oof_s))


# ── Final LR fit for scanner embedding ───────────────────────────────────────────

def fit_final_lr(data, target="bin_profit"):
    """Fit LR on ALL priced trades (chosen target) and return med/std/coef/intercept."""
    from sklearn.linear_model import LogisticRegression
    med = col_medians(data["X_raw"])
    X = impute(data["X_raw"], med)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    if target == "bin_bigwin":
        y = (data["net"] > BIG_WIN_THRESH).astype(int)
    else:
        y = (data["net"] > 0).astype(int)
    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced").fit(Xs, y)
    feats = data["feats"]
    return {
        "med":  {f: float(mu[i]) for i, f in enumerate(feats)},
        "std":  {f: float(sd[i]) for i, f in enumerate(feats)},
        "coef": {f: float(lr.coef_[0][i]) for i, f in enumerate(feats)},
        "intercept": float(lr.intercept_[0]),
    }


# ── Driver ────────────────────────────────────────────────────────────────────

def evaluate_structure(records, structure, out):
    data = prep(records, structure)
    out.append("=" * 70)
    out.append(f"  {structure.upper()} CALENDAR  (n={data['n']} priced trades, return over debit, post-comm)")
    out.append("=" * 70)

    base_win = (data["net"] > 0).mean() * 100
    out.append(f"  Unconditional: win {base_win:.1f}% | EV mean {data['net'].mean():+.1f}% | "
               f"EV median {float(np.median(data['net'])):+.1f}%  (winsorized [{WINSOR_LO:.0f},{WINSOR_HI:.0f}])")

    # Number of purged folds
    folds = list(purged_folds(data["scan"], data["held"]))
    out.append(f"  Purged walk-forward folds: {len(folds)} (embargo {EMBARGO_DAYS}d, leakage-purged)")
    for _, te, ts in folds:
        out.append(f"     test from {ts}  (n={len(te)})")

    # Target comparison
    out.append("\n  TARGET COMPARISON (pooled out-of-fold; metric = top-quintile realized EV)")
    out.append(f"  {'target':<14}{'OOF AUC':>10}{'top20% EV':>12}{'EV 90% CI':>22}")
    best = None
    for tgt in ("bin_profit", "bin_bigwin", "reg_winsor"):
        s, nt, yb = oof_eval(data, tgt)
        if len(s) == 0:
            out.append(f"  {tgt:<14}{'n/a':>10}")
            continue
        a  = auc(yb, s) if tgt != "reg_winsor" else float("nan")
        ev = topq_ev(s, nt)
        lo, hi = bootstrap_metric(s, nt, yb, lambda ss, nn, yy: topq_ev(ss, nn))
        a_str = f"{a:.3f}" if not np.isnan(a) else "  —  "
        out.append(f"  {tgt:<14}{a_str:>10}{ev:>+11.1f}%   [{lo:+.1f}%, {hi:+.1f}%]")
        if best is None or ev > best[1]:
            best = (tgt, ev)

    # XGBoost ceiling check
    xa = xgb_oof_auc(data)
    out.append(f"\n  XGBoost ceiling (honest OOF AUC, bin_profit): {xa:.3f}"
               if not np.isnan(xa) else "\n  XGBoost ceiling: n/a")

    # AUC CI for the profit target
    s, nt, yb = oof_eval(data, "bin_profit")
    a_lo, a_hi = bootstrap_metric(s, nt, yb, lambda ss, nn, yy: auc(yy, ss))
    out.append(f"  LR OOF AUC (bin_profit): {auc(yb, s):.3f}  90% CI [{a_lo:.3f}, {a_hi:.3f}]")

    # Calibration (Brier) for profit target
    from sklearn.metrics import brier_score_loss
    if len(set(yb)) > 1:
        out.append(f"  LR OOF Brier score: {brier_score_loss(yb, np.clip(s,0,1)):.4f} "
                   f"(lower better; 0.25 = uninformative)")

    # Min-debit sweep
    out.append("\n  MIN-DEBIT SWEEP (the dominant risk lever — independent of ML)")
    out.append(f"  {'min debit':>10}{'n kept':>9}{'% kept':>9}{'win%':>8}{'EV mean':>10}{'EV med':>9}")
    for r in min_debit_sweep(data):
        out.append(f"  {('$%.2f'%r['thr']):>10}{r['n']:>9}{r['pct_kept']:>8.0f}%"
                   f"{r['win_rate']:>7.1f}%{r['ev_mean']:>+9.1f}%{r['ev_median']:>+8.1f}%")

    out.append(f"\n  >> Best target by held-out top-quintile EV: {best[0]} ({best[1]:+.1f}%)\n")
    return data, best[0]


def main():
    records = load_data()
    out = []
    out.append("=" * 70)
    out.append("CSFF ML — RIGOROUS OUT-OF-FOLD EVALUATION")
    out.append(f"Generated {date.today()}")
    out.append("All metrics are out-of-fold (purged walk-forward). No train=test numbers.")
    out.append("=" * 70 + "\n")

    s_data, s_best = evaluate_structure(records, "straddle", out)
    p_data, p_best = evaluate_structure(records, "put", out)

    # Export final LR coefficients (use the EV-winning target) for the scanner
    out.append("=" * 70)
    out.append("FINAL LR COEFFICIENTS FOR ff_trade_scanner.py")
    out.append("=" * 70)
    for label, data, tgt in (("STRADDLE", s_data, s_best), ("PUT", p_data, p_best)):
        m = fit_final_lr(data, tgt)
        out.append(f"\n# --- {label} (target={tgt}, {len(data['feats'])} trimmed features) ---")
        out.append(f"_ML_MED  = {{{', '.join(f'{k!r}: {v:.4f}' for k,v in m['med'].items())}}}")
        out.append(f"_ML_STD  = {{{', '.join(f'{k!r}: {v:.4f}' for k,v in m['std'].items())}}}")
        out.append(f"_ML_COEF = {{{', '.join(f'{k!r}: {v:.4f}' for k,v in m['coef'].items())}}}")
        out.append(f"intercept = {m['intercept']:.4f}")

    text = "\n".join(out)
    OUT_TEXT.write_text(text)
    print(text)
    print(f"\nWritten: {OUT_TEXT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
