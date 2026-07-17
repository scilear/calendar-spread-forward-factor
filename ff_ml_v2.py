#!/usr/bin/env python3
"""
ff_ml_v2.py — ML entry scoring on the full 177k-trade dataset.

Improvements over ff_ml_eval.py:
  - 177k trades (vs ~1,500 from opex-only 2024-25)
  - Richer features: debit ratio, IV proxy, seasonality, ticker type
  - Targets: bin_bigwin (hold>50%), bin_profit (hold>0%), bin_tp40 (TP40>50%)
  - Purged/embargoed walk-forward: 6 annual folds, 14-day embargo
  - Models: LogisticRegression (embeddable), XGBoost (best AUC)
  - Deployment: top-quintile EV, LR coefficients exported

Usage:
  python ff_ml_v2.py                   # full run
  python ff_ml_v2.py --ff-min 0        # all trades (default)
  python ff_ml_v2.py --ff-min 8        # FF>8% only
  python ff_ml_v2.py --model xgb       # XGB only
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent

ETF_TICKERS = {
    "SPY","DIA","MDY","XLK","XLF","XLV","XLE","XLI","XLU","XLY",
    "XLP","XLB","XLC","XLRE","XBI","XRT","XHB","XME","XOP","KRE","XSD","XAR",
}

# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    f["entry_ff"]          = df["entry_ff"].clip(-100, 200)
    f["entry_ff_sq"]       = f["entry_ff"] ** 2
    f["entry_ff_pos"]      = f["entry_ff"].clip(lower=0)

    f["entry_debit"]       = df["entry_debit"].clip(0.01, 50)
    f["log_debit"]         = np.log1p(f["entry_debit"])

    f["back_straddle"]     = df["back_straddle_entry"].clip(0.01, 200)
    f["log_back_straddle"] = np.log1p(f["back_straddle"])

    # debit as fraction of back straddle (calendar spread cost efficiency)
    f["debit_ratio"]       = (df["entry_debit"] / df["back_straddle_entry"].clip(0.01)).clip(0, 1)

    # DTE structure
    f["t_front"]           = df["t_front_entry"]
    f["t_back"]            = df["t_back_entry"]
    f["t_fwd"]             = df["t_fwd_entry"]
    f["t_fwd_ratio"]       = (df["t_fwd_entry"] / df["t_back_entry"].clip(1)).clip(0, 1)

    # max profit relative to debit (reward / risk)
    mp = df["max_profit"].fillna(0)
    f["max_profit_ratio"]  = (mp / df["entry_debit"].clip(0.01)).clip(-2, 20)

    # rough IV proxy: back straddle / sqrt(t_back/365)
    t_back_yr = (df["t_back_entry"] / 365).clip(0.001)
    f["iv_proxy"]          = (df["back_straddle_entry"] / (df["strike"].clip(1) * np.sqrt(t_back_yr))).clip(0, 2)

    # FF × fwd period interaction
    f["ff_x_tfwd"]         = f["entry_ff"] * np.sqrt(f["t_fwd"].clip(1))

    # Seasonality
    entry_dates = pd.to_datetime(df["entry_date"])
    f["month_sin"]         = np.sin(2 * np.pi * entry_dates.dt.month / 12)
    f["month_cos"]         = np.cos(2 * np.pi * entry_dates.dt.month / 12)
    f["quarter"]           = entry_dates.dt.quarter

    # Ticker type
    f["is_etf"]            = df["ticker"].isin(ETF_TICKERS).astype(int)

    return f


def build_features_put(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put-calendar features: entry_debit and back_straddle halved.
    At ATM, put ≈ straddle/2 by put-call parity, so all ratio features
    (debit_ratio, max_profit_ratio) are unchanged; only absolute and log
    features (entry_debit, log_debit, back_straddle, log_back_straddle,
    iv_proxy) differ. Return % is the same as straddle calendar at ATM.
    """
    df_put = df.copy()
    df_put["entry_debit"]         = df["entry_debit"] / 2
    df_put["back_straddle_entry"] = df["back_straddle_entry"] / 2
    return build_features(df_put)


# ── Targets ────────────────────────────────────────────────────────────────────

def build_targets(df: pd.DataFrame, sim_tp40: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame(index=df.index)
    r = df["hold_mid_ret"]
    t["bin_bigwin"]  = (r > 50).astype(int)
    t["bin_profit"]  = (r > 0).astype(int)
    t["ret_wins"]    = r.clip(-100, 300)   # winsorized return for EV calc

    # TP40 target from sweep
    tp40_map = sim_tp40.set_index("trade_id")["tp40_ret"]
    t["tp40_ret"]    = df["trade_id"].map(tp40_map).values
    t["bin_tp40"]    = (t["tp40_ret"] > 50).astype(int)

    return t


# ── Walk-forward CV ───────────────────────────────────────────────────────────

def make_folds(df: pd.DataFrame, n_folds: int = 5, embargo_days: int = 14):
    """
    Purged/embargoed walk-forward: train on years 1..k-1, test on year k.
    Embargo: exclude train rows within `embargo_days` of test start.
    Returns list of (train_idx, test_idx).
    """
    dates = pd.to_datetime(df["entry_date"])
    years = sorted(dates.dt.year.unique())
    folds = []
    for i in range(1, len(years)):
        test_yr   = years[i]
        test_mask = dates.dt.year == test_yr
        test_idx  = df.index[test_mask].tolist()

        test_start = dates[test_mask].min()
        embargo_cutoff = test_start - pd.Timedelta(days=embargo_days)
        train_mask = (dates.dt.year < test_yr) & (dates < embargo_cutoff)
        train_idx  = df.index[train_mask].tolist()

        if len(train_idx) < 500 or len(test_idx) < 200:
            continue
        folds.append((train_idx, test_idx))
    return folds


# ── Model wrappers ─────────────────────────────────────────────────────────────

def fit_lr(X_tr, y_tr, C=0.1):
    sc = StandardScaler()
    X_s = sc.fit_transform(X_tr)
    m   = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                             solver="lbfgs")
    m.fit(X_s, y_tr)
    return m, sc


def predict_lr(m, sc, X_te):
    return m.predict_proba(sc.transform(X_te))[:, 1]


def fit_xgb(X_tr, y_tr):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, None
    pos = y_tr.mean()
    scale_pos = (1 - pos) / max(pos, 1e-6)
    m = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric="logloss",
        n_jobs=-1, random_state=42,
    )
    m.fit(X_tr, y_tr, verbose=False)
    return m, None


def predict_xgb(m, _, X_te):
    return m.predict_proba(X_te)[:, 1]


# ── Evaluation ─────────────────────────────────────────────────────────────────

def top_q_ev(scores, rets, q=0.20):
    """Mean realized return (winsorized) in top-q score quintile."""
    thresh = np.quantile(scores, 1 - q)
    mask   = scores >= thresh
    return rets[mask].mean() if mask.sum() > 0 else np.nan


def run_cv(df: pd.DataFrame, features: pd.DataFrame, targets: pd.DataFrame,
           target_col: str, model: str = "both") -> dict:
    folds = make_folds(df)
    feat_cols = features.columns.tolist()

    oof_scores_lr  = np.full(len(df), np.nan)
    oof_scores_xgb = np.full(len(df), np.nan)
    oof_labels     = np.full(len(df), np.nan)

    valid_mask = targets[target_col].notna() & features.notna().all(axis=1)

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        tr_m = [i for i in tr_idx if valid_mask.iloc[i]]
        te_m = [i for i in te_idx if valid_mask.iloc[i]]
        if not tr_m or not te_m:
            continue

        X_tr = features.iloc[tr_m][feat_cols].values
        y_tr = targets[target_col].iloc[tr_m].values.astype(int)
        X_te = features.iloc[te_m][feat_cols].values
        y_te = targets[target_col].iloc[te_m].values.astype(int)

        if y_tr.mean() < 0.02 or y_tr.mean() > 0.98:
            continue

        if model in ("lr", "both"):
            lr_m, lr_sc = fit_lr(X_tr, y_tr)
            oof_scores_lr[te_m]  = predict_lr(lr_m, lr_sc, X_te)

        if model in ("xgb", "both"):
            xgb_m, _ = fit_xgb(X_tr, y_tr)
            if xgb_m is not None:
                oof_scores_xgb[te_m] = predict_xgb(xgb_m, None, X_te)

        oof_labels[te_m] = y_te

    mask = ~np.isnan(oof_labels) & ~np.isnan(oof_scores_lr if model != "xgb" else oof_scores_xgb)
    rets = targets["ret_wins"].values

    results = {"target": target_col, "n_oof": int(mask.sum())}

    if model in ("lr", "both") and not np.isnan(oof_scores_lr[mask]).all():
        sc = oof_scores_lr[mask]
        lb = oof_labels[mask].astype(int)
        rt = rets[mask]
        results["lr"] = {
            "auc":      round(roc_auc_score(lb, sc), 4),
            "brier":    round(brier_score_loss(lb, sc), 4),
            "ev_top20": round(top_q_ev(sc, rt, 0.20), 2),
            "ev_top10": round(top_q_ev(sc, rt, 0.10), 2),
            "ev_top5":  round(top_q_ev(sc, rt, 0.05), 2),
            "ev_top2":  round(top_q_ev(sc, rt, 0.02), 2),
        }

    if model in ("xgb", "both") and not np.isnan(oof_scores_xgb[mask]).all():
        sc = oof_scores_xgb[mask]
        lb = oof_labels[mask].astype(int)
        rt = rets[mask]
        results["xgb"] = {
            "auc":      round(roc_auc_score(lb, sc), 4),
            "brier":    round(brier_score_loss(lb, sc), 4),
            "ev_top20": round(top_q_ev(sc, rt, 0.20), 2),
            "ev_top10": round(top_q_ev(sc, rt, 0.10), 2),
            "ev_top5":  round(top_q_ev(sc, rt, 0.05), 2),
            "ev_top2":  round(top_q_ev(sc, rt, 0.02), 2),
        }

    return results, oof_scores_lr, oof_scores_xgb, oof_labels


# ── LR coefficient export ──────────────────────────────────────────────────────

def export_lr_coefs(df: pd.DataFrame, features: pd.DataFrame,
                    targets: pd.DataFrame, target_col: str):
    valid = targets[target_col].notna() & features.notna().all(axis=1)
    X = features[valid].values
    y = targets[target_col][valid].values.astype(int)
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    m = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced",
                           solver="lbfgs")
    m.fit(X_s, y)
    coefs = dict(zip(features.columns, m.coef_[0]))
    return {"intercept": float(m.intercept_[0]),
            "coefs": {k: round(float(v), 6) for k, v in
                      sorted(coefs.items(), key=lambda x: -abs(x[1]))},
            "scaler_mean": dict(zip(features.columns, sc.mean_.tolist())),
            "scaler_std":  dict(zip(features.columns, sc.scale_.tolist()))}


# ── Per-year OOF breakdown ─────────────────────────────────────────────────────

def year_breakdown(df, oof_lr, oof_xgb, oof_labels, rets):
    dates = pd.to_datetime(df["entry_date"])
    print(f"\n{'Year':>6}{'n_oof':>8}  "
          f"{'LR AUC':>8}{'LR EV20':>9}  "
          f"{'XGB AUC':>9}{'XGB EV20':>10}  "
          f"{'Base EV':>9}")
    print("─" * 70)
    for yr in sorted(dates.dt.year.unique()):
        m = (dates.dt.year == yr).values & ~np.isnan(oof_labels)
        if m.sum() < 50:
            continue
        lb = oof_labels[m].astype(int)
        rt = rets[m]
        base_ev = rt.mean()

        lr_auc = lr_ev = xgb_auc = xgb_ev = float("nan")
        if not np.isnan(oof_lr[m]).all():
            try:
                lr_auc = roc_auc_score(lb, oof_lr[m])
                lr_ev  = top_q_ev(oof_lr[m], rt, 0.20)
            except Exception:
                pass
        if not np.isnan(oof_xgb[m]).all():
            try:
                xgb_auc = roc_auc_score(lb, oof_xgb[m])
                xgb_ev  = top_q_ev(oof_xgb[m], rt, 0.20)
            except Exception:
                pass

        def fmt(v):
            return f"{v:>8.3f}" if not math.isnan(v) else f"{'—':>8}"

        print(f"{yr:>6}{m.sum():>8}  "
              f"{fmt(lr_auc)}{fmt(lr_ev)}  "
              f"{fmt(xgb_auc)}{fmt(xgb_ev)}  "
              f"{base_ev:>8.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ff-min",  type=float, default=None)
    parser.add_argument("--model",   default="both", choices=["lr", "xgb", "both"])
    parser.add_argument("--targets", nargs="+",
                        default=["bin_bigwin", "bin_profit", "bin_tp40"])
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    trades = pd.read_csv(OUT_DIR / "ff_all_trades.csv")
    sim    = pd.read_csv(OUT_DIR / "ff_exit_sim.csv")

    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    tp40   = sim[sim.strategy == "TP40"][["trade_id", "realized_ret"]].rename(
        columns={"realized_ret": "tp40_ret"})

    # Filter to trades with a return observation
    df = trades[trades["hold_mid_ret"].notna()].copy().reset_index(drop=True)
    print(f"Loaded {len(df):,} trades with exit return")

    if args.ff_min is not None:
        df = df[df["entry_ff"] > args.ff_min].reset_index(drop=True)
        print(f"After FF>{args.ff_min}% filter: {len(df):,} trades")

    # ── Features & targets ────────────────────────────────────────────────────
    features = build_features(df)
    targets  = build_targets(df, tp40)

    feat_cols = features.columns.tolist()
    print(f"Features ({len(feat_cols)}): {feat_cols}")

    target_rates = {col: targets[col].mean() for col in args.targets
                    if col in targets.columns}
    print(f"Target rates: {target_rates}")
    print(f"Walk-forward folds: {len(make_folds(df))}")

    # ── Run CV ────────────────────────────────────────────────────────────────
    all_results = []
    best_lr = best_xgb = None

    for target_col in args.targets:
        if target_col not in targets.columns:
            print(f"  Skipping {target_col} (not available)")
            continue
        print(f"\n{'═'*60}")
        print(f"  Target: {target_col}  "
              f"(positive rate: {targets[target_col].mean()*100:.1f}%)")
        print(f"{'═'*60}")

        res, oof_lr, oof_xgb, oof_labels = run_cv(
            df, features, targets, target_col, args.model)

        n = res["n_oof"]
        print(f"  OOF n={n:,}")

        for mdl in ("lr", "xgb"):
            if mdl in res:
                r = res[mdl]
                print(f"  {mdl.upper():4s}  AUC={r['auc']:.4f}  "
                      f"Brier={r['brier']:.4f}  "
                      f"EV_top20={r['ev_top20']:+.1f}%  "
                      f"EV_top10={r['ev_top10']:+.1f}%")

        # Year breakdown for best target
        if target_col == "bin_bigwin":
            rets = targets["ret_wins"].values
            year_breakdown(df, oof_lr, oof_xgb, oof_labels, rets)
            best_lr  = oof_lr
            best_xgb = oof_xgb

        all_results.append(res)

    # ── FF threshold vs ML score: feature importance ──────────────────────────
    print(f"\n{'═'*60}")
    print("  LR COEFFICIENTS (full-data fit, bin_bigwin)")
    print(f"{'═'*60}")
    if "bin_bigwin" in targets.columns:
        coef_data = export_lr_coefs(df, features, targets, "bin_bigwin")
        for feat, val in list(coef_data["coefs"].items())[:12]:
            bar = "█" * int(abs(val) * 10)
            sign = "+" if val > 0 else "-"
            print(f"  {feat:<22} {sign}{abs(val):.4f}  {bar}")
        coef_path = OUT_DIR / "ff_ml_lr_coefs_v2.json"
        with open(coef_path, "w") as f:
            json.dump(coef_data, f, indent=2)
        print(f"\n  Saved LR coefs → {coef_path.name}")

        # ── Put-calendar model (entry_debit and back_straddle halved) ──────────
        print(f"\n{'═'*60}")
        print("  PUT CALENDAR MODEL — bin_bigwin (put ≈ straddle/2 at ATM)")
        print(f"{'═'*60}")
        features_put = build_features_put(df)
        res_put, oof_lr_put, _, oof_labels_put = run_cv(
            df, features_put, targets, "bin_bigwin", "lr")
        if "lr" in res_put:
            r = res_put["lr"]
            print(f"  LR OOF  AUC={r['auc']:.4f}  "
                  f"EV_top20={r['ev_top20']:+.1f}%  "
                  f"EV_top10={r['ev_top10']:+.1f}%  "
                  f"EV_top5={r['ev_top5']:+.1f}%")
        coef_data_put = export_lr_coefs(df, features_put, targets, "bin_bigwin")
        coef_path_put = OUT_DIR / "ff_ml_lr_coefs_put_v1.json"
        with open(coef_path_put, "w") as f:
            json.dump(coef_data_put, f, indent=2)
        print(f"  Saved put LR coefs → {coef_path_put.name}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*85}")
    print("  SUMMARY — OOF metrics across targets")
    print(f"{'═'*85}")
    print(f"{'Target':<14}{'Model':>6}{'AUC':>8}{'Brier':>8}"
          f"{'EV_top20':>10}{'EV_top10':>10}{'EV_top5':>9}{'EV_top2':>9}")
    print("─" * 85)
    for res in all_results:
        for mdl in ("lr", "xgb"):
            if mdl in res:
                r = res[mdl]
                print(f"{res['target']:<14}{mdl.upper():>6}"
                      f"{r['auc']:>8.4f}{r['brier']:>8.4f}"
                      f"{r['ev_top20']:>+10.1f}%{r['ev_top10']:>+10.1f}%"
                      f"{r['ev_top5']:>+9.1f}%{r['ev_top2']:>+9.1f}%")

    # ── Trade-count & uniqueness analysis ────────────────────────────────────
    print(f"\n{'═'*85}")
    print("  POSITION SIZING — unique calendars per day (top-N% cutoffs, LR bin_bigwin)")
    print(f"{'═'*85}")

    # Rerun LR on bin_bigwin to get scores aligned with df
    if best_lr is not None:
        scores = best_lr
        valid_oof = ~np.isnan(scores) & ~np.isnan(oof_labels)
        df_oof = df[valid_oof].copy()
        df_oof["lr_score"] = scores[valid_oof]

        years = sorted(pd.to_datetime(df_oof["entry_date"]).dt.year.unique())
        print(f"\n{'Cutoff':>8}{'All trades':>13}{'trades/day':>12}"
              f"{'Uniq cal/day':>14}{'Mean EV':>10}")
        print("─" * 60)

        trading_days = df_oof["entry_date"].nunique()
        for pct in [0.20, 0.10, 0.05, 0.02, 0.01]:
            thresh = np.quantile(df_oof["lr_score"], 1 - pct)
            sub = df_oof[df_oof["lr_score"] >= thresh]
            n_trades = len(sub)
            n_per_day = n_trades / trading_days
            # unique calendar = unique (ticker, front_exp, back_exp)
            n_uniq_cal = sub.groupby("entry_date").apply(
                lambda g: g[["ticker","front_exp","back_exp"]].drop_duplicates().shape[0]
            ).mean()
            mean_ev = sub["hold_mid_ret"].mean() if "hold_mid_ret" in sub else float("nan")
            print(f"{pct*100:>7.0f}%{n_trades:>13,}{n_per_day:>12.1f}"
                  f"{n_uniq_cal:>14.1f}{mean_ev:>+10.1f}%")

        # Also show: if we DEDUPLICATE (only first entry per calendar)
        print(f"\n  ── Deduped: first entry only per (ticker, front_exp, back_exp) ──")
        for pct in [0.20, 0.10, 0.05, 0.02, 0.01]:
            thresh = np.quantile(df_oof["lr_score"], 1 - pct)
            sub = df_oof[df_oof["lr_score"] >= thresh].copy()
            sub_dedup = sub.sort_values("entry_date").drop_duplicates(
                subset=["ticker","front_exp","back_exp"], keep="first")
            n_per_day = len(sub_dedup) / trading_days
            mean_ev   = sub_dedup["hold_mid_ret"].mean()
            print(f"{pct*100:>7.0f}%  deduped: {len(sub_dedup):>7,} trades"
                  f"  {n_per_day:.1f}/day  EV={mean_ev:+.1f}%")


if __name__ == "__main__":
    main()
