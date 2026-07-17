#!/usr/bin/env python3
"""
ff_ml_experiments.py — model/target experiments on top of the honest harness.

Reuses ff_ml_eval's purged walk-forward infra. Answers:
  - Two-model "barbell": P(big win) - lambda*P(big loss) vs single bigwin.
  - EV-weighted ordinal (coarse bins, predict expected EV) vs 5%-increment binning.
  - Does XGBoost's small AUC edge translate to top-quintile EV, or is LR enough?
All metrics are pooled out-of-fold (purged WF). Metric = top-quintile realized EV.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from ff_ml_eval import (prep, purged_folds, impute, col_medians, topq_ev, auc,
                        BIG_WIN_THRESH)
from ff_ml_analysis import load_data

BIG_LOSS_THRESH = -50.0


def oof_predictions(data, builder):
    """builder(Xtr, ntr) -> fitted-scorer(Xte)->scores. Returns pooled (score, net)."""
    from sklearn.preprocessing import StandardScaler
    net = data["net"]
    S, N = [], []
    for tr, te, _ in purged_folds(data["scan"], data["held"]):
        med = col_medians(data["X_raw"][tr])
        Xtr = impute(data["X_raw"][tr], med); Xte = impute(data["X_raw"][te], med)
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        scorer = builder(Xtr, net[tr])
        if scorer is None:
            continue
        S.extend(scorer(Xte).tolist()); N.extend(net[te].tolist())
    return np.array(S), np.array(N)


def lr_binary(thresh, above=True):
    from sklearn.linear_model import LogisticRegression
    def build(Xtr, ntr):
        y = (ntr > thresh).astype(int) if above else (ntr < thresh).astype(int)
        if len(set(y)) < 2:
            return None
        m = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced").fit(Xtr, y)
        return lambda Xte: m.predict_proba(Xte)[:, 1]
    return build


def barbell(lam=1.0):
    """Score = P(bigwin) - lam*P(bigloss)."""
    from sklearn.linear_model import LogisticRegression
    def build(Xtr, ntr):
        yw = (ntr > BIG_WIN_THRESH).astype(int)
        yl = (ntr < BIG_LOSS_THRESH).astype(int)
        if len(set(yw)) < 2 or len(set(yl)) < 2:
            return None
        mw = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced").fit(Xtr, yw)
        ml = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced").fit(Xtr, yl)
        return lambda Xte: mw.predict_proba(Xte)[:, 1] - lam * ml.predict_proba(Xte)[:, 1]
    return build


def ev_ordinal(bins, above_clip=300, below_clip=-100):
    """Multiclass LR over coarse bins; predict EV = sum(p_class * train_class_mean)."""
    from sklearn.linear_model import LogisticRegression
    def build(Xtr, ntr):
        cls = np.digitize(ntr, bins)
        means = {}
        for c in np.unique(cls):
            means[c] = float(np.clip(ntr[cls == c], below_clip, above_clip).mean())
        if len(means) < 2:
            return None
        m = LogisticRegression(C=0.1, max_iter=1000,
                               class_weight="balanced").fit(Xtr, cls)
        classes = m.classes_
        mean_vec = np.array([means[c] for c in classes])
        return lambda Xte: m.predict_proba(Xte) @ mean_vec
    return build


def xgb_builder(thresh=0.0):
    def build(Xtr, ntr):
        try:
            import xgboost as xgb
        except ImportError:
            return None
        y = (ntr > thresh).astype(int)
        if len(set(y)) < 2:
            return None
        m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              random_state=42, n_jobs=-1).fit(Xtr, y)
        return lambda Xte: m.predict_proba(Xte)[:, 1]
    return build


def main():
    records = load_data()
    for structure in ("straddle", "put"):
        data = prep(records, structure)
        print("=" * 64)
        print(f"  {structure.upper()}  (n={data['n']}, top-quintile realized EV, OOF)")
        print("=" * 64)
        experiments = [
            ("LR bin_profit",        lr_binary(0.0, above=True)),
            ("LR bin_bigwin",        lr_binary(BIG_WIN_THRESH, above=True)),
            ("LR avoid_bigloss",     lambda: None),  # placeholder, handled below
            ("LR barbell λ=0.5",     barbell(0.5)),
            ("LR barbell λ=1.0",     barbell(1.0)),
            ("LR barbell λ=2.0",     barbell(2.0)),
            ("LR EV-ordinal 5-bin",  ev_ordinal([-75, -10, 10, 50])),
            ("LR EV-ordinal 9-bin",  ev_ordinal([-75,-40,-15,0,15,40,80,150])),
            ("XGB bin_profit",       xgb_builder(0.0)),
            ("XGB bin_bigwin",       xgb_builder(BIG_WIN_THRESH)),
        ]
        for name, builder in experiments:
            if name == "LR avoid_bigloss":
                # rank by -P(bigloss)
                s, n = oof_predictions(data, lr_binary(BIG_LOSS_THRESH, above=False))
                s = -s
            else:
                s, n = oof_predictions(data, builder)
            if len(s) == 0:
                print(f"  {name:<22} n/a"); continue
            ev = topq_ev(s, n)
            print(f"  {name:<22} top20% EV = {ev:+6.1f}%")
        print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
