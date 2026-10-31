"""Day 5: train an Isolation Forest on the simulated transactions and measure it.

    pip install scikit-learn
    python -m risk.train

Isolation Forest is UNSUPERVISED: it never sees the fraud labels while learning.
It learns what normal traffic looks like and scores how odd each payment is.
The labels are used only afterwards, to grade it.
"""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from risk.features import FEATURE_NAMES, build_features


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def metrics(y_true, flagged):
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, tp, fp, fn


def train(rows, train_share=0.7, flag_rate=0.03, seed=42):
    """Fit on the earlier part of the timeline, test on the later part."""
    X = np.array(build_features(rows))
    y = np.array([int(r["is_fraud"]) for r in rows])
    kinds = np.array([r["fraud_type"] for r in rows])
    cut = int(len(rows) * train_share)

    model = IsolationForest(n_estimators=200, random_state=seed)
    model.fit(X[:cut])

    # Higher score = more anomalous. The alert threshold is set on TRAINING data
    # so that about `flag_rate` of normal-looking traffic would be flagged.
    train_scores = -model.score_samples(X[:cut])
    threshold = float(np.quantile(train_scores, 1 - flag_rate))
    test_scores = -model.score_samples(X[cut:])
    return model, threshold, test_scores, y[cut:], kinds[cut:]


def report(model, threshold, scores, y, kinds):
    print(f"Test set: {len(y):,} transactions, {int(y.sum())} fraud ({y.mean():.2%})")
    print(f"ROC AUC: {roc_auc_score(y, scores):.3f}   (1.0 = perfect ranking, 0.5 = random)\n")

    flagged = scores >= threshold
    precision, recall, tp, fp, fn = metrics(y, flagged)
    print(f"At the chosen threshold, {flagged.mean():.2%} of transactions are flagged:")
    print(f"  fraud caught (recall):   {recall:.1%}  ({tp} of {tp + fn})")
    print(f"  flags that are fraud:    {precision:.1%}  (precision; {fp} false alarms)\n")

    print("Recall by fraud type:")
    for kind in sorted(set(kinds) - {""}):
        mask = kinds == kind
        print(f"  {kind:<17} {flagged[mask].mean():.1%}  ({int(flagged[mask].sum())} of {int(mask.sum())})")

    print("\nTrade-off: flag more, catch more fraud but raise more false alarms")
    print("  flag top   recall   precision   false alarms")
    for share in (0.01, 0.02, 0.03, 0.05, 0.10):
        cutoff = np.quantile(scores, 1 - share)
        p, r, _, f, _ = metrics(y, scores >= cutoff)
        print(f"  {share:>7.0%}   {r:>6.1%}   {p:>8.1%}   {f:>8}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/transactions.csv")
    ap.add_argument("--out", default="models/risk_model.joblib")
    ap.add_argument("--flag-rate", type=float, default=0.03)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    model, threshold, scores, y, kinds = train(rows, flag_rate=args.flag_rate)
    report(model, threshold, scores, y, kinds)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURE_NAMES}, args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
