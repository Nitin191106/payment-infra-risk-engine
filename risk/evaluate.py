"""Day 6: grade the hybrid engine on the held-out part of the simulated timeline.

    python -m risk.train        # first, so the saved model has the new fields
    python -m risk.evaluate

Runs every transaction through the engine in time order, exactly like live
traffic, and compares decisions with the true fraud labels.
"""
import argparse
from collections import Counter

from risk.engine import RiskEngine
from risk.train import load_rows


def run_stream(rows, engine):
    decisions = []
    for row in rows:
        d = engine.assess(row)
        engine.record(row, d)
        decisions.append(d)
    return decisions


def share(n, total):
    return f"{n / total:.1%}" if total else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/transactions.csv")
    ap.add_argument("--model", default="models/risk_model.joblib")
    ap.add_argument("--train-share", type=float, default=0.7)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    engine = RiskEngine.load(args.model)
    decisions = run_stream(rows, engine)

    cut = int(len(rows) * args.train_share)
    test = list(zip(rows[cut:], decisions[cut:]))
    fraud = [(r, d) for r, d in test if r["is_fraud"] == "1"]
    normal = [(r, d) for r, d in test if r["is_fraud"] == "0"]

    print(f"Held-out period: {len(test):,} transactions, {len(fraud)} fraud\n")

    print("What happened to FRAUD payments:")
    c = Counter(d.decision for _, d in fraud)
    for name in ("block", "review", "approve"):
        label = {"block": "blocked", "review": "sent to review", "approve": "approved (missed)"}[name]
        print(f"  {label:<19} {share(c[name], len(fraud)):>6}  ({c[name]})")

    print("\nWhat happened to NORMAL payments:")
    c = Counter(d.decision for _, d in normal)
    for name in ("approve", "review", "block"):
        label = {"approve": "approved", "review": "sent to review", "block": "wrongly blocked"}[name]
        print(f"  {label:<19} {share(c[name], len(normal)):>6}  ({c[name]})")

    print("\nBy fraud type (blocked / review / missed):")
    for kind in sorted({r["fraud_type"] for r, _ in fraud}):
        group = [d for r, d in fraud if r["fraud_type"] == kind]
        c = Counter(d.decision for d in group)
        print(f"  {kind:<17} {c['block']:>3} / {c['review']:>3} / {c['approve']:>3}   (of {len(group)})")

    cfg = engine.config
    caught = [d for _, d in fraud if d.decision != "approve"]
    by_rule = sum(1 for d in caught if d.rules and d.model_percentile < cfg.review_percentile)
    by_model = sum(1 for d in caught if not d.rules)
    both = len(caught) - by_rule - by_model
    print(f"\nWho caught the {len(caught)} fraud payments that were flagged:")
    print(f"  rules only {by_rule}, model only {by_model}, both {both}")

    def show(title, pair):
        print(f"\n{title}")
        if pair is None:
            print("  (none)")
            return
        r, d = pair
        print(f"  {r['sender_id']} -> {r['recipient_id']}  Rs {int(r['amount_paise']) / 100:,.0f}"
              f"  [{r['fraud_type'] or 'normal'}]  => {d.decision.upper()} (risk {d.risk_score:.2f})")
        for reason in d.reasons:
            print(f"    - {reason}")

    show("Example: fraud that was blocked",
         next(((r, d) for r, d in fraud if d.decision == "block" and len(d.reasons) >= 2), None))
    show("Example: fraud that slipped through",
         next(((r, d) for r, d in fraud if d.decision == "approve"), None))
    show("Example: a normal payment that was flagged",
         next(((r, d) for r, d in normal if d.decision != "approve"), None))


if __name__ == "__main__":
    main()
