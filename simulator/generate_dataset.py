"""Day 4: synthetic transaction generator.

Produces a CSV of realistic payments: mostly normal behaviour, with three
kinds of fraud injected and labelled. Day 5 trains the risk engine on this file.

Pure Python standard library, so there is nothing extra to install.

    python simulator/generate_dataset.py
    python simulator/generate_dataset.py --users 1000 --days 60 --fraud-rate 0.03

Fraud patterns (each labelled in the `fraud_type` column):
  account_takeover  new device, 1-4 am, amount 8-20x the user's usual, new recipient
  velocity_burst    5-12 transfers within two minutes to unfamiliar recipients
  structuring       4-8 round transfers just under Rs 50,000 to one recipient
"""
import argparse
import csv
import math
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

START = datetime(2026, 9, 1, tzinfo=timezone.utc)
MAX_RUPEES = 200_000

# How likely a normal payment is at each hour 0-23: quiet at night, busy in the evening.
HOUR_WEIGHTS = [1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 9, 10, 9, 8, 8, 9, 10, 11, 11, 9, 6, 3, 2]

FIELDS = [
    "txn_id", "timestamp", "sender_id", "recipient_id",
    "amount_paise", "device_id", "is_fraud", "fraud_type",
]


def poisson(rng, lam):
    """Random count with average `lam` (Knuth's method)."""
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def make_row(ts, sender, recipient, rupees, device, fraud_type=""):
    rupees = max(10, min(MAX_RUPEES, int(round(rupees))))
    return {
        "timestamp": ts.isoformat(),
        "sender_id": sender,
        "recipient_id": recipient,
        "amount_paise": rupees * 100,
        "device_id": device,
        "is_fraud": 1 if fraud_type else 0,
        "fraud_type": fraud_type,
    }


def make_profiles(rng, user_ids, merchants):
    """Each user gets a typical amount, an activity level, devices and regular recipients."""
    profiles = []
    for uid in user_ids:
        regulars = rng.sample(merchants, rng.randint(3, 8))
        regulars += [u for u in rng.sample(user_ids, rng.randint(1, 4)) if u != uid]
        profiles.append({
            "id": uid,
            "avg_rupees": rng.lognormvariate(math.log(1500), 0.8),
            "rate": max(0.2, rng.lognormvariate(math.log(0.7), 0.6)),  # payments per day
            "devices": [f"dev_{uid}_{c}" for c in "ab"[: rng.randint(1, 2)]],
            "regulars": regulars,
        })
    return profiles


def normal_transactions(rng, profiles, merchants, user_ids, days):
    rows = []
    for p in profiles:
        for day in range(days):
            for _ in range(poisson(rng, p["rate"])):
                hour = rng.choices(range(24), HOUR_WEIGHTS)[0]
                ts = START + timedelta(days=day, hours=hour,
                                       minutes=rng.randrange(60), seconds=rng.randrange(60))
                if rng.random() < 0.85:
                    recipient = rng.choice(p["regulars"])
                else:  # occasionally pays someone new, which is perfectly normal
                    recipient = rng.choice(merchants + user_ids)
                    if recipient == p["id"]:
                        continue
                device = rng.choice(p["devices"]) if rng.random() < 0.95 else f"dev_{p['id']}_x{rng.randrange(1000, 9999)}"
                rupees = rng.lognormvariate(math.log(p["avg_rupees"]), 0.5)
                rupees = round(rupees, -1) if rng.random() < 0.5 else rupees
                rows.append(make_row(ts, p["id"], recipient, rupees, device))
    return rows


def account_takeover(rng, profiles, mules, days):
    p = rng.choice(profiles)
    base = START + timedelta(days=rng.randrange(days), hours=rng.randint(1, 4), minutes=rng.randrange(60))
    device = f"dev_new_{rng.randrange(10**6):06d}"
    rows = []
    for i in range(rng.randint(1, 3)):
        ts = base + timedelta(minutes=i * rng.randint(1, 10))
        rows.append(make_row(ts, p["id"], rng.choice(mules),
                             p["avg_rupees"] * rng.uniform(8, 20), device, "account_takeover"))
    return rows


def velocity_burst(rng, profiles, mules, days):
    p = rng.choice(profiles)
    base = START + timedelta(days=rng.randrange(days), hours=rng.randrange(24), minutes=rng.randrange(60))
    device = rng.choice(p["devices"]) if rng.random() < 0.5 else f"dev_new_{rng.randrange(10**6):06d}"
    return [
        make_row(base + timedelta(seconds=rng.randrange(120)), p["id"], rng.choice(mules),
                 rng.randint(100, 5000), device, "velocity_burst")
        for _ in range(rng.randint(5, 12))
    ]


def structuring(rng, profiles, mules, days):
    p = rng.choice(profiles)
    base = START + timedelta(days=rng.randrange(days), hours=rng.randint(9, 18), minutes=rng.randrange(60))
    mule = rng.choice(mules)
    rows = []
    for i in range(rng.randint(4, 8)):
        ts = base + timedelta(minutes=i * rng.randint(5, 25))
        rows.append(make_row(ts, p["id"], mule, rng.choice([49000, 49500, 49800, 49900]),
                             rng.choice(p["devices"]), "structuring"))
    return rows


def generate(users=500, days=30, seed=42, fraud_rate=0.02):
    """Return a list of transaction dicts, sorted by time. Same seed, same data."""
    rng = random.Random(seed)
    merchants = [f"m{i:03d}" for i in range(1, 101)]
    mules = [f"mule{i:03d}" for i in range(1, 31)]
    user_ids = [f"u{i:04d}" for i in range(1, users + 1)]

    profiles = make_profiles(rng, user_ids, merchants)
    rows = normal_transactions(rng, profiles, merchants, user_ids, days)

    target = round(len(rows) * fraud_rate / (1 - fraud_rate))
    fraud = []
    while len(fraud) < target:
        maker = rng.choice([account_takeover, velocity_burst, structuring])
        fraud += maker(rng, profiles, mules, days)

    rows += fraud
    rows.sort(key=lambda r: r["timestamp"])
    for i, row in enumerate(rows, 1):
        row["txn_id"] = f"t{i:07d}"
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, default=500)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraud-rate", type=float, default=0.02, help="share of rows that are fraud (default 0.02)")
    ap.add_argument("--out", default="data/transactions.csv")
    args = ap.parse_args()

    rows = generate(args.users, args.days, args.seed, args.fraud_rate)
    write_csv(rows, args.out)

    fraud = [r for r in rows if r["is_fraud"]]
    print(f"Wrote {len(rows):,} transactions to {args.out}")
    print(f"Fraud rows: {len(fraud):,} ({len(fraud) / len(rows):.2%})")
    for kind, n in sorted(Counter(r["fraud_type"] for r in fraud).items()):
        print(f"  {kind}: {n}")
    normal_amounts = sorted(r["amount_paise"] / 100 for r in rows if not r["is_fraud"])
    fraud_amounts = sorted(r["amount_paise"] / 100 for r in fraud)
    print(f"Median normal amount: Rs {normal_amounts[len(normal_amounts) // 2]:,.0f}")
    print(f"Median fraud amount:  Rs {fraud_amounts[len(fraud_amounts) // 2]:,.0f}")


if __name__ == "__main__":
    main()
