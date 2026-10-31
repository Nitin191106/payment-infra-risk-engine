"""Replay simulated transactions through the real API, risk engine included.

Creates accounts for the people in the CSV, gives them test money, then sends
the transfers to POST /transfers with idempotency keys. Each payment carries its
device and its simulated time, so the risk engine sees the same story the
dataset tells. Prints what happened to fraud and to normal payments, plus
throughput and latency.

    python simulator/generate_dataset.py --users 60 --fraud-rate 0.03 --seed 7 --out data/live.csv
    python simulator/run_live.py --csv data/live.csv --limit 1500

Needs the API running with RISK_SIMULATION_MODE=1 (docker-compose.yml sets it).
Keep --workers 1: the risk engine expects each sender's payments in time order.
"""
import argparse
import csv
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx

DEPOSIT_PAISE = 100_000_000  # Rs 10 lakh of test money per sender
OUTCOME = {201: "approved", 202: "held for review", 403: "blocked"}


def account_type(sim_id):
    return "merchant" if sim_id[0] == "m" and sim_id[1].isdigit() else "user"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/transactions.csv")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (keep 1 for ordered replay)")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = [r for _, r in zip(range(args.limit), csv.DictReader(f))]
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    client = httpx.Client(base_url=args.api, timeout=60)
    run_tag = uuid.uuid4().hex[:8]  # new keys each run, so reruns create fresh transfers

    health = client.get("/health").json()
    print(f"Risk engine in the API: {health.get('risk_engine', 'unknown')}")

    # 1. Accounts for everyone who appears, and test money for the senders.
    people = sorted({r["sender_id"] for r in rows} | {r["recipient_id"] for r in rows})
    senders = {r["sender_id"] for r in rows}
    ids = {}
    print(f"Creating {len(people)} accounts...")
    for sim_id in people:
        resp = client.post("/accounts", json={"owner_name": sim_id, "account_type": account_type(sim_id)})
        resp.raise_for_status()
        ids[sim_id] = resp.json()["id"]
    for sim_id in senders:
        client.post("/deposits", json={"account_id": ids[sim_id], "amount": DEPOSIT_PAISE}).raise_for_status()

    # 2. Replay the transfers.
    def send(row):
        started = time.perf_counter()
        resp = client.post(
            "/transfers",
            json={
                "from_account_id": ids[row["sender_id"]],
                "to_account_id": ids[row["recipient_id"]],
                "amount": int(row["amount_paise"]),
                "description": f"sim {row['txn_id']}",
                "device_id": row["device_id"],
            },
            headers={
                "Idempotency-Key": f"sim-{row['txn_id']}-{run_tag}",
                "X-Simulated-Time": row["timestamp"],
            },
        )
        return row, resp.status_code, time.perf_counter() - started

    print(f"Sending {len(rows)} transfers with {args.workers} worker(s)...")
    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(send, rows))
    wall = time.perf_counter() - wall

    print(f"\nStatus codes: {dict(Counter(code for _, code, _ in results))}")
    for label, name in (("1", "FRAUD"), ("0", "NORMAL")):
        group = [code for row, code, _ in results if row["is_fraud"] == label]
        if not group:
            continue
        counts = Counter(group)
        parts = ", ".join(f"{OUTCOME.get(c, c)} {counts[c] / len(group):.0%} ({counts[c]})"
                          for c in sorted(counts))
        print(f"{name} payments ({len(group)}): {parts}")

    latencies = sorted(lat * 1000 for _, _, lat in results)
    print(f"\nThroughput: {len(rows) / wall:,.1f} requests/second")
    print(f"Latency: p50 {latencies[len(latencies) // 2]:.0f} ms, "
          f"p95 {latencies[int(len(latencies) * 0.95) - 1]:.0f} ms")
    print("Held payments are waiting at GET /reviews.")


if __name__ == "__main__":
    main()
