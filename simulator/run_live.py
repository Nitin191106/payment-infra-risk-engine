"""Day 4: replay simulated transactions through the real API.

Creates accounts for the people in the CSV, gives them test money, then sends
the transfers to POST /transfers with idempotency keys. Prints throughput and
latency, which doubles as a small load test.

    python simulator/run_live.py                 # first 300 transactions, in order
    python simulator/run_live.py --limit 2000 --workers 8

Needs the API running (docker compose up) and `python simulator/generate_dataset.py` run first.
"""
import argparse
import csv
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx

DEPOSIT_PAISE = 100_000_000  # Rs 10 lakh of test money per sender


def account_type(sim_id):
    return "merchant" if sim_id[0] == "m" and sim_id[1].isdigit() else "user"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/transactions.csv")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (1 keeps the original order)")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = [r for _, r in zip(range(args.limit), csv.DictReader(f))]
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    client = httpx.Client(base_url=args.api, timeout=30)
    run_tag = uuid.uuid4().hex[:8]  # new keys each run, so reruns create fresh transfers

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
            },
            headers={"Idempotency-Key": f"sim-{row['txn_id']}-{run_tag}"},
        )
        return resp.status_code, time.perf_counter() - started

    print(f"Sending {len(rows)} transfers with {args.workers} worker(s)...")
    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(send, rows))
    wall = time.perf_counter() - wall

    statuses = {}
    for code, _ in results:
        statuses[code] = statuses.get(code, 0) + 1
    latencies = sorted(lat * 1000 for _, lat in results)
    print(f"Status codes: {statuses}")
    print(f"Throughput: {len(rows) / wall:,.1f} requests/second")
    print(f"Latency: p50 {latencies[len(latencies) // 2]:.0f} ms, "
          f"p95 {latencies[int(len(latencies) * 0.95) - 1]:.0f} ms")


if __name__ == "__main__":
    main()
