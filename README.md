# PayGuard: Payment Infrastructure & AI Risk Engine

A payment backend where every transfer is judged by a **hybrid fraud risk engine** (fixed rules plus an Isolation Forest model) before any money moves. Built on a **double-entry ledger** with **idempotent transfers**, and tested with a **synthetic fraud simulator**.

FastAPI, PostgreSQL, scikit-learn, Docker. Everything uses test money; no real payments are processed.

## What it does

- **Double-entry ledger.** Every transfer is a debit plus a matching credit. Money is stored as integers (paise), never floats. Balances are computed from entries, never edited.
- **Rules enforced by the database itself.** A deferred constraint trigger rejects any transaction whose debits and credits don't match, and a trigger blocks updates and deletes on ledger entries.
- **Safe under concurrency.** Transfers lock both account rows in a fixed order (no deadlocks) and check funds after locking, so simultaneous requests can't spend the same money.
- **Idempotent API.** `POST /transfers` requires an `Idempotency-Key` header. A retry with the same key returns the original response instead of moving money again. The key, ledger entries, audit row and saved response are written in one database transaction.
- **Risk engine in the payment path.** Each transfer is approved, held for human review, or blocked, with plain-English reasons. Every decision is stored in an audit table.
- **Review queue.** Held payments move no money until a person approves or rejects them.
- **Transaction simulator.** Generates thousands of realistic payments with three labelled fraud patterns and can replay them through the real API.

## How a payment is decided

| Decision | What happens | HTTP |
|---|---|---|
| approve | Posted to the ledger | 201 |
| review | Held, no money moves until approved at `/reviews` | 202 |
| block | Refused, with reasons | 403 |

The engine combines two things. Any block rule blocks; otherwise the model's score decides.

**Fixed rules:** a hard amount limit, velocity (a 3rd payment inside a minute goes to review, a 4th is blocked), repeated round payments just under Rs 50,000, and a takeover pattern (8x or more the sender's usual amount on a new device and a new recipient).

**Model:** an Isolation Forest trained without fraud labels on 8 features per payment (amount, amount compared with the sender's usual, payments in the last minute and hour, new recipient, new device, hour of day, round amount). Every feature uses only the sender's past activity. A payment more unusual than 97% of normal traffic goes to review, and more unusual than 99.5% is blocked.

Blocked payments are not remembered as normal behaviour, so an attacker can't teach the engine that a new device is fine.

Example decision:

```
Rs 49,500 to an unfamiliar recipient: BLOCK (risk 0.99)
  - repeated round payments just under Rs 50,000
  - behaviour is more unusual than 98.8% of normal traffic
```

## Results

All numbers come from simulated data, where I designed the fraud patterns. They show the pipeline works; they are not a claim about accuracy on real-world fraud.

**Held-out period (the last 30% of a 30-day timeline, never used for training):** 3,477 payments, 76 fraud.

| | Blocked | Held for review | Approved |
|---|---|---|---|
| Fraud | 47.4% | 42.1% | 10.5% (missed) |
| Normal | 0% | 1.0% | 99.0% |

The model alone scores ROC AUC 0.989. Most missed fraud is the first payments of a velocity burst, which look ordinary one at a time.

**Live replay through the API** (1,404 payments, 42 fraud, risk engine in the request path):

| | Blocked | Held for review | Approved |
|---|---|---|---|
| Fraud (42) | 43% | 40% | 17% |
| Normal (1,362) | 0% | 1% | 99% |

Sequential replay: 16.7 requests/second, p50 48 ms, p95 101 ms (laptop, Docker Desktop, single API process). Scoring costs roughly 12 ms per payment. Before the risk engine was added, a ledger-only load test reached 113.9 requests/second with 8 parallel workers.

## Quick start

You need Docker Desktop and Python 3.12 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python simulator/generate_dataset.py     # writes data/transactions.csv
python -m risk.train                     # trains the model into models/

docker compose up --build
```

If `models/risk_model.joblib` is missing, the API still runs using the rules only (`/health` shows `"risk_engine": "rules_only"`).

Open http://localhost:8000/docs. A good first try:

1. `POST /accounts` twice (copy both ids)
2. `POST /deposits` to give the first account test money (amounts are in paise, so `10000` = Rs 100.00)
3. `POST /transfers` with an `Idempotency-Key` such as `demo-key-001`, then click Execute again; the second response has the header `idempotent-replay: true`

To see the risk engine work on realistic traffic:

```bash
python simulator/generate_dataset.py --users 60 --fraud-rate 0.03 --seed 7 --out data/live.csv
python simulator/run_live.py --csv data/live.csv --limit 1500
```

Then open `GET /reviews` to see the held payments and approve one with `POST /reviews/{id}/approve`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/accounts` | Create an account |
| GET | `/accounts/{id}` | Account details and computed balance |
| GET | `/accounts/{id}/entries` | Latest ledger entries |
| POST | `/deposits` | Test-money faucet (moves money from a system Treasury account) |
| POST | `/transfers` | Risk-checked transfer (requires `Idempotency-Key`, optional `device_id`) |
| GET | `/reviews` | Payments held for review |
| POST | `/reviews/{id}/approve` | Post a held payment |
| POST | `/reviews/{id}/reject` | Cancel a held payment |
| GET | `/health` | Database check and which risk engine is running |

Transfer errors: `404` unknown account, `409` insufficient funds, `422` invalid request or a reused key with a different body.

## Tests

```bash
docker compose up -d db
pytest -v
```

50 tests (about 2.5 minutes). 24 need the database: ledger rules, transfers, idempotency, and the risk engine inside the API. 26 need no database: feature engineering, the engine and the simulator. Highlights: unbalanced transactions are rejected, ledger entries are immutable, the whole ledger always sums to zero, 20 parallel transfers never overdraw an account, 10 simultaneous duplicate requests move money once, features never use future data, blocked payments move no money, and held payments post only when approved.

API tests run on the rules only with the clock set by a header, so they give the same result at any time of day.

## Project structure

```
app/
  api/          accounts, transfers, reviews
  risk_service.py   shared risk engine for the API
  main.py
risk/           features, model training, hybrid engine, evaluation
simulator/      dataset generator and live replay
db/             SQL schema, run in order: 001_init, 002_idempotency, 003_risk
tests/
docker-compose.yml
```

## Design decisions

- **Integers for money.** Floats can't represent amounts like 0.10 exactly.
- **Guarantees live in the database.** Application code can have bugs; a constraint trigger can't be skipped by one.
- **Lock order.** Accounts are always locked in sorted order, so opposite transfers can't deadlock.
- **Decisions are idempotent too.** A retried request that was blocked or held gets the same answer back. Failed requests (such as insufficient funds) are not remembered, so the client can fix the problem and retry with the same key.
- **Rules plus a model.** Rules are instant and auditable; the model catches odd behaviour nobody wrote a rule for. The engine falls back to rules only if the model can't load.
- **No leakage.** Every feature comes from the sender's past only, and a test proves it.

## Limitations

- The engine keeps sender history in memory, so it resets when the API restarts and works with a single API process.
- `RISK_SIMULATION_MODE=1` (set in `docker-compose.yml`) lets the simulator set the clock. It is for development only; in production a client could fake timestamps to dodge the velocity rules, so remove it.
- Trained and evaluated on simulated data only.
- No authentication, rate limiting or regulatory compliance work. This is a learning and portfolio project, not production banking software.

## Roadmap

- [x] Double-entry ledger with database-enforced rules
- [x] Transfers with row locking and concurrency tests
- [x] Idempotency keys
- [x] Transaction simulator and load test
- [x] Feature engineering and Isolation Forest model
- [x] Hybrid risk engine with reasons, wired into the transfer flow
- [x] Review queue and risk audit log
- [ ] Rebuild engine history from the audit log on restart
- [ ] Monitoring dashboard (decisions over time, model drift)

## License

MIT