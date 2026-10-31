# Payment Infrastructure & AI Risk Engine

Backend payment system with a double-entry ledger, idempotent APIs and an AI risk engine.

## Day 1: run it
```bash
docker compose up --build
# in another terminal
curl localhost:8000/health
curl -X POST localhost:8000/accounts -H 'Content-Type: application/json' \
  -d '{"owner_name":"Alice","account_type":"user"}'
```
API docs: http://localhost:8000/docs

## Run the schema tests
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d db
pytest -v
```

## Reset the database (re-runs db/init.sql)
```bash
docker compose down -v
```
