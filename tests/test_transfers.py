"""Day 2 + Day 3 tests. Need the database running:  docker compose up -d db"""
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.api.transfers import get_balance, post_transfer
from app.db.session import SessionLocal, engine
from app.main import app

client = TestClient(app)


def new_account(name, kind="user"):
    r = client.post("/accounts", json={"owner_name": name, "account_type": kind})
    assert r.status_code == 201
    return r.json()["id"]


def fund(account_id, amount):
    r = client.post("/deposits", json={"account_id": account_id, "amount": amount})
    assert r.status_code == 201
    return r.json()


def transfer(src, dst, amount, key=None, http=None):
    """Send a transfer with an Idempotency-Key (a fresh one unless given)."""
    http = http or client
    return http.post(
        "/transfers",
        json={"from_account_id": src, "to_account_id": dst, "amount": amount},
        headers={"Idempotency-Key": key or str(uuid4())},
    )


def balance_of(account_id):
    with SessionLocal() as db:
        return get_balance(db, UUID(account_id))


# ---------- Day 2 behaviour (still must work) ----------

def test_deposit_then_transfer_moves_money():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 10_000)
    r = transfer(alice, bob, 2_500)
    assert r.status_code == 201
    assert r.json()["from_balance"] == 7_500
    assert r.json()["to_balance"] == 2_500


def test_insufficient_funds_is_rejected_and_nothing_moves():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 1_000)
    r = transfer(alice, bob, 5_000)
    assert r.status_code == 409
    assert balance_of(alice) == 1_000
    assert balance_of(bob) == 0


def test_cannot_transfer_to_same_account():
    alice = new_account("Alice")
    fund(alice, 1_000)
    assert transfer(alice, alice, 100).status_code == 422


def test_unknown_account_returns_404():
    alice = new_account("Alice")
    fund(alice, 1_000)
    r = transfer(alice, "00000000-0000-0000-0000-000000000000", 100)
    assert r.status_code == 404


def test_amount_must_be_positive():
    alice, bob = new_account("Alice"), new_account("Bob")
    assert transfer(alice, bob, -5).status_code == 422


def test_whole_ledger_always_sums_to_zero():
    """Money is never created or destroyed: all balances across all accounts add to 0."""
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 9_999)
    transfer(alice, bob, 1_234)
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COALESCE(SUM(balance), 0) FROM account_balances")).scalar_one()
    assert int(total) == 0


def test_concurrent_transfers_never_overdraw():
    """20 parallel transfers of 100 from an account holding 1000: exactly 10 may succeed."""
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 1_000)

    def attempt(_):
        with SessionLocal() as db:
            try:
                post_transfer(db, UUID(alice), UUID(bob), 100, "race test")
                return True
            except HTTPException:
                return False

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(attempt, range(20)))

    assert sum(results) == 10
    assert balance_of(alice) == 0
    assert balance_of(bob) == 1_000


# ---------- Day 3: idempotency ----------

def test_missing_idempotency_key_is_rejected():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 1_000)
    r = client.post(
        "/transfers",
        json={"from_account_id": alice, "to_account_id": bob, "amount": 100},
    )
    assert r.status_code == 422
    assert balance_of(alice) == 1_000


def test_same_key_twice_moves_money_only_once():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 10_000)
    key = str(uuid4())

    first = transfer(alice, bob, 3_000, key=key)
    second = transfer(alice, bob, 3_000, key=key)

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.headers.get("Idempotent-Replay") == "true"
    assert first.headers.get("Idempotent-Replay") is None
    assert second.json() == first.json()          # identical stored response
    assert balance_of(alice) == 7_000             # charged once, not twice
    assert balance_of(bob) == 3_000


def test_same_key_with_different_request_is_rejected():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 10_000)
    key = str(uuid4())

    assert transfer(alice, bob, 1_000, key=key).status_code == 201
    r = transfer(alice, bob, 9_000, key=key)      # same key, different amount
    assert r.status_code == 422
    assert balance_of(alice) == 9_000             # only the first transfer happened


def test_failed_request_is_not_remembered_so_retry_can_succeed():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 500)
    key = str(uuid4())

    assert transfer(alice, bob, 2_000, key=key).status_code == 409   # not enough money
    fund(alice, 5_000)
    r = transfer(alice, bob, 2_000, key=key)                         # same key, same body
    assert r.status_code == 201
    assert balance_of(bob) == 2_000


def test_concurrent_duplicates_move_money_once():
    """10 simultaneous requests with the SAME key: money moves once, all get the same answer."""
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 10_000)
    key = str(uuid4())

    def send(_):
        return transfer(alice, bob, 1_000, key=key, http=TestClient(app))

    with ThreadPoolExecutor(max_workers=10) as pool:
        responses = list(pool.map(send, range(10)))

    assert all(r.status_code == 201 for r in responses)
    originals = [r for r in responses if r.headers.get("Idempotent-Replay") is None]
    assert len(originals) == 1                     # exactly one request did the work
    assert len({r.json()["transaction_id"] for r in responses}) == 1
    assert balance_of(alice) == 9_000
    assert balance_of(bob) == 1_000
