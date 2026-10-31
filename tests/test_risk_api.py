"""Day 7 tests: the risk engine inside POST /transfers. Needs the database running.

Rules only (see conftest.py). Each test uses fresh accounts, so tests never
influence each other through the engine's memory of a sender.
"""
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db.session import SessionLocal
from app.main import app

client = TestClient(app)


def at(second):
    """A simulated clock time, so velocity rules behave the same on every run."""
    return f"2026-09-01T10:00:{second:02d}+00:00"


def new_account(name):
    r = client.post("/accounts", json={"owner_name": name, "account_type": "user"})
    assert r.status_code == 201
    return r.json()["id"]


def fund(account_id, paise):
    r = client.post("/deposits", json={"account_id": account_id, "amount": paise})
    assert r.status_code == 201


def balance(account_id):
    return int(client.get(f"/accounts/{account_id}").json()["balance"])


def send(src, dst, paise, key=None, second=None):
    headers = {"Idempotency-Key": key or str(uuid4())}
    if second is not None:
        headers["X-Simulated-Time"] = at(second)
    return client.post(
        "/transfers",
        json={"from_account_id": src, "to_account_id": dst, "amount": paise},
        headers=headers,
    )


def burst(alice, bob):
    """Four payments one second apart: approved, approved, held, blocked."""
    return [send(alice, bob, 1_000, second=s) for s in range(4)]


def test_normal_payment_is_approved_and_carries_risk_fields():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 100_000)
    r = send(alice, bob, 1_000)
    body = r.json()
    assert r.status_code == 201
    assert body["decision"] == "approve"
    assert 0 <= body["risk_score"] <= 1
    assert body["reasons"] == []


def test_payment_over_the_hard_limit_is_blocked_and_no_money_moves():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 20_000_000)                      # Rs 200,000
    r = send(alice, bob, 15_000_000)             # Rs 150,000
    body = r.json()
    assert r.status_code == 403
    assert body["status"] == "blocked"
    assert body["decision"] == "block"
    assert any("hard limit" in reason for reason in body["reasons"])
    assert balance(alice) == 20_000_000
    assert balance(bob) == 0


def test_blocked_answer_is_replayed_for_the_same_key():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 20_000_000)
    key = str(uuid4())
    first = send(alice, bob, 15_000_000, key=key)
    second = send(alice, bob, 15_000_000, key=key)
    assert first.status_code == second.status_code == 403
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.json() == first.json()
    assert balance(alice) == 20_000_000


def test_velocity_burst_is_approved_then_held_then_blocked():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 1_000_000)
    statuses = [r.status_code for r in burst(alice, bob)]
    assert statuses == [201, 201, 202, 403]
    assert balance(alice) == 1_000_000 - 2_000   # only the two approved payments moved
    assert balance(bob) == 2_000


def test_held_payment_waits_in_the_queue_then_posts_when_approved():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 1_000_000)
    held = burst(alice, bob)[2].json()
    assert held["status"] == "pending_review"
    txn_id = held["transaction_id"]

    queue = client.get("/reviews").json()
    assert txn_id in [item["transaction_id"] for item in queue]
    assert balance(alice) == 998_000             # nothing moved while it waits

    approved = client.post(f"/reviews/{txn_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "posted"
    assert balance(alice) == 997_000
    assert balance(bob) == 3_000

    assert txn_id not in [item["transaction_id"] for item in client.get("/reviews").json()]
    assert client.post(f"/reviews/{txn_id}/approve").status_code == 409   # only once


def test_rejected_held_payment_moves_no_money():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 1_000_000)
    txn_id = burst(alice, bob)[2].json()["transaction_id"]

    rejected = client.post(f"/reviews/{txn_id}/reject")
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert balance(alice) == 998_000
    assert balance(bob) == 2_000
    assert client.post(f"/reviews/{txn_id}/reject").status_code == 409
    assert client.post(f"/reviews/{txn_id}/approve").status_code == 409


def test_unknown_held_payment_returns_404():
    r = client.post(f"/reviews/{uuid4()}/approve")
    assert r.status_code == 404


def test_every_decision_is_logged_including_blocked_ones():
    alice, bob = new_account("Alice"), new_account("Bob")
    fund(alice, 1_000_000)
    burst(alice, bob)
    with SessionLocal() as db:
        rows = db.execute(
            text("SELECT decision, transaction_id FROM risk_assessments "
                 "WHERE sender_id = :s ORDER BY id"),
            {"s": alice},
        ).all()
    assert [r[0] for r in rows] == ["approve", "approve", "review", "block"]
    assert rows[0][1] is not None and rows[2][1] is not None   # these have a transaction
    assert rows[3][1] is None                                  # blocked: no transaction


def test_health_reports_which_engine_is_running():
    body = client.get("/health").json()
    assert body == {"status": "ok", "risk_engine": "rules_only"}
