"""Day 1 tests: prove the database itself enforces double-entry rules.

Run with the database up:  docker compose up -d db  &&  pytest
"""
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.db.session import engine


@pytest.fixture()
def conn():
    with engine.connect() as c:
        yield c
        c.rollback()


def _make_account(conn, name):
    return conn.execute(
        text("INSERT INTO accounts (owner_name, account_type) VALUES (:n, 'user') RETURNING id"),
        {"n": name},
    ).scalar_one()


def _make_txn(conn):
    return conn.execute(text("INSERT INTO transactions DEFAULT VALUES RETURNING id")).scalar_one()


def _entry(conn, t, acc, direction, amount):
    conn.execute(
        text("INSERT INTO ledger_entries (transaction_id, account_id, direction, amount) "
             "VALUES (:t, :a, :d, :amt)"),
        {"t": t, "a": acc, "d": direction, "amt": amount},
    )


def test_balanced_transaction_commits(conn):
    a, b, t = _make_account(conn, "alice"), _make_account(conn, "bob"), _make_txn(conn)
    _entry(conn, t, a, "debit", 5000)
    _entry(conn, t, b, "credit", 5000)
    conn.commit()  # must not raise


def test_unbalanced_transaction_is_rejected(conn):
    a, t = _make_account(conn, "carol"), _make_txn(conn)
    _entry(conn, t, a, "debit", 5000)
    with pytest.raises(DBAPIError):
        conn.commit()


def test_entries_are_immutable(conn):
    a, b, t = _make_account(conn, "dave"), _make_account(conn, "erin"), _make_txn(conn)
    _entry(conn, t, a, "debit", 100)
    _entry(conn, t, b, "credit", 100)
    conn.commit()
    with pytest.raises(DBAPIError):
        conn.execute(text("UPDATE ledger_entries SET amount = 999 WHERE transaction_id = :t"), {"t": t})
