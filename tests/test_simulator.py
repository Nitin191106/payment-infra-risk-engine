"""Day 4 tests for the dataset generator. No database needed."""
from simulator.generate_dataset import FIELDS, generate


def test_same_seed_gives_identical_data():
    assert generate(users=50, days=7, seed=7) == generate(users=50, days=7, seed=7)


def test_different_seed_gives_different_data():
    assert generate(users=50, days=7, seed=1) != generate(users=50, days=7, seed=2)


def test_rows_have_all_columns_and_valid_values():
    rows = generate(users=50, days=7)
    for r in rows:
        assert set(FIELDS) == set(r)
        assert isinstance(r["amount_paise"], int) and r["amount_paise"] > 0
        assert r["sender_id"] != r["recipient_id"]
        assert r["is_fraud"] in (0, 1)
        assert (r["is_fraud"] == 1) == bool(r["fraud_type"])


def test_rows_are_sorted_by_time_with_unique_ids():
    rows = generate(users=50, days=7)
    stamps = [r["timestamp"] for r in rows]
    assert stamps == sorted(stamps)
    assert len({r["txn_id"] for r in rows}) == len(rows)


def test_fraud_rate_is_close_to_requested():
    rows = generate(users=300, days=30, fraud_rate=0.02)
    rate = sum(r["is_fraud"] for r in rows) / len(rows)
    assert 0.015 <= rate <= 0.035


def test_all_three_fraud_types_appear():
    rows = generate(users=300, days=30)
    kinds = {r["fraud_type"] for r in rows if r["is_fraud"]}
    assert kinds == {"account_takeover", "velocity_burst", "structuring"}


def test_account_takeover_looks_different_from_normal_traffic():
    rows = generate(users=300, days=30)
    normal = sorted(r["amount_paise"] for r in rows if not r["is_fraud"])
    takeover = sorted(r["amount_paise"] for r in rows if r["fraud_type"] == "account_takeover")
    assert takeover[len(takeover) // 2] > normal[len(normal) // 2]
