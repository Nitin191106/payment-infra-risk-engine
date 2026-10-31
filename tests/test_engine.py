"""Day 6 tests: the hybrid risk engine. Rules-only tests need no trained model."""
import json

import pytest

from risk.engine import RiskEngine


def tx(ts, rupees=1000, sender="u1", recipient="m1", device="d1"):
    return {
        "timestamp": f"2026-09-01T{ts}+00:00",
        "sender_id": sender,
        "recipient_id": recipient,
        "amount_paise": int(rupees * 100),
        "device_id": device,
    }


def judge(engine, row):
    """Assess a payment and remember it, like live traffic."""
    decision = engine.assess(row)
    engine.record(row, decision)
    return decision


def warmed_up_engine():
    """A rules-only engine that has seen three ordinary payments from u1."""
    engine = RiskEngine()
    for hour in (9, 10, 11):
        judge(engine, tx(f"{hour:02d}:00:00"))
    return engine


def test_ordinary_payment_is_approved_with_no_reasons():
    engine = warmed_up_engine()
    d = engine.assess(tx("12:00:00", rupees=1100))
    assert d.decision == "approve"
    assert d.reasons == []
    assert d.rules == []


def test_hard_limit_blocks():
    d = RiskEngine().assess(tx("10:00:00", rupees=150_000))
    assert d.decision == "block"
    assert d.rules == ["hard_limit"]
    assert "hard limit" in d.reasons[0]


def test_third_payment_in_a_minute_is_reviewed_and_fourth_is_blocked():
    engine = RiskEngine()
    decisions = [judge(engine, tx(f"10:00:0{i}")).decision for i in range(4)]
    assert decisions == ["approve", "approve", "review", "block"]


def test_big_spike_on_new_device_and_new_recipient_is_blocked():
    engine = warmed_up_engine()
    d = engine.assess(tx("03:00:00", rupees=20_000, recipient="mule1", device="dev_new"))
    assert d.decision == "block"
    assert "takeover_pattern" in d.rules
    assert "new device" in " ".join(d.reasons)


def test_big_spike_with_only_a_new_recipient_goes_to_review():
    engine = warmed_up_engine()
    d = engine.assess(tx("12:00:00", rupees=20_000, recipient="mule1"))
    assert d.decision == "review"
    assert "new recipient" in " ".join(d.reasons)


def test_repeated_round_payments_just_under_the_limit_escalate():
    engine = RiskEngine()
    decisions = [judge(engine, tx(f"10:{m}:00", rupees=49_500, recipient="mule1")).decision
                 for m in ("00", "10", "20")]
    assert decisions == ["approve", "review", "block"]


def test_blocked_payments_are_not_remembered_as_normal():
    engine = warmed_up_engine()
    attack = tx("03:00:00", rupees=20_000, recipient="mule1", device="dev_new")
    assert judge(engine, attack).decision == "block"
    # The same attack later is still a new device and new recipient.
    again = engine.assess(tx("15:00:00", rupees=20_000, recipient="mule1", device="dev_new"))
    assert again.decision == "block"
    assert "takeover_pattern" in again.rules


def test_decision_is_json_friendly_and_score_orders_sensibly():
    engine = warmed_up_engine()
    calm = engine.assess(tx("12:00:00"))
    risky = engine.assess(tx("03:00:00", rupees=20_000, recipient="mule1", device="dev_new"))
    assert json.loads(json.dumps(risky.to_dict()))["decision"] == "block"
    assert 0 <= calm.risk_score < risky.risk_score <= 1


def test_hybrid_engine_catches_most_fraud_and_rarely_blocks_normal_payments():
    pytest.importorskip("sklearn")
    from risk.evaluate import run_stream
    from risk.train import train
    from simulator.generate_dataset import generate

    rows = generate(users=300, days=30, seed=42)
    bundle, *_ = train(rows)
    decisions = run_stream(rows, RiskEngine(bundle))

    cut = int(len(rows) * 0.7)
    fraud = [d for r, d in zip(rows[cut:], decisions[cut:]) if r["is_fraud"]]
    normal = [d for r, d in zip(rows[cut:], decisions[cut:]) if not r["is_fraud"]]
    assert sum(d.decision != "approve" for d in fraud) / len(fraud) > 0.8
    assert sum(d.decision == "block" for d in normal) / len(normal) < 0.01
    assert sum(d.decision == "approve" for d in normal) / len(normal) > 0.95


def test_held_payment_counts_toward_velocity_but_is_not_trusted():
    engine = warmed_up_engine()
    engine.record_attempt(tx("12:00:00", device="dev_new"))
    f = engine.state.compute(tx("12:00:10", device="dev_new"))
    assert f["count_1min"] == 1.0      # the held payment counts as recent activity
    assert f["new_device"] == 1.0      # but the device is still not trusted


def test_approved_held_payment_becomes_trusted_without_double_counting():
    engine = warmed_up_engine()
    held = tx("12:00:00", device="dev_new")
    engine.record_attempt(held)
    engine.trust(held)
    f = engine.state.compute(tx("12:00:10", device="dev_new"))
    assert f["new_device"] == 0.0      # now a known device
    assert f["count_1min"] == 1.0      # still one attempt, not two
