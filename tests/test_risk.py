"""Day 5 tests: feature engineering (no database needed) and the model."""
import pytest

from risk.features import FEATURE_NAMES, build_features
from simulator.generate_dataset import generate


def tx(ts, sender="u1", recipient="m1", rupees=1000, device="d1"):
    return {
        "timestamp": f"2026-09-01T{ts}+00:00",
        "sender_id": sender,
        "recipient_id": recipient,
        "amount_paise": rupees * 100,
        "device_id": device,
    }


def col(vectors, name):
    return [v[FEATURE_NAMES.index(name)] for v in vectors]


def test_vector_has_one_number_per_feature():
    vectors = build_features([tx("10:00:00")])
    assert len(vectors[0]) == len(FEATURE_NAMES)


def test_first_payments_are_not_called_new():
    """Nothing is known about a brand-new sender, so no 'new recipient' alarm."""
    vectors = build_features([tx("10:00:00"), tx("11:00:00", recipient="m2")])
    assert col(vectors, "new_recipient") == [0.0, 0.0]


def test_new_recipient_and_device_detected_after_warmup():
    rows = [tx(f"1{i}:00:00") for i in range(3)]               # 3 normal payments
    rows.append(tx("14:00:00", recipient="m9"))                # new recipient
    rows.append(tx("15:00:00", device="dev_new"))              # new device
    rows.append(tx("16:00:00"))                                # all familiar again
    vectors = build_features(rows)
    assert col(vectors, "new_recipient")[3:] == [1.0, 0.0, 0.0]
    assert col(vectors, "new_device")[3:] == [0.0, 1.0, 0.0]


def test_velocity_counts_recent_payments_only():
    rows = [tx(f"10:00:0{i}") for i in range(5)]               # 5 payments in 5 seconds
    rows.append(tx("10:00:30"))                                # 6th, 30 s later
    rows.append(tx("12:30:00"))                                # much later
    vectors = build_features(rows)
    assert col(vectors, "count_1min")[5] == 5.0
    assert col(vectors, "count_1h")[5] == 5.0
    assert col(vectors, "count_1min")[6] == 0.0
    assert col(vectors, "count_1h")[6] == 0.0


def test_amount_ratio_compares_with_senders_own_history():
    rows = [tx(f"1{i}:00:00", rupees=1000) for i in range(3)]
    rows.append(tx("14:00:00", rupees=5000))
    ratios = col(build_features(rows), "amount_ratio")
    assert ratios[0] == 1.0          # first payment: nothing to compare with
    assert ratios[3] == 5.0          # five times this sender's usual


def test_senders_do_not_affect_each_other():
    alone = build_features([tx("10:00:00", sender="u1"), tx("10:00:05", sender="u1")])
    mixed = build_features([
        tx("10:00:00", sender="u1"), tx("10:00:02", sender="u2"),
        tx("10:00:03", sender="u2"), tx("10:00:05", sender="u1"),
    ])
    assert mixed[3] == alone[1]


def test_features_never_use_the_future():
    """Features for the first 200 rows must be identical whether or not later rows exist."""
    rows = generate(users=50, days=10, seed=3)
    assert build_features(rows[:200]) == build_features(rows)[:200]


def test_model_ranks_fraud_above_normal_traffic():
    pytest.importorskip("sklearn")
    from sklearn.metrics import roc_auc_score
    from risk.train import train

    rows = generate(users=300, days=30, seed=42)
    bundle, scores, y, _ = train(rows)
    threshold = bundle["threshold"]
    assert roc_auc_score(y, scores) > 0.9
    flagged = scores >= threshold
    assert flagged[y == 1].mean() > 0.6          # catches most fraud
    assert flagged.mean() < 0.10                 # without flagging everything
