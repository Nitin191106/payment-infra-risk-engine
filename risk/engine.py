"""Day 6: the hybrid risk engine.

Combines fixed RULES (clear, auditable, instant) with the learned MODEL
(catches odd behaviour nobody wrote a rule for) and returns one decision:

    approve  - let it through
    review   - let a human look at it
    block    - refuse it

with plain-English reasons. Rules and model each have the final say when they
are confident: any block rule blocks; otherwise the model's percentile decides.

    engine = RiskEngine.load("models/risk_model.joblib")
    decision = engine.assess(transaction)     # judge a payment
    engine.record(transaction, decision)      # then remember it

Blocked payments are NOT remembered as normal behaviour, so a fraudster can't
"teach" the engine that a new device or recipient is fine.
"""
import threading
from dataclasses import asdict, dataclass, field

import numpy as np

from risk.features import FeatureState, to_vector


@dataclass(frozen=True)
class RiskConfig:
    hard_limit_rupees: float = 100_000    # at or above: block
    velocity_review: int = 2              # earlier payments in the last minute -> review
    velocity_block: int = 3               # ... -> block (the 4th payment within a minute)
    structuring_low: float = 45_000       # round payments between these two amounts,
    structuring_high: float = 50_000      # repeated within an hour, look like structuring
    spike_ratio: float = 8.0              # this many times the sender's usual amount
    review_percentile: float = 0.97       # model: more unusual than 97% of normal traffic
    block_percentile: float = 0.995       # model: more unusual than 99.5%


@dataclass
class Decision:
    decision: str                         # approve / review / block
    risk_score: float                     # 0 to 1, higher = riskier
    reasons: list = field(default_factory=list)
    rules: list = field(default_factory=list)   # codes of the rules that fired
    model_percentile: float = 0.0         # how unusual the model thinks it is, 0 to 1

    def to_dict(self):
        return asdict(self)


def evaluate_rules(f, cfg):
    """Return [(action, code, reason)] for every fixed rule that fires on features `f`."""
    fired = []
    amount = _rupees(f)

    if amount >= cfg.hard_limit_rupees:
        fired.append(("block", "hard_limit",
                      f"amount Rs {amount:,.0f} is above the hard limit of Rs {cfg.hard_limit_rupees:,.0f}"))

    prior = int(f["count_1min"])
    if prior >= cfg.velocity_block:
        fired.append(("block", "velocity",
                      f"{prior + 1} payments within a minute"))
    elif prior >= cfg.velocity_review:
        fired.append(("review", "velocity",
                      f"{prior + 1} payments within a minute"))

    if (cfg.structuring_low <= amount < cfg.structuring_high
            and f["round_amount"] == 1 and f["count_1h"] >= 1):
        action = "block" if f["count_1h"] >= 2 else "review"
        fired.append((action, "structuring",
                      f"repeated round payments just under Rs {cfg.structuring_high:,.0f}"))

    if f["amount_ratio"] >= cfg.spike_ratio:
        if f["new_device"] and f["new_recipient"]:
            fired.append(("block", "takeover_pattern",
                          f"amount is {f['amount_ratio']:.0f}x this sender's usual, "
                          "from a new device to a new recipient"))
        elif f["new_device"] or f["new_recipient"]:
            what = "new device" if f["new_device"] else "new recipient"
            fired.append(("review", "takeover_pattern",
                          f"amount is {f['amount_ratio']:.0f}x this sender's usual, with a {what}"))
    return fired


def _rupees(f):
    from math import exp
    return exp(f["log_amount"])


class RiskEngine:
    def __init__(self, bundle=None, config=None):
        self.config = config or RiskConfig()
        self.state = FeatureState()
        self._lock = threading.Lock()  # the API serves requests on several threads
        self.model = bundle["model"] if bundle else None
        self.quantiles = bundle["score_quantiles"] if bundle else None

    @classmethod
    def load(cls, path, config=None):
        import joblib
        return cls(joblib.load(path), config)

    def _model_percentile(self, features):
        if self.model is None:
            return 0.0
        score = -self.model.score_samples([to_vector(features)])[0]
        rank = np.searchsorted(self.quantiles, score, side="right")
        return float(min(1.0, rank / (len(self.quantiles) - 1)))

    def assess(self, row):
        cfg = self.config
        with self._lock:
            f = self.state.compute(row)
        pct = self._model_percentile(f)  # slow part, done outside the lock
        fired = evaluate_rules(f, cfg)

        if any(a == "block" for a, _, _ in fired) or pct >= cfg.block_percentile:
            decision = "block"
        elif fired or pct >= cfg.review_percentile:
            decision = "review"
        else:
            decision = "approve"

        reasons = [reason for _, _, reason in fired]
        if decision != "approve":
            if pct >= cfg.review_percentile:
                reasons.append(f"behaviour is more unusual than {pct:.1%} of normal traffic")
            reasons += self._observations(f, reasons)

        floor = 0.99 if any(a == "block" for a, _, _ in fired) else 0.9 if fired else 0.0
        return Decision(
            decision=decision,
            risk_score=round(max(pct, floor), 4),
            reasons=reasons[:5],
            rules=[code for _, code, _ in fired],
            model_percentile=round(pct, 4),
        )

    @staticmethod
    def _observations(f, existing):
        """Plain facts about the payment that help a human reviewer."""
        text = " ".join(existing)
        facts = []
        if f["amount_ratio"] >= 5 and "usual" not in text:
            facts.append(f"amount is {f['amount_ratio']:.0f}x this sender's usual")
        if f["new_device"] and "new device" not in text:
            facts.append("first time this sender uses this device")
        if f["new_recipient"] and "new recipient" not in text:
            facts.append("first payment from this sender to this recipient")
        if f["count_1min"] >= 1 and "within a minute" not in text:
            facts.append(f"{int(f['count_1min']) + 1} payments within a minute")
        if f["hour"] <= 4:
            facts.append(f"unusual hour ({int(f['hour']):02d}:00)")
        return facts

    def record(self, row, decision):
        """Remember a payment. Blocked ones are not trusted as normal behaviour."""
        if decision.decision != "block":
            with self._lock:
                self.state.update(row)

    def record_attempt(self, row):
        """A payment held for review: it counts toward velocity but is not yet trusted."""
        with self._lock:
            self.state.update(row, trusted=False)

    def trust(self, row):
        """A held payment was approved by a human: now it is normal behaviour.
        Its attempt was already counted, so it is not counted twice."""
        with self._lock:
            self.state.update(row, trusted=True, attempt=False)
