"""Day 5: turn a raw transaction into numbers a model can learn from.

Every feature is computed from the sender's PAST activity only, so the model
never "sees the future". The same FeatureState class can run on a live stream
later: call compute() to get features for a new transaction, then update()
once it is done.

Transactions must arrive in time order. Standard library only.
"""
from collections import defaultdict, deque
from datetime import datetime

FEATURE_NAMES = [
    "log_amount",      # natural log of the amount in rupees
    "amount_ratio",    # amount divided by this sender's average so far (1.0 = typical)
    "count_1min",      # sender's transfers in the previous 60 seconds
    "count_1h",        # sender's transfers in the previous hour
    "new_recipient",   # 1 if the sender never paid this recipient before
    "new_device",      # 1 if the sender never used this device before
    "hour",            # hour of day, 0-23
    "round_amount",    # 1 if the amount is a multiple of Rs 100
]

WARMUP = 3  # a sender's first few transfers are not judged "new" (we know nothing yet)


class FeatureState:
    """Remembers each sender's history so features use only the past."""

    def __init__(self):
        self.count = defaultdict(int)
        self.amount_sum = defaultdict(float)
        self.recipients = defaultdict(set)
        self.devices = defaultdict(set)
        self.recent = defaultdict(deque)  # timestamps (seconds) within the last hour

    def compute(self, row):
        sender = row["sender_id"]
        when = datetime.fromisoformat(row["timestamp"])
        t = when.timestamp()
        amount = int(row["amount_paise"]) / 100

        seen = self.count[sender]
        average = self.amount_sum[sender] / seen if seen else 0.0
        ratio = amount / average if average else 1.0

        recent = self.recent[sender]
        while recent and t - recent[0] > 3600:
            recent.popleft()
        count_1h = len(recent)
        count_1min = sum(1 for x in recent if t - x <= 60)

        warmed_up = seen >= WARMUP
        return {
            "log_amount": _log(amount),
            "amount_ratio": ratio,
            "count_1min": float(count_1min),
            "count_1h": float(count_1h),
            "new_recipient": float(warmed_up and row["recipient_id"] not in self.recipients[sender]),
            "new_device": float(warmed_up and row["device_id"] not in self.devices[sender]),
            "hour": float(when.hour),
            "round_amount": float(amount % 100 == 0),
        }

    def update(self, row):
        sender = row["sender_id"]
        self.count[sender] += 1
        self.amount_sum[sender] += int(row["amount_paise"]) / 100
        self.recipients[sender].add(row["recipient_id"])
        self.devices[sender].add(row["device_id"])
        self.recent[sender].append(datetime.fromisoformat(row["timestamp"]).timestamp())


def _log(x):
    from math import log
    return log(max(x, 1.0))


def to_vector(features):
    return [features[name] for name in FEATURE_NAMES]


def build_features(rows):
    """Feature vectors for time-ordered rows, one list of numbers per row."""
    state = FeatureState()
    vectors = []
    for row in rows:
        vectors.append(to_vector(state.compute(row)))
        state.update(row)
    return vectors
