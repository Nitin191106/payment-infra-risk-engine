-- Day 7: audit log of every risk decision.
-- One row per payment judged by the risk engine, including blocked ones
-- (which never become a ledger transaction, so transaction_id is NULL for them).
CREATE TABLE IF NOT EXISTS risk_assessments (
    id               BIGSERIAL PRIMARY KEY,
    transaction_id   UUID REFERENCES transactions (id),
    sender_id        UUID        NOT NULL REFERENCES accounts (id),
    recipient_id     UUID        NOT NULL REFERENCES accounts (id),
    amount           BIGINT      NOT NULL CHECK (amount > 0),
    device_id        TEXT        NOT NULL,
    decision         TEXT        NOT NULL CHECK (decision IN ('approve', 'review', 'block')),
    risk_score       NUMERIC(5, 4) NOT NULL,
    model_percentile NUMERIC(5, 4) NOT NULL,
    rules            JSONB       NOT NULL DEFAULT '[]',
    reasons          JSONB       NOT NULL DEFAULT '[]',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_risk_decision ON risk_assessments (decision, created_at);
CREATE INDEX IF NOT EXISTS idx_risk_txn      ON risk_assessments (transaction_id);
