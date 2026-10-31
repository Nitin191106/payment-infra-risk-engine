-- Day 3: idempotency keys
-- One row per Idempotency-Key. The row is created in the SAME database
-- transaction as the transfer, so it exists if and only if the money moved.
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key             TEXT        PRIMARY KEY,
    request_hash    TEXT        NOT NULL,
    response_status INT,
    response_body   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
