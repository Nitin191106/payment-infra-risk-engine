-- Day 1: double-entry ledger schema
-- Convention: balance = SUM(credits) - SUM(debits). Amounts are integers in the
-- smallest currency unit (paise for INR). Never use floats for money.

CREATE TABLE accounts (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_name   TEXT        NOT NULL,
    account_type TEXT        NOT NULL CHECK (account_type IN ('user', 'merchant', 'system')),
    currency     CHAR(3)     NOT NULL DEFAULT 'INR',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE transactions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status        TEXT        NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'posted', 'rejected', 'review')),
    description   TEXT,
    risk_score    NUMERIC(5, 4),          -- filled in by the risk engine later
    risk_decision TEXT,                   -- approve / review / block
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ledger_entries (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id UUID        NOT NULL REFERENCES transactions (id),
    account_id     UUID        NOT NULL REFERENCES accounts (id),
    direction      TEXT        NOT NULL CHECK (direction IN ('debit', 'credit')),
    amount         BIGINT      NOT NULL CHECK (amount > 0),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_entries_account ON ledger_entries (account_id);
CREATE INDEX idx_entries_txn     ON ledger_entries (transaction_id);

-- Rule 1: entries are immutable. Fix mistakes with a reversing transaction.
CREATE FUNCTION forbid_entry_changes() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'ledger_entries are immutable; post a reversing transaction instead';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ledger_entries_immutable
    BEFORE UPDATE OR DELETE ON ledger_entries
    FOR EACH ROW EXECUTE FUNCTION forbid_entry_changes();

-- Rule 2: every transaction must balance (debits = credits).
-- Deferred, so it is checked at COMMIT after all entries are inserted.
CREATE FUNCTION check_transaction_balanced() RETURNS trigger AS $$
DECLARE
    debits  BIGINT;
    credits BIGINT;
BEGIN
    SELECT COALESCE(SUM(amount) FILTER (WHERE direction = 'debit'), 0),
           COALESCE(SUM(amount) FILTER (WHERE direction = 'credit'), 0)
      INTO debits, credits
      FROM ledger_entries
     WHERE transaction_id = NEW.transaction_id;

    IF debits <> credits THEN
        RAISE EXCEPTION 'transaction % is unbalanced: debits=% credits=%',
            NEW.transaction_id, debits, credits;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE CONSTRAINT TRIGGER ledger_entries_balanced
    AFTER INSERT ON ledger_entries
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION check_transaction_balanced();

-- Convenience view for balances
CREATE VIEW account_balances AS
SELECT a.id AS account_id,
       a.owner_name,
       a.currency,
       COALESCE(SUM(CASE WHEN e.direction = 'credit' THEN e.amount ELSE -e.amount END), 0) AS balance
  FROM accounts a
  LEFT JOIN ledger_entries e ON e.account_id = a.id
 GROUP BY a.id;
