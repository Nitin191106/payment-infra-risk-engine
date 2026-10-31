"""Day 3: transfers with idempotency keys.

Idempotency idea: the client sends an `Idempotency-Key` header. If the same key
arrives again (retry, double click, network glitch) we return the ORIGINAL
response instead of moving money a second time.

How it stays safe: the key row, the ledger entries and the saved response are
all written in ONE database transaction. A duplicate request that arrives while
the first is still running blocks on the key's unique index until the first one
commits, then replays its result. No "in progress" state is needed.
"""
import hashlib
import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter(tags=["transfers"])


class TransferCreate(BaseModel):
    from_account_id: UUID
    to_account_id: UUID
    amount: int = Field(gt=0, description="Amount in paise (5000 = Rs 50.00)")
    description: str | None = None


class DepositCreate(BaseModel):
    account_id: UUID
    amount: int = Field(gt=0, description="Amount in paise")


def get_balance(db: Session, account_id: UUID) -> int:
    value = db.execute(
        text(
            "SELECT COALESCE(SUM(CASE WHEN direction = 'credit' "
            "THEN amount ELSE -amount END), 0) "
            "FROM ledger_entries WHERE account_id = :id"
        ),
        {"id": account_id},
    ).scalar_one()
    return int(value)


def _post_transfer_uncommitted(
    db: Session,
    from_id: UUID,
    to_id: UUID,
    amount: int,
    description: str | None,
    allow_overdraft: bool = False,
) -> UUID:
    """Lock, check funds and insert the debit + credit. Does NOT commit."""
    if from_id == to_id:
        raise HTTPException(422, "from_account_id and to_account_id must differ")

    # Lock both accounts in a fixed order so opposite transfers can't deadlock.
    accounts = {}
    for acc_id in sorted([from_id, to_id]):
        row = db.execute(
            text("SELECT id, account_type, currency FROM accounts WHERE id = :id FOR UPDATE"),
            {"id": acc_id},
        ).mappings().first()
        if row is None:
            raise HTTPException(404, f"account {acc_id} not found")
        accounts[acc_id] = row

    if accounts[from_id]["currency"] != accounts[to_id]["currency"]:
        raise HTTPException(422, "currency mismatch between accounts")

    # Check funds AFTER locking.
    if not allow_overdraft:
        balance = get_balance(db, from_id)
        if balance < amount:
            raise HTTPException(409, f"insufficient funds: balance {balance}, requested {amount}")

    txn_id = db.execute(
        text(
            "INSERT INTO transactions (status, description) "
            "VALUES ('posted', :d) RETURNING id"
        ),
        {"d": description},
    ).scalar_one()

    for acc_id, direction in ((from_id, "debit"), (to_id, "credit")):
        db.execute(
            text(
                "INSERT INTO ledger_entries (transaction_id, account_id, direction, amount) "
                "VALUES (:t, :a, :d, :amt)"
            ),
            {"t": txn_id, "a": acc_id, "d": direction, "amt": amount},
        )
    return txn_id


def _commit_or_500(db: Session) -> None:
    """Commit. The deferred trigger checks debits == credits right here."""
    try:
        db.commit()
    except DBAPIError:
        db.rollback()
        raise HTTPException(500, "ledger rejected the transaction")


def post_transfer(
    db: Session,
    from_id: UUID,
    to_id: UUID,
    amount: int,
    description: str | None,
    allow_overdraft: bool = False,
) -> UUID:
    """Move money atomically and commit. Returns the transaction id."""
    txn_id = _post_transfer_uncommitted(db, from_id, to_id, amount, description, allow_overdraft)
    _commit_or_500(db)
    return txn_id


def _hash_request(body: BaseModel) -> str:
    canonical = json.dumps(body.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


@router.post("/transfers", status_code=201)
def create_transfer(
    body: TransferCreate,
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=8, max_length=255)
    ],
    db: Session = Depends(get_db),
):
    request_hash = _hash_request(body)

    # 1. Try to claim the key. If another request holds it uncommitted, this
    #    statement WAITS until that request commits or rolls back.
    claimed = db.execute(
        text(
            "INSERT INTO idempotency_keys (key, request_hash) VALUES (:k, :h) "
            "ON CONFLICT (key) DO NOTHING RETURNING key"
        ),
        {"k": idempotency_key, "h": request_hash},
    ).first()

    # 2. Key already used: replay the stored response (or reject a mismatch).
    if claimed is None:
        row = db.execute(
            text(
                "SELECT request_hash, response_status, response_body "
                "FROM idempotency_keys WHERE key = :k"
            ),
            {"k": idempotency_key},
        ).mappings().first()
        db.rollback()
        if row is None or row["response_status"] is None:
            raise HTTPException(409, "request with this Idempotency-Key is still being processed; retry shortly")
        if row["request_hash"] != request_hash:
            raise HTTPException(422, "Idempotency-Key was already used with a different request")
        return JSONResponse(
            status_code=row["response_status"],
            content=row["response_body"],
            headers={"Idempotent-Replay": "true"},
        )

    # 3. First time we see this key: do the transfer and save the response,
    #    all inside one transaction.
    try:
        txn_id = _post_transfer_uncommitted(
            db, body.from_account_id, body.to_account_id, body.amount, body.description
        )
        response = {
            "transaction_id": str(txn_id),
            "status": "posted",
            "from_balance": get_balance(db, body.from_account_id),
            "to_balance": get_balance(db, body.to_account_id),
        }
        db.execute(
            text(
                "UPDATE idempotency_keys SET response_status = 201, "
                "response_body = CAST(:b AS jsonb) WHERE key = :k"
            ),
            {"b": json.dumps(response), "k": idempotency_key},
        )
    except HTTPException:
        # Failed requests are NOT remembered: rolling back also removes the key,
        # so the client can fix the problem and retry with the same key.
        db.rollback()
        raise

    _commit_or_500(db)
    return JSONResponse(status_code=201, content=response)


def get_or_create_treasury(db: Session, currency: str) -> UUID:
    row = db.execute(
        text(
            "SELECT id FROM accounts WHERE account_type = 'system' "
            "AND owner_name = 'Treasury' AND currency = :c ORDER BY created_at LIMIT 1"
        ),
        {"c": currency},
    ).first()
    if row:
        return row[0]
    new_id = db.execute(
        text(
            "INSERT INTO accounts (owner_name, account_type, currency) "
            "VALUES ('Treasury', 'system', :c) RETURNING id"
        ),
        {"c": currency},
    ).scalar_one()
    db.commit()
    return new_id


@router.post("/deposits", status_code=201)
def create_deposit(body: DepositCreate, db: Session = Depends(get_db)):
    """Test-money faucet: moves money from a system Treasury account (allowed to go negative)."""
    acc = db.execute(
        text("SELECT currency FROM accounts WHERE id = :id"), {"id": body.account_id}
    ).first()
    if acc is None:
        raise HTTPException(404, "account not found")
    treasury_id = get_or_create_treasury(db, acc[0])
    txn_id = post_transfer(
        db, treasury_id, body.account_id, body.amount, "deposit", allow_overdraft=True
    )
    return {
        "transaction_id": txn_id,
        "status": "posted",
        "balance": get_balance(db, body.account_id),
    }


@router.get("/accounts/{account_id}/entries")
def list_entries(account_id: UUID, db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            "SELECT id, transaction_id, direction, amount, created_at "
            "FROM ledger_entries WHERE account_id = :id ORDER BY id DESC LIMIT 100"
        ),
        {"id": account_id},
    ).mappings().all()
    return rows
