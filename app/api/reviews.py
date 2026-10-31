"""Day 7: the manual review queue for payments the risk engine held.

A held payment has status 'review' and has moved no money. A human either
approves it (the transfer is posted now) or rejects it (it never happens).
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.transfers import _commit_or_500, _post_transfer_uncommitted, get_balance
from app.db.session import get_db
from app.risk_service import get_engine

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("")
def list_reviews(db: Session = Depends(get_db)):
    """Payments waiting for a human decision, oldest first."""
    rows = db.execute(
        text(
            "SELECT t.id AS transaction_id, ra.sender_id, ra.recipient_id, ra.amount, "
            "       ra.risk_score, ra.reasons, ra.created_at "
            "FROM transactions t JOIN risk_assessments ra ON ra.transaction_id = t.id "
            "WHERE t.status = 'review' ORDER BY ra.created_at LIMIT 100"
        )
    ).mappings().all()
    return rows


def _load_held(db: Session, transaction_id: UUID):
    """Lock a held transaction and return what was asked for. 404 / 409 otherwise."""
    status = db.execute(
        text("SELECT status FROM transactions WHERE id = :id FOR UPDATE"),
        {"id": transaction_id},
    ).scalar()
    if status is None:
        raise HTTPException(404, "transaction not found")
    if status != "review":
        raise HTTPException(409, f"transaction is {status}, not waiting for review")
    return db.execute(
        text(
            "SELECT ra.sender_id, ra.recipient_id, ra.amount, ra.device_id, ra.created_at, "
            "       t.description "
            "FROM risk_assessments ra JOIN transactions t ON t.id = ra.transaction_id "
            "WHERE ra.transaction_id = :id"
        ),
        {"id": transaction_id},
    ).mappings().one()


@router.post("/{transaction_id}/approve")
def approve(transaction_id: UUID, db: Session = Depends(get_db)):
    try:
        held = _load_held(db, transaction_id)
        sender, recipient = UUID(str(held["sender_id"])), UUID(str(held["recipient_id"]))
        _post_transfer_uncommitted(
            db, sender, recipient, held["amount"], held["description"],
            existing_txn_id=transaction_id,
        )
    except HTTPException:
        db.rollback()  # e.g. the sender no longer has the money: it stays in the queue
        raise
    _commit_or_500(db)

    get_engine().trust({
        "timestamp": held["created_at"].isoformat(),
        "sender_id": str(sender),
        "recipient_id": str(recipient),
        "amount_paise": held["amount"],
        "device_id": held["device_id"],
    })
    return {
        "transaction_id": str(transaction_id),
        "status": "posted",
        "from_balance": get_balance(db, sender),
        "to_balance": get_balance(db, recipient),
    }


@router.post("/{transaction_id}/reject")
def reject(transaction_id: UUID, db: Session = Depends(get_db)):
    try:
        _load_held(db, transaction_id)
    except HTTPException:
        db.rollback()
        raise
    db.execute(
        text("UPDATE transactions SET status = 'rejected' WHERE id = :id"),
        {"id": transaction_id},
    )
    _commit_or_500(db)
    return {"transaction_id": str(transaction_id), "status": "rejected"}
