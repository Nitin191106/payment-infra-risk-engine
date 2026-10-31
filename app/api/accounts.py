from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter(prefix="/accounts", tags=["accounts"])


class AccountCreate(BaseModel):
    owner_name: str = Field(min_length=1)
    account_type: str = Field(pattern="^(user|merchant|system)$")
    currency: str = Field(default="INR", min_length=3, max_length=3)


@router.post("", status_code=201)
def create_account(body: AccountCreate, db: Session = Depends(get_db)):
    row = db.execute(
        text(
            "INSERT INTO accounts (owner_name, account_type, currency) "
            "VALUES (:o, :t, :c) RETURNING id, owner_name, account_type, currency, created_at"
        ),
        {"o": body.owner_name, "t": body.account_type, "c": body.currency},
    ).mappings().one()
    db.commit()
    return {**row, "balance": 0}


@router.get("/{account_id}")
def get_account(account_id: UUID, db: Session = Depends(get_db)):
    row = db.execute(
        text(
            "SELECT a.id, a.owner_name, a.account_type, a.currency, a.created_at, "
            "       b.balance "
            "FROM accounts a JOIN account_balances b ON b.account_id = a.id "
            "WHERE a.id = :id"
        ),
        {"id": account_id},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="account not found")
    return row
