from fastapi import FastAPI
from sqlalchemy import text

from app.api import accounts, transfers
from app.db.session import engine

app = FastAPI(title="Payment Infrastructure & AI Risk Engine")
app.include_router(accounts.router)
app.include_router(transfers.router)


@app.get("/health")
def health():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}
