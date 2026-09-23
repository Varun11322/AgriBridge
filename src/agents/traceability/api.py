import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import get_traceability_model

app = FastAPI(
    title="AgriBridge Traceability Agent API",
    description="Verification agent scoring produce batch custody chain integrity",
    version="1.0.0"
)

DATA_DIR = os.path.join(os.getcwd(), "data")
BATCHES_JSON_PATH = os.path.join(DATA_DIR, "synthetic_batches.json")

# In-memory batch database loaded from dataset
_BATCH_DB = {}

def _load_batch_db():
    global _BATCH_DB
    if os.path.exists(BATCHES_JSON_PATH):
        try:
            with open(BATCHES_JSON_PATH, "r", encoding="utf-8") as f:
                events = json.load(f)
            db = {}
            for e in events:
                bid = e["batch_id"]
                if bid not in db:
                    db[bid] = []
                db[bid].append(e)
            _BATCH_DB = db
            print(f"Loaded {len(_BATCH_DB)} batches into Traceability Agent API memory.")
        except Exception as err:
            print(f"Error loading batch DB: {err}")

# Load DB on startup
_load_batch_db()

class EventItem(BaseModel):
    batch_id: str
    event_type: str
    actor_id: str
    actor_role: str
    timestamp: str
    prev_hash: Optional[str] = ""
    location: Optional[str] = ""
    lat: Optional[float] = 0.0
    lon: Optional[float] = 0.0
    tx_hash: Optional[str] = ""

class VerifyBatchRequest(BaseModel):
    events: Optional[List[EventItem]] = None

class VerifyBatchResponse(BaseModel):
    batch_id: str
    status: str
    confidence_score: int
    chain_length: int
    issues: List[str]

@app.get("/health")
def health_check():
    return {"status": "ok", "agent": "Traceability Agent", "batches_indexed": len(_BATCH_DB)}

@app.post("/verify-batch/{batch_id}", response_model=VerifyBatchResponse)
def verify_batch(batch_id: str, body: Optional[VerifyBatchRequest] = None):
    """
    Verifies the custody chain integrity for a given produce batch ID.
    Returns status ('VERIFIED' | 'FLAGGED' | 'INCOMPLETE'), confidence_score (0-100), chain_length, and issues.
    """
    model = get_traceability_model()
    
    events = None
    if body and body.events:
        events = [e.model_dump() for e in body.events]
    elif batch_id in _BATCH_DB:
        events = _BATCH_DB[batch_id]
    elif batch_id.startswith("AG-2"): # Fallback lookup in DB
        events = _BATCH_DB.get(batch_id)
        
    if not events:
        # Return fallback INCOMPLETE record for unknown batch_id
        return VerifyBatchResponse(
            batch_id=batch_id,
            status="INCOMPLETE",
            confidence_score=0,
            chain_length=0,
            issues=[f"Batch ID '{batch_id}' not found in blockchain event index"]
        )
        
    res = model.evaluate_batch(batch_id, events)
    return VerifyBatchResponse(**res)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
