
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd

# Local imports
from models.content import score_content
from models.hybrid import score_hybrid
from models.cf import score_cf
from fair.rerank import rerank_head_tail
from models.popularity import score_popularity

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

app = FastAPI(title="MVP Recommender API", version="0.1.0")

# UI directory (simple static HTML UI)
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")

# Serve /ui as static files (index.html inside)
if os.path.isdir(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

@app.get("/")
def root():
    # redirect root to the UI homepage if present, else just return a basic health
    if os.path.isdir(UI_DIR):
        return RedirectResponse(url="/ui/")
    return {"ok": True}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class Item(BaseModel):
    provider_id: str
    display_name: str
    score: float
    rationale: List[str] = []

class TopNResponse(BaseModel):
    mode: str
    fair: int
    user_id: Optional[str] = None
    items: List[Item]

def _load_data():
    users = providers = inter = None
    try:
        users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
        providers = pd.read_csv(os.path.join(DATA_DIR, "providers.csv"))
        inter = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    except Exception:
        pass
    return users, providers, inter

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/topN", response_model=TopNResponse)
def topN(
    mode: str = Query("content", pattern="^(pop|content|cf|hybrid)$"),
    fair: int = Query(0, ge=0, le=1),
    k: int = Query(10, ge=1, le=50),
    user_id: Optional[str] = Query(None),
):
    users, providers, inter = _load_data()

    # Fallback dummy list if data missing
    if providers is None or len(providers) == 0:
        items = [
            {
                "provider_id": f"p_{i:03d}",
                "display_name": f"Provider {i:03d}",
                "score": 1.0 - i * 0.01,
                "rationale": ["dummy"],
            }
            for i in range(k)
        ]
    else:
        if mode == "pop":
            items = score_popularity(users, providers, inter, user_id=user_id, k=k)
        elif mode == "content":
            items = score_content(users, providers, inter, user_id=user_id, k=k)
        elif mode == "cf":
            items = score_cf(users, providers, inter, user_id=user_id, k=k)
        else:  # hybrid
            items = score_hybrid(users, providers, inter, user_id=user_id, k=k)

    if fair == 1:
        items = rerank_head_tail(items, providers, k=k)

    return {"mode": mode, "fair": fair, "user_id": user_id, "items": items}
