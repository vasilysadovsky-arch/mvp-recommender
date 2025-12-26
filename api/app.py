
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
import re

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


class MetaResponse(BaseModel):
    sectors: List[str] = []
    intents: List[str] = []
    users: List[str] = []

class MetricsResponse(BaseModel):
    mode: str
    fair: int
    k: int
    precision_at_k: float
    ndcg_at_k: float

def _parse_csv_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p]

def _filter_providers(providers: pd.DataFrame, sectors: Optional[str]) -> pd.DataFrame:
    if providers is None or len(providers) == 0:
        return providers
    sec_list = _parse_csv_list(sectors)
    if not sec_list:
        return providers
    return providers.loc[providers["sector"].astype(str).isin(sec_list)].copy()

def _keyword_boost(items: List[dict], providers: pd.DataFrame, q: Optional[str]) -> List[dict]:
    if not q or providers is None or len(providers) == 0 or not items:
        return items
    tokens = [t for t in re.split(r"\s+", q.strip().lower()) if t]
    if not tokens:
        return items

    prov = providers.set_index("provider_id")[["display_name", "sector", "keywords"]].fillna("").astype(str)

    boosted = []
    for it in items:
        pid = it.get("provider_id")
        try:
            row = prov.loc[pid]
            hay = f"{row['display_name']} {row['sector']} {row['keywords']}".lower()
        except Exception:
            hay = ""
        hit = any(tok in hay for tok in tokens)
        score = float(it.get("score", 0.0))
        rationale = list(it.get("rationale") or [])
        if hit:
            # small, stable bump to keep ranking interpretable
            score = score + 0.02
            if "keyword match" not in rationale:
                rationale.append("keyword match")
        boosted.append({**it, "score": score, "rationale": rationale})
    return boosted

def _sort_items(items: List[dict], sort: str) -> List[dict]:
    if not items:
        return items
    if sort == "name_asc":
        return sorted(items, key=lambda x: str(x.get("display_name", "")))
    # default
    return sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)


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

@app.get("/meta", response_model=MetaResponse)
def meta():
    users, providers, inter = _load_data()

    # Users list for UI datalist
    user_ids = []
    if users is not None and len(users) and "user_id" in users.columns:
        user_ids = users["user_id"].astype(str).tolist()

    # Optional: sectors list (may be unused in UI, but harmless)
    sectors_list = []
    if providers is not None and len(providers) and "sector" in providers.columns:
        sectors_list = sorted(set(providers["sector"].astype(str).tolist()))

    # Optional: intents list (only if present in users.csv)
    intents_list = []
    if users is not None and len(users) and "intent" in users.columns:
        intents_list = sorted(set(users["intent"].astype(str).tolist()))

    return {"sectors": sectors_list, "intents": intents_list, "users": user_ids}


@app.get("/metrics", response_model=MetricsResponse)
def metrics(
    mode: str = Query(..., pattern="^(pop|content|cf|hybrid)$"),
    fair: int = Query(0, ge=0, le=1),
    k: int = Query(10, ge=1, le=50),
):
    # MVP: read precomputed aggregate metrics from reports/eval_agg.csv
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    fp = os.path.join(reports_dir, "eval_agg.csv")
    if not os.path.exists(fp):
        return {"mode": mode, "fair": fair, "k": k, "precision_at_k": 0.0, "ndcg_at_k": 0.0}
    df = pd.read_csv(fp)
    row = df.loc[(df["mode"] == mode) & (df["fair"] == fair)]
    if len(row) == 0:
        return {"mode": mode, "fair": fair, "k": k, "precision_at_k": 0.0, "ndcg_at_k": 0.0}
    r = row.iloc[0]
    # Columns are named precision@10, nDCG@10 in the CSV
    prec = float(r.get(f"precision@{k}", r.get("precision@10", 0.0)))
    ndcg = float(r.get(f"nDCG@{k}", r.get("nDCG@10", 0.0)))
    return {"mode": mode, "fair": fair, "k": k, "precision_at_k": prec, "ndcg_at_k": ndcg}

@app.get("/topN", response_model=TopNResponse)
def topN(
    mode: str = Query("content", pattern="^(pop|content|cf|hybrid)$"),
    fair: int = Query(0, ge=0, le=1),
    k: int = Query(10, ge=1, le=50),
    user_id: Optional[str] = Query(None),
    sectors: Optional[str] = Query(None, description="Comma-separated provider sectors"),
    q: Optional[str] = Query(None, description="Keyword search terms"),
    sort: str = Query("score_desc", pattern="^(score_desc|name_asc)$"),
    intent: Optional[str] = Query(None, description="Override user intent for the current request"),
):
    users, providers, inter = _load_data()

    # Apply request-scoped overrides / filters
    if intent is not None and users is not None and user_id is not None and "intent" in users.columns:
        users = users.copy()
        users.loc[users["user_id"].astype(str) == str(user_id), "intent"] = str(intent)

    providers_f = _filter_providers(providers, sectors)

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
            items = score_popularity(users, providers_f, inter, user_id=user_id, k=k)
        elif mode == "content":
            items = score_content(users, providers_f, inter, user_id=user_id, k=k)
        elif mode == "cf":
            items = score_cf(users, providers_f, inter, user_id=user_id, k=k)
        else:  # hybrid
            items = score_hybrid(users, providers_f, inter, user_id=user_id, k=k)

    # Keyword boost (applied before fairness so reranker sees updated scores)
    items = _keyword_boost(items, providers_f if providers_f is not None else providers, q)
    items = _sort_items(items, 'score_desc')

    # Apply fairness reranking WITHOUT undoing it via score sorting
    if fair == 1:
        items = rerank_head_tail(items, providers_f if providers_f is not None else providers, k=k)
        if sort == "name_asc":
            items = _sort_items(items, "name_asc")
    else:
        items = _sort_items(items, sort)

    return {"mode": mode, "fair": fair, "user_id": user_id, "items": items}

