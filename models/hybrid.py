from typing import List, Optional
import numpy as np
import pandas as pd
from .content import score_content
from .cf import score_cf

def _to_map(rows: List[dict]) -> dict:
    return {r["provider_id"]: float(r["score"]) for r in rows}

def _minmax(d: dict) -> dict:
    if not d:
        return d
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}

def score_hybrid(users, providers, inter, user_id: Optional[str], k: int) -> List[dict]:
    cont = score_content(users, providers, inter, user_id, k*3)
    cf = score_cf(users, providers, inter, user_id, k*3)

    mc = _minmax(_to_map(cont))
    mf = _minmax(_to_map(cf))

    # Cold-start heuristic – if user_id is None or absent in interactions, lean more on content
    alpha = 0.7 if (user_id is None or inter is None or user_id not in set(inter["user_id"])) else 0.5

    # Union of candidate IDs
    ids = set(mc) | set(mf)
    rows = []
    for pid in ids:
        sc = mc.get(pid, 0.0)
        sf = mf.get(pid, 0.0)
        score = alpha*sc + (1 - alpha)*sf
        rows.append({
            "provider_id": pid,
            "display_name": f"Provider {pid}" if " " not in pid else pid,
            "score": float(score),
            "rationale": ["hybrid blend"]
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:k]
