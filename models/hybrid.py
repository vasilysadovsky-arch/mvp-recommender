
from typing import List, Optional
from .content import score_content
from .cf import score_cf

def score_hybrid(users, providers, inter, user_id: Optional[str], k: int) -> List[dict]:
    # Simple blend: average rank of content and CF lists
    cont = score_content(users, providers, inter, user_id, k*2)
    cfd = score_cf(users, providers, inter, user_id, k*2)

    # Convert to rank maps
    rank_c = {it["provider_id"]: i for i, it in enumerate(cont, start=1)}
    rank_f = {it["provider_id"]: i for i, it in enumerate(cfd, start=1)}

    # Collect all IDs
    ids = set(rank_c) | set(rank_f)
    rows = []
    for pid in ids:
        rc = rank_c.get(pid, len(ids)+1)
        rf = rank_f.get(pid, len(ids)+1)
        rank = (rc + rf) / 2.0
        rows.append({"provider_id": pid, "display_name": f"Provider {pid}", "score": -rank, "rationale": ["hybrid blend"]})
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:k]
