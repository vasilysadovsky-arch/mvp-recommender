# fair/rerank.py

from typing import List, Dict, Any
import pandas as pd


def rerank_head_tail(
    items: List[Dict[str, Any]],
    providers: pd.DataFrame,
    k: int
) -> List[Dict[str, Any]]:
    """
    Very simple fairness-oriented reranker:
    - Split items into 'head' and 'tail' groups based on providers['tier'].
    - Interleave tail and head items to increase tail exposure.
    - Preserve original order within each group.
    """
    if not items:
        return items

    if "provider_id" not in providers.columns or "tier" not in providers.columns:
        return items[:k]

    tier_map = providers.set_index("provider_id")["tier"].to_dict()

    head = []
    tail = []
    other = []

    for it in items:
        pid = it.get("provider_id")
        tier = tier_map.get(pid)
        if tier == "head":
            head.append(it)
        elif tier == "tail":
            tail.append(it)
        else:
            other.append(it)

    out: List[Dict[str, Any]] = []
    hi = ti = oi = 0
    next_tail = True  # try to start with tail

    while len(out) < k and (ti < len(tail) or hi < len(head) or oi < len(other)):
        if next_tail and ti < len(tail):
            out.append(tail[ti]); ti += 1
        elif hi < len(head):
            out.append(head[hi]); hi += 1
        elif ti < len(tail):
            out.append(tail[ti]); ti += 1
        elif oi < len(other):
            out.append(other[oi]); oi += 1
        next_tail = not next_tail

    # if we still do not have k items, append remaining in original order
    for lst, idx in ((tail, ti), (head, hi), (other, oi)):
        while len(out) < k and idx < len(lst):
            out.append(lst[idx])
            idx += 1

    return out[:k]
