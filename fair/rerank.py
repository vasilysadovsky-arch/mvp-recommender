
from typing import List
# Minimal fairness placeholder: rotate head/tail labels if present in providers
def rerank_exposure(items: List[dict], providers_df):
    if providers_df is None or "tier" not in providers_df.columns:
        return items
    tier_map = dict(zip(providers_df["provider_id"], providers_df["tier"]))
    head = [it for it in items if tier_map.get(it["provider_id"], "head") == "head"]
    tail = [it for it in items if tier_map.get(it["provider_id"], "head") == "tail"]
    # Interleave to increase tail exposure
    mixed = []
    while head or tail:
        if tail:
            mixed.append(tail.pop(0))
        if head:
            mixed.append(head.pop(0))
    return mixed[:len(items)]
