
from typing import List, Optional
import numpy as np

def score_cf(users, providers, inter, user_id: Optional[str], k: int) -> List[dict]:
    # Placeholder CF: popularity by provider frequency in interactions
    if inter is None or len(inter) == 0 or providers is None or len(providers) == 0:
        return [{"provider_id": f"p_{i:03d}", "display_name": f"Provider {i:03d}", "score": 1.0 - i*0.01, "rationale": ["popularity"]} for i in range(k)]
    counts = inter.groupby("provider_id").size().rename("cnt")
    merged = providers.merge(counts, on="provider_id", how="left").fillna({"cnt": 0})
    merged["score"] = merged["cnt"].astype(float)
    merged = merged.sort_values("score", ascending=False).head(k)
    out = [{"provider_id": r.provider_id, "display_name": r.display_name, "score": float(r.score), "rationale": ["popularity"]} for r in merged.itertuples(index=False)]
    return out
