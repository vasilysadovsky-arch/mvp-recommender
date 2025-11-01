
from typing import List, Optional
import numpy as np

# Placeholder: returns deterministic pseudo-scores based on provider_id hash.
def score_content(users, providers, inter, user_id: Optional[str], k: int) -> List[dict]:
    rng = np.random.default_rng(42)  # fixed seed for determinism
    rows = []
    for pid, name in zip(providers.get('provider_id', []), providers.get('display_name', [])):
        # Simple deterministic score: length of id + a small jitter
        base = len(str(pid))
        jitter = (abs(hash(pid)) % 100) / 10000.0
        rows.append({"provider_id": pid, "display_name": name, "score": base + jitter, "rationale": ["sector match"]})
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:k]
