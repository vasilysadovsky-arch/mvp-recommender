# eval/metrics.py

from typing import Iterable, List, Dict
import math
import pandas as pd


def precision_at_k(relevant_ids: Iterable[str], ranked_ids: List[str], k: int = 10) -> float:
    relevant = set(relevant_ids)
    if k <= 0:
        return 0.0
    hits = sum(1 for x in ranked_ids[:k] if x in relevant)
    return hits / float(k)


def ndcg_at_k(relevant_ids: Iterable[str], ranked_ids: List[str], k: int = 10) -> float:
    relevant = set(relevant_ids)
    gains = [1.0 if pid in relevant else 0.0 for pid in ranked_ids[:k]]
    if not gains:
        return 0.0

    dcg = 0.0
    for i, g in enumerate(gains):
        dcg += g / (1.0 if i == 0 else math.log2(i + 2))

    ideal = sorted(gains, reverse=True)
    idcg = 0.0
    for i, g in enumerate(ideal):
        idcg += g / (1.0 if i == 0 else math.log2(i + 2))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def exposure_at_k(
    ranked_ids: List[str],
    providers: pd.DataFrame,
    k: int = 10,
    group_col: str = "tier"
) -> Dict[str, float]:
    """
    Simple exposure metric with position bias: weight = 1/log2(rank+1).
    Returns exposure per group (e.g. 'head', 'tail') in the providers[group_col].
    """
    # map provider_id -> group value
    if "provider_id" not in providers.columns:
        return {}
    prov_df = providers.set_index("provider_id")
    exposure: Dict[str, float] = {}

    for i, pid in enumerate(ranked_ids[:k]):
        # position i = 0.., weight 1/log2(i+2)
        w = 1.0 if i == 0 else 1.0 / math.log2(i + 2)
        if pid not in prov_df.index:
            group_val = "unknown"
        else:
            group_val = str(prov_df.loc[pid, group_col]) if group_col in prov_df.columns else "unknown"
        exposure[group_val] = exposure.get(group_val, 0.0) + w

    return exposure
