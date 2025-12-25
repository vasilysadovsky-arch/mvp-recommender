# models/popularity.py

from typing import List, Dict, Any
import pandas as pd


def _compute_popularity(interactions: pd.DataFrame) -> pd.Series:
    """
    Returns a Series: index = provider_id, value = interaction count.
    """
    if interactions.empty or "provider_id" not in interactions.columns:
        return pd.Series(dtype=float)

    return (
        interactions
        .groupby("provider_id")["user_id"]
        .count()
        .sort_values(ascending=False)
    )


def score_popularity(
    users: pd.DataFrame,
    providers: pd.DataFrame,
    interactions: pd.DataFrame,
    user_id: str,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Simple popularity baseline:
    - Rank exhibitors by global interaction count.
    - Optionally exclude providers already interacted with by this user.
    - Return top-k as list of dicts with provider_id, score, rationale.
    """
    pop = _compute_popularity(interactions)

    # if no interactions at all, fall back to arbitrary provider order
    if pop.empty:
        ranked_ids = providers["provider_id"].tolist()
    else:
        ranked_ids = pop.index.tolist()

    # exclude providers already interacted with by this user (optional but nice)
    if "user_id" in interactions.columns and "provider_id" in interactions.columns:
        seen = set(
            interactions.loc[interactions["user_id"] == user_id, "provider_id"].tolist()
        )
        ranked_ids = [pid for pid in ranked_ids if pid not in seen]

    # truncate to at most k items, but we might return fewer if there are not enough
    ranked_ids = ranked_ids[:k]

    # build lookup for provider display info
    prov_df = providers.set_index("provider_id")
    items: List[Dict[str, Any]] = []

    for pid in ranked_ids:
        if pid in prov_df.index:
            row = prov_df.loc[pid]
            display_name = str(row.get("display_name", pid))
            tier = str(row.get("tier", "unknown"))
        else:
            display_name = pid
            tier = "unknown"

        score = float(pop.get(pid, 0.0))  # 0.0 if missing

        items.append(
            {
                "provider_id": pid,
                "display_name": display_name,
                "score": score,
                "rationale": [
                    "popularity baseline",
                    f"tier={tier}",
                ],
            }
        )

    return items
