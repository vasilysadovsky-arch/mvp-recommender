from typing import List, Optional
import numpy as np
import pandas as pd

def _popularity(inter: pd.DataFrame, providers: pd.DataFrame, k: int) -> List[dict]:
    counts = inter.groupby("provider_id").size().rename("cnt")
    merged = providers.merge(counts, on="provider_id", how="left").fillna({"cnt": 0})
    merged["score"] = merged["cnt"].astype(float)
    merged = merged.sort_values("score", ascending=False).head(k)
    return [{"provider_id": r.provider_id, "display_name": r.display_name, "score": float(r.score), "rationale": ["popularity"]} for r in merged.itertuples(index=False)]

def score_cf(users, providers, inter, user_id: Optional[str], k: int) -> List[dict]:
    # Fallbacks
    if providers is None or len(providers) == 0:
        return [{"provider_id": f"p_{i:03d}", "display_name": f"Provider {i:03d}", "score": 1.0 - i*0.01, "rationale": ["popularity"]} for i in range(k)]
    if inter is None or len(inter) == 0 or user_id is None or user_id not in set(inter["user_id"]):
        return _popularity(inter if inter is not None else pd.DataFrame(columns=["user_id","provider_id"]), providers, k)

    # Build item-user matrix (co-occurrence)
    iu = inter.drop_duplicates()[["user_id", "provider_id"]]
    users_list = iu["user_id"].unique()
    items_list = providers["provider_id"].tolist()
    u_index = {u:i for i,u in enumerate(users_list)}
    i_index = {it:i for i,it in enumerate(items_list)}

    # Sparse binary matrix (numpy)
    M = np.zeros((len(items_list), len(users_list)), dtype=np.float32)
    for r in iu.itertuples(index=False):
        i = i_index.get(r.provider_id)
        u = u_index.get(r.user_id)
        if i is not None and u is not None:
            M[i, u] = 1.0

    # Items the user has interacted with
    seen = set(iu.loc[iu["user_id"] == user_id, "provider_id"].tolist())
    if not seen:
        return _popularity(inter, providers, k)

    # Item-item cosine similarities: S = M * M^T ; normalize rows
    # (small scale → fine with dense; for larger, switch to sparse CSR)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Mn = M / norms
    S = Mn @ Mn.T  # (n_items x n_items)

    # Score = sum of similarities to items the user has seen (exclude seen)
    seen_idx = [i_index[s] for s in seen if s in i_index]
    sim_vec = S[:, seen_idx].sum(axis=1)

    # Rank items not seen
    candidates = []
    for pid in items_list:
        if pid in seen:
            continue
        idx = i_index[pid]
        score = float(sim_vec[idx])
        candidates.append((pid, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:k]
    prov_map = providers.set_index("provider_id")["display_name"].to_dict()
    out = [{"provider_id": pid, "display_name": prov_map.get(pid, pid), "score": sc, "rationale": ["item-based CF similarity"]} for pid, sc in top]
    if not out:
        return _popularity(inter, providers, k)
    return out
    if not any(sc > 0 for _, sc in candidates):
        return _popularity(inter, providers, k)
