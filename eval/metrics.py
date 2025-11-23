def precision_at_k(relevant_ids, ranked_ids, k=10):
    if k <= 0: return 0.0
    hits = sum(1 for x in ranked_ids[:k] if x in relevant_ids)
    return hits / float(k)
def exposure_parity_at_k(ranked_ids, providers_df, k=10):
    head = set(providers_df.loc[providers_df["tier"] == "head", "provider_id"])
    tail = set(providers_df.loc[providers_df["tier"] == "tail", "provider_id"])
    topk = ranked_ids[:k]
    e_head = sum(1 for x in topk if x in head) / float(k or 1)
    e_tail = sum(1 for x in topk if x in tail) / float(k or 1)
    return e_head, e_tail, e_tail - e_head  # >0 means more tail exposure

