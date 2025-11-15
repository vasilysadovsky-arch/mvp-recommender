def precision_at_k(relevant_ids, ranked_ids, k=10):
    if k <= 0: return 0.0
    hits = sum(1 for x in ranked_ids[:k] if x in relevant_ids)
    return hits / float(k)
