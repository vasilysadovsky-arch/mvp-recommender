
# Placeholder evaluation utilities to be expanded later.
def precision_at_k(relevant_ids, ranked_ids, k=10):
    hit = sum(1 for x in ranked_ids[:k] if x in relevant_ids)
    return hit / float(k or 1)
