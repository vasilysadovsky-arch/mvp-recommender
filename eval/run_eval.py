# eval/run_eval.py

import os
import csv
import pandas as pd

from eval.metrics import precision_at_k, ndcg_at_k, exposure_at_k
from models.content import score_content
from models.cf import score_cf
from models.hybrid import score_hybrid
from fair.rerank import rerank_head_tail
from models.popularity import score_popularity

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)

users = pd.read_csv(os.path.join(DATA, "users.csv"))
providers = pd.read_csv(os.path.join(DATA, "providers.csv"))
inter = pd.read_csv(os.path.join(DATA, "interactions.csv"))

# Count interactions per user to define cohorts
inter_counts = (
    inter.groupby("user_id")["provider_id"]
    .size()
    .to_dict()
)

def user_cohort(uid: str) -> str:
    """Return 'cold-start' if the user has 0 interactions, else 'non-cold-start'."""
    return "cold-start" if inter_counts.get(uid, 0) == 0 else "non-cold-start"

MODES = {
    "pop": score_popularity,
    "content": score_content,
    "cf": score_cf,
    "hybrid": score_hybrid,
}

K = 10

rows = []

for uid in users["user_id"].tolist():
    relevant = set(inter.loc[inter["user_id"] == uid, "provider_id"].tolist())

    for mode, fn in MODES.items():
        # base ranking (fair=0)
        recs_base = fn(users, providers, inter, user_id=uid, k=K * 2)  # a bit more, just in case
        ranked_base = [r["provider_id"] for r in recs_base[:K]]

        p_base = precision_at_k(relevant, ranked_base, k=K)
        ndcg_base = ndcg_at_k(relevant, ranked_base, k=K)
        exp_base = exposure_at_k(ranked_base, providers, k=K, group_col="tier")
        head_exp_base = exp_base.get("head", 0.0)
        tail_exp_base = exp_base.get("tail", 0.0)

        rows.append({
            "user_id": uid,
            "mode": mode,
            "fair": 0,
            "precision@10": p_base,
            "nDCG@10": ndcg_base,
            "head_exposure@10": head_exp_base,
            "tail_exposure@10": tail_exp_base,
        })

        # fairness reranked (fair=1)
        recs_fair = rerank_head_tail(recs_base, providers, k=K)
        ranked_fair = [r["provider_id"] for r in recs_fair]

        p_fair = precision_at_k(relevant, ranked_fair, k=K)
        ndcg_fair = ndcg_at_k(relevant, ranked_fair, k=K)
        exp_fair = exposure_at_k(ranked_fair, providers, k=K, group_col="tier")
        head_exp_fair = exp_fair.get("head", 0.0)
        tail_exp_fair = exp_fair.get("tail", 0.0)

        rows.append({
            "user_id": uid,
            "mode": mode,
            "fair": 1,
            "precision@10": p_fair,
            "nDCG@10": ndcg_fair,
            "head_exposure@10": head_exp_fair,
            "tail_exposure@10": tail_exp_fair,
        })

# save per-user
out_csv = os.path.join(REPORTS, "eval_summary.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "user_id",
            "mode",
            "fair",
            "precision@10",
            "nDCG@10",
            "head_exposure@10",
            "tail_exposure@10",
        ],
    )
    w.writeheader()
    w.writerows(rows)

# aggregate by mode + fair flag
df = pd.DataFrame(rows)
agg = (
    df.groupby(["mode", "fair"])[
        ["precision@10", "nDCG@10", "head_exposure@10", "tail_exposure@10"]
    ]
    .mean()
    .reset_index()
)

agg_csv = os.path.join(REPORTS, "eval_agg.csv")
agg.to_csv(agg_csv, index=False)

print(f"Wrote per-user metrics to {out_csv}")
print(f"Wrote aggregate metrics to {agg_csv}")
print(agg)

# --- Cohort-level aggregates (cold-start vs non-cold-start) ---
df["cohort"] = df["user_id"].map(user_cohort)

cohort_agg = (
    df.groupby(["mode", "fair", "cohort"])[
        ["precision@10", "nDCG@10", "head_exposure@10", "tail_exposure@10"]
    ]
    .mean()
    .reset_index()
)

cohort_csv = os.path.join(REPORTS, "results_by_cohort.csv")
cohort_agg.to_csv(cohort_csv, index=False)

print(f"Wrote cohort metrics to {cohort_csv}")
print(cohort_agg)
