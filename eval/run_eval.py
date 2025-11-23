import os, csv, math
import pandas as pd
from eval.metrics import precision_at_k, exposure_parity_at_k
from models.content import score_content
from models.cf import score_cf
from models.hybrid import score_hybrid

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)

# Load data
users = pd.read_csv(os.path.join(DATA, "users.csv"))
providers = pd.read_csv(os.path.join(DATA, "providers.csv"))
inter = pd.read_csv(os.path.join(DATA, "interactions.csv"))

# Modes to evaluate
MODES = {
    "content": score_content,
    "cf": score_cf,
    "hybrid": score_hybrid,
}

K = 10
rows = []

for uid in users["user_id"].tolist():
    relevant = set(inter.loc[inter["user_id"] == uid, "provider_id"].tolist())

    for mode, fn in MODES.items():
        recs = fn(users, providers, inter, user_id=uid, k=K)
        ranked = [r["provider_id"] for r in recs]

        # precision@K
        p = precision_at_k(relevant, ranked, k=K)

        # nDCG@K (binary gains)
        gains = [1.0 if pid in relevant else 0.0 for pid in ranked]
        dcg = 0.0
        for i, g in enumerate(gains[:K]):
            dcg += g / (1.0 if i == 0 else math.log2(i + 2))
        ideal = sorted(gains[:K], reverse=True)
        idcg = 0.0
        for i, g in enumerate(ideal):
            idcg += g / (1.0 if i == 0 else math.log2(i + 2))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        # Exposure / fairness @K (head vs tail)
        e_head, e_tail, gap = exposure_parity_at_k(ranked, providers, k=K)

        rows.append({
            "user_id": uid,
            "mode": mode,
            "precision@10": p,
            "nDCG@10": ndcg,
            "exp_head@10": e_head,
            "exp_tail@10": e_tail,
            "exp_gap@10": gap
        })

# Write per-user summary
out_csv = os.path.join(REPORTS, "eval_summary.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "user_id", "mode",
            "precision@10", "nDCG@10",
            "exp_head@10", "exp_tail@10", "exp_gap@10"
        ],
    )
    w.writeheader()
    w.writerows(rows)

# Aggregate by mode
df = pd.DataFrame(rows)
agg = (
    df.groupby("mode")[["precision@10", "nDCG@10", "exp_head@10", "exp_tail@10", "exp_gap@10"]]
    .mean()
    .reset_index()
)
agg_csv = os.path.join(REPORTS, "eval_agg.csv")
agg.to_csv(agg_csv, index=False)

print(f"Wrote per-user metrics to {out_csv}")
print(f"Wrote aggregate metrics to {agg_csv}")
print(agg)
