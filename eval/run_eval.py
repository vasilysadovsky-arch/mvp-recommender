import os, csv
import pandas as pd
from eval.metrics import precision_at_k
from models.content import score_content
from models.cf import score_cf
from models.hybrid import score_hybrid

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)

users = pd.read_csv(os.path.join(DATA, "users.csv"))
providers = pd.read_csv(os.path.join(DATA, "providers.csv"))
inter = pd.read_csv(os.path.join(DATA, "interactions.csv"))

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
        p = precision_at_k(relevant, ranked, k=K)

        # quick nDCG@K
        gains = [1.0 if pid in relevant else 0.0 for pid in ranked]
        dcg = sum(gains[i] / (1.0 if i == 0 else __import__("math").log2(i+2)) for i in range(len(gains)))
        ideal = sorted(gains, reverse=True)
        idcg = sum(ideal[i] / (1.0 if i == 0 else __import__("math").log2(i+2)) for i in range(len(ideal)))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        rows.append({"user_id": uid, "mode": mode, "precision@10": p, "nDCG@10": ndcg})

out_csv = os.path.join(REPORTS, "eval_summary.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["user_id","mode","precision@10","nDCG@10"])
    w.writeheader(); w.writerows(rows)

# Aggregate
df = pd.DataFrame(rows)
agg = df.groupby("mode")[["precision@10","nDCG@10"]].mean().reset_index()
agg_csv = os.path.join(REPORTS, "eval_agg.csv")
agg.to_csv(agg_csv, index=False)

print(f"Wrote per-user metrics to {out_csv}")
print(f"Wrote aggregate metrics to {agg_csv}")
print(agg)
