
import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

rng = np.random.default_rng(20251101)

# Users
users = pd.DataFrame({
    "user_id": [f"u_{i:03d}" for i in range(1, 21)],
    "sector": rng.choice(["Food", "Textiles", "Tech"], size=20),
    "intent": rng.choice(["Buy", "Sell", "Partner"], size=20),
})
users.to_csv(os.path.join(DATA, "users.csv"), index=False)

# Providers (head/tail tiers for fairness demo)
providers = pd.DataFrame({
    "provider_id": [f"p_{i:03d}" for i in range(1, 51)],
    "display_name": [f"Provider {i:03d}" for i in range(1, 51)],
    "tier": ["head"]*15 + ["tail"]*35
})
providers.to_csv(os.path.join(DATA, "providers.csv"), index=False)

# Interactions (implicit feedback) – skewed to head
pairs = []
for u in users["user_id"]:
    n = rng.integers(5, 15)
    # 70% head, 30% tail
    head_ids = [f"p_{i:03d}" for i in range(1, 16)]
    tail_ids = [f"p_{i:03d}" for i in range(16, 51)]
    chosen = rng.choice(head_ids, size=int(n*0.7)).tolist() + rng.choice(tail_ids, size=int(n*0.3)).tolist()
    for pid in chosen:
        pairs.append((u, pid))
inter = pd.DataFrame(pairs, columns=["user_id", "provider_id"])
inter.to_csv(os.path.join(DATA, "interactions.csv"), index=False)

print(f"Generated users={len(users)}, providers={len(providers)}, interactions={len(inter)} at {DATA}")
