import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

rng = np.random.default_rng(20251101)

sectors = ["Food", "Textiles", "Tech"]
intents = ["Buy", "Sell", "Partner"]
keywords_by_sector = {
    "Food": ["dairy", "meat", "seafood", "organic", "ingredients"],
    "Textiles": ["cotton", "fabric", "yarn", "garment", "sourcing"],
    "Tech": ["software", "ai", "cloud", "hardware", "automation"],
}

# Users
users = pd.DataFrame({
    "user_id": [f"u_{i:03d}" for i in range(1, 21)],
    "sector": rng.choice(sectors, size=20),
    "intent": rng.choice(intents, size=20),
})
users.to_csv(os.path.join(DATA, "users.csv"), index=False)

# Providers (head/tail tiers + sector + keywords for content model)
prov_ids = [f"p_{i:03d}" for i in range(1, 51)]
prov_secs = rng.choice(sectors, size=len(prov_ids))
prov_kw = []
for s in prov_secs:
    kw = rng.choice(keywords_by_sector[s], size=2, replace=False)
    prov_kw.append(" ".join(kw))

providers = pd.DataFrame({
    "provider_id": prov_ids,
    "display_name": [f"Provider {i:03d}" for i in range(1, 51)],
    "tier": ["head"]*15 + ["tail"]*35,
    "sector": prov_secs,
    "keywords": prov_kw
})
providers.to_csv(os.path.join(DATA, "providers.csv"), index=False)

# Interactions (implicit feedback) – skewed to head and user sector
pairs = []
for u, u_sec in zip(users["user_id"], users["sector"]):
    n = rng.integers(6, 14)
    head_ids = [f"p_{i:03d}" for i in range(1, 16)]
    tail_ids = [f"p_{i:03d}" for i in range(16, 51)]
    # bias toward providers in the same sector
    same_sec = providers[providers["sector"] == u_sec]["provider_id"].tolist()
    rest = [pid for pid in prov_ids if pid not in same_sec]
    chosen = rng.choice(same_sec, size=max(1, int(n*0.6))).tolist() + rng.choice(rest, size=max(1, n - int(n*0.6))).tolist()
    # add head/tail tilt
    chosen = rng.choice(head_ids, size=int(n*0.5)).tolist() + rng.choice(tail_ids, size=n - int(n*0.5)).tolist()
    for pid in chosen:
        pairs.append((u, pid))

inter = pd.DataFrame(pairs, columns=["user_id", "provider_id"])
inter.to_csv(os.path.join(DATA, "interactions.csv"), index=False)

print(f"Generated users={len(users)}, providers={len(providers)}, interactions={len(inter)} at {DATA}")
