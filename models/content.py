from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _provider_text(df: pd.DataFrame) -> List[str]:
    # Build text like: "sector_Food intent_Buy keywords: dairy organic"
    texts = []
    for r in df.itertuples(index=False):
        sector = getattr(r, "sector", "")
        keywords = getattr(r, "keywords", "")
        name = getattr(r, "display_name", "")
        texts.append(f"sector_{sector} {keywords} {name}")
    return texts

def _user_text_row(user_row: pd.Series) -> str:
    sector = user_row.get("sector", "")
    intent = user_row.get("intent", "")
    return f"sector_{sector} intent_{intent}"

def score_content(users, providers, inter, user_id: Optional[str], k: int) -> List[dict]:
    if providers is None or len(providers) == 0:
        return [{"provider_id": f"p_{i:03d}", "display_name": f"Provider {i:03d}", "score": 1.0 - i*0.01, "rationale": ["content"]} for i in range(k)]

    # Vectorize providers
    prov_texts = _provider_text(providers)
    vec = TfidfVectorizer(min_df=1)
    Xp = vec.fit_transform(prov_texts)

    # Build user query text
    if user_id and users is not None and user_id in set(users["user_id"]):
        urow = users.loc[users["user_id"] == user_id].iloc[0]
        utext = _user_text_row(urow)
    else:
        # Cold-start user – neutral query to avoid empty vector
        utext = "sector_Food sector_Textiles sector_Tech intent_Buy intent_Sell intent_Partner"

    Xu = vec.transform([utext])

    # Cosine similarity
    sims = cosine_similarity(Xu, Xp).ravel()  # shape (n_providers,)
    order = np.argsort(-sims)[:k]

    out = []
    for idx in order:
        row = providers.iloc[idx]
        out.append({
            "provider_id": str(row["provider_id"]),
            "display_name": str(row["display_name"]),
            "score": float(sims[idx]),
            "rationale": ["sector/intent/keywords match"]
        })
    return out
