
# MVP Recommender – Starter Repo

A tiny, runnable scaffold for your capstone MVP.  
**Goal:** serve a `/topN` endpoint and return a Top‑N list using placeholder logic, so you can iterate fast.

## Quick start

1) **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) **Install deps**
```bash
pip install -r requirements.txt
```

3) **Generate tiny synthetic data (optional)**
```bash
python data_synth/generate.py
```

4) **Run the API**
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

5) **Test the endpoint**
```bash
curl "http://127.0.0.1:8000/topN?mode=content&fair=0&k=5&user_id=u_001"
```

You should get a JSON with `items` (dummy list if no data is present yet).

## Repo structure
```
api/           # FastAPI app with /topN
models/        # Content/CF/Hybrid placeholders
fair/          # Fairness re-ranker placeholder
eval/          # Metrics placeholder
data_synth/    # Synthetic data generator
data/          # (git-ignored) CSVs generated here
reports/       # (git-ignored) outputs, plots, tables
tests/         # Minimal smoke tests
```

## Next steps
- Replace placeholder scoring with TF‑IDF (content) and add ALS/BPR (CF) later.
- Implement evaluation scripts in `eval/`.
- Add rationale badges and UI if needed.

## Evaluate (Week 3)

```bash
source .venv/bin/activate
make -B eval
```

## Outputs (in reports/):
- eval_summary.csv – per user × mode metrics (precision@10, nDCG@10)
- eval_agg.csv – average metrics by mode
