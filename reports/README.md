# Evaluation snapshot

Artifacts:
- `eval_summary.csv` – per user per mode (precision@10, nDCG@10)
- `eval_agg.csv` – averages by mode

How to reproduce:
```bash
source .venv/bin/activate
make -B eval
