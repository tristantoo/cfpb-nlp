# cfpb-nlp

Predict the CFPB complaint `Product` from the consumer narrative with TF-IDF + logistic regression, using a forward split: train on 2024-01..12, test on 2025-01..08. NMF topic clusters provide a small lending-vocabulary glossary.

- Pre-registered plan and deviations: `docs/PLAN.md`
- Write-up (hypothesis rulings, per-class tables, sensitivity): `docs/REPORT.md`
- Results: `outputs/results.json` (single pre-registered scoring) and `outputs/posthoc_results.json` (exploratory extras, deviation D5)

Headline: test macro-F1 0.785 [0.778, 0.791] against a majority baseline of 0.123. H2 (forward drift lowers test below validation) is falsified: test 0.785 vs validation 0.761.

## Run

```
uv sync
uv run python scripts/extract.py    # needs data/raw/complaints_wayback_20251209.csv.zip (sha256 in PLAN §2)
uv run python scripts/run_model.py
uv run python scripts/run_topics.py
uv run python scripts/posthoc.py
uv run pytest
```

Raw and interim data are gitignored. `extract.py` streams the 7.5 GB CSV in 100k-row chunks.

## Authorship

Built with AI coding assistance (Claude Code). I set the scope and questions and reviewed the methods, code and results.
