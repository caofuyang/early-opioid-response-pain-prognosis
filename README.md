# Analysis code — previous opioid response and subsequent pain relief

This repository reproduces the analyses reported in:

> Previous opioid response and subsequent pain relief in perioperative and intensive care cohorts: a retrospective two-database study

It contains no patient-level data. MOVER and MIMIC-IV are available to credentialed researchers under their respective data-use agreements.

## Inputs (derived tables produced upstream)

| File | Description |
|---|---|
| `data/MOVER/derived/early_analgesic_response_cohort_stage3.parquet` | eligible index doses (one row per case) |
| `data/MOVER/derived/postop_opioid_doses_24h_stage2.parquet` | opioid doses in the 24 h after anaesthesia |
| `data/MOVER/derived/postop_pain_scores_stage1.parquet` | time-stamped pain scores |
| `data/MIMIC-IV/derived/early_analgesic_response_external_stage2.parquet` | eligible index doses (one row per stay) |
| `data/MIMIC-IV/derived/opioid_boluses_external_stage1.parquet` | intravenous morphine and hydromorphone boluses |
| `data/MIMIC-IV/derived/pain_scores_external_stage1.parquet` | bedside pain scores |

Point `MOVER_DERIVED`, `MIMIC_DERIVED` and `PAIN_RESPONSE_RESULTS` at your own paths if these differ.

## Run order

```bash
pip install -r requirements.txt

python scripts/20_rebuild_mover_episodes.py     # MOVER episodes (audited rules)
python scripts/21_rebuild_mimic_episodes.py     # MIMIC-IV episodes
python scripts/26_models_v3.py                  # proportions, bootstrap CIs, models
python scripts/28_baseline_intervening_audit.py # baseline-window audit
python scripts/29_cluster_cis.py                # supplementary and panel B CIs
python scripts/30_route_window_sensitivity.py   # route-specific window sensitivity
python scripts/27_figure1_flow.py               # Figure 1
python scripts/25_figures_v3.py                 # Figure 2
```

## Outputs

`results/pain_response_v2/` receives the episode tables, `models_v31.csv`, `baseline_audit_*.csv`, `panelB_cluster.csv`, `table_S1–S3.csv`, `route_window_summary.csv` and the audit table. Figures are written to `results/figures/` (PNG 300 dpi, PDF, TIFF 600 dpi).

## Note on earlier scripts

`scripts/01`–`14` in the repository are from the earlier submission of this dataset and are retained for provenance. The pipeline described above uses `scripts/20`–`30`.

## Quality-control scripts

`98_manuscript_qa.py` checks wording, citation numbering, abstract length, section completeness and word count. `99_crossref_audit.py` checks that every table, figure and supplementary item cited in the manuscript exists.

## Software

Python 3.10; duckdb, pandas, numpy, statsmodels, matplotlib, pyarrow. Exact versions used for the submitted analysis are listed in `requirements.txt`.

## Licence

MIT. See `LICENSE`.
