# Reproducibility code: early opioid-associated pain response

This archive contains the analysis code accompanying the manuscript submitted to *Regional Anesthesia & Pain Medicine*. It covers MOVER cohort construction, association and sensitivity analyses, temporal evaluation, the frozen 2021 test, directional MIMIC-IV replication, statistical checks, and figure generation.

## Data are not included

No patient-level or row-level derived data are included. MOVER and MIMIC-IV access is controlled by their respective data-use agreements. Users must obtain the source datasets independently and comply with all training, credentialing, privacy, and institutional requirements.

## Configuration

Python 3.11 or later is recommended. Install dependencies with:

    python -m pip install -r requirements.txt

The scripts use environment variables rather than machine-specific paths. Copy `config.example.env` to your preferred environment configuration and set the paths for your authorised local copies of MOVER and MIMIC-IV. The default expected layout is documented in `DATA_NOT_INCLUDED.md`.

## Run order

Run from the `scripts` directory so that `config.py` is importable:

1. `01_extract_mover_postop_pain_opioids.py`
2. `02_build_mover_response_pairs.py`
3. `03_build_mover_analysis_cohort.py`
4. `04_temporal_evaluation.py`
5. `05_confounding_sensitivity.py`
6. `06_locked_2021_test.py`
7. `07_extract_mimic_external.py`
8. `08_build_mimic_replication.py`
9. `09_baseline_table.py`
10. `10_submission_statistical_checks.py`
11. `11_main_figures.py`
12. `12_supplementary_figures.py`

Scripts 07-08 can be run independently after authorised MIMIC-IV data have been placed in the configured location. Outputs are written below `PAIN_RESPONSE_RESULTS`; derived MOVER and MIMIC files are written below their configured derived directories.

## Reproducibility scope

The code is provided to document the implemented analysis. Exact reruns require the same database versions, source-table layouts, and data-use permissions described in the manuscript. The archive does not transfer permission to use either database.

## Randomness

Bootstrap and model-evaluation scripts use the fixed seed `20260827` where stochastic resampling is performed.

## Citation and licence

Citation metadata and an open-source licence should be added when the public repository is created. Until then, copyright remains with the authors and the archive is supplied for editorial and peer-review assessment.
