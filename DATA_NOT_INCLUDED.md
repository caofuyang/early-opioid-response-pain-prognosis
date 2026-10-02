# Data layout and access

The code archive intentionally contains no research data.

Default paths below are relative to the archive root and can be overridden by environment variables:

```
data/
  MOVER/
    flowsheets_cleaned/flowsheet_part1.csv
    EPIC_EMR/EMR/patient_information.csv
    EPIC_EMR/EMR/patient_medications.csv
    EPIC_EMR/EMR/patient_history.csv
    EPIC_EMR/EMR/patient_post_op_complications.csv
  MIMIC-IV/mimic-iv-3.1/
    hosp/admissions.csv.gz
    hosp/patients.csv.gz
    icu/chartevents.csv.gz
    icu/icustays.csv.gz
    icu/inputevents.csv.gz
  derived/
    MOVER/
    MIMIC-IV/
```

Actual MOVER file organisation may depend on the authorised distribution. If filenames or locations differ, update only the environment configuration or the corresponding entries in `config.py`; do not place credentials in scripts.
