from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb


FLOW = str(MOVER_ROOT / "flowsheets_cleaned" / "flowsheet_part1.csv")
MEDS = str(MOVER_ROOT / "EPIC_EMR" / "EMR" / "patient_medications.csv")
DERIVED = MOVER_DERIVED
PAIN = DERIVED / "postop_pain_scores_stage1.parquet"
OPIOIDS = DERIVED / "postop_opioid_administrations_stage1.parquet"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA preserve_insertion_order=false")

con.execute(
    f"""
    COPY (
        SELECT
            LOG_ID,
            MRN,
            try_cast(RECORDED_TIME AS TIMESTAMP) AS recorded_time,
            try_cast(MEAS_VALUE AS DOUBLE) AS pain_score,
            FLO_NAME AS source_template,
            FLO_DISPLAY_NAME,
            RECORD_TYPE,
            UNITS
        FROM read_csv_auto(?, sample_size=500000, all_varchar=true)
        WHERE upper(coalesce(RECORD_TYPE, '')) = 'POST-OP'
          AND lower(trim(coalesce(FLO_DISPLAY_NAME, ''))) = 'pain score'
          AND try_cast(MEAS_VALUE AS DOUBLE) BETWEEN 0 AND 10
          AND try_cast(RECORDED_TIME AS TIMESTAMP) IS NOT NULL
          AND LOG_ID IS NOT NULL
    ) TO '{sql_path(PAIN)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
    """,
    [FLOW],
)

con.execute(
    f"""
    COPY (
        SELECT
            LOG_ID,
            MRN,
            try_cast(MED_ACTION_TIME AS TIMESTAMP) AS administration_time,
            CASE
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'hydromorphone') THEN 'hydromorphone'
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'hydrocodone') THEN 'hydrocodone'
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'oxycodone') THEN 'oxycodone'
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'morphine') THEN 'morphine'
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'fentanyl') THEN 'fentanyl'
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'tramadol') THEN 'tramadol'
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'codeine') THEN 'codeine'
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'meperidine') THEN 'meperidine'
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'methadone') THEN 'methadone'
                WHEN regexp_matches(lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')), 'buprenorphine') THEN 'buprenorphine'
                ELSE 'other_opioid'
            END AS opioid,
            try_cast(ADMIN_SIG AS DOUBLE) AS administered_dose,
            DOSE_UNIT_NM AS dose_unit,
            MED_ROUTE_NM AS route,
            DISPLAY_NAME,
            MEDICATION_NM,
            MAR_ACTION_NM,
            RECORD_TYPE
        FROM read_csv_auto(?, sample_size=500000, all_varchar=true)
        WHERE upper(coalesce(RECORD_TYPE, '')) = 'POST-OP'
          AND lower(trim(coalesce(MAR_ACTION_NM, ''))) = 'given'
          AND try_cast(MED_ACTION_TIME AS TIMESTAMP) IS NOT NULL
          AND LOG_ID IS NOT NULL
          AND regexp_matches(
              lower(coalesce(DISPLAY_NAME,'') || ' ' || coalesce(MEDICATION_NM,'')),
              'morphine|hydromorphone|fentanyl|oxycodone|hydrocodone|tramadol|codeine|meperidine|methadone|buprenorphine'
          )
    ) TO '{sql_path(OPIOIDS)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
    """,
    [MEDS],
)

pain_summary = con.execute(
    """
    SELECT count(*) AS records, count(DISTINCT LOG_ID) AS cases,
           count(DISTINCT MRN) AS patients, median(pain_score) AS median_score,
           quantile_cont(pain_score, 0.25) AS p25,
           quantile_cont(pain_score, 0.75) AS p75
    FROM read_parquet(?)
    """,
    [str(PAIN)],
).fetchdf()
opioid_summary = con.execute(
    """
    SELECT opioid, count(*) AS administrations, count(DISTINCT LOG_ID) AS cases,
           count(*) FILTER (WHERE administered_dose IS NULL) AS missing_dose
    FROM read_parquet(?) GROUP BY opioid ORDER BY administrations DESC
    """,
    [str(OPIOIDS)],
).fetchdf()
linkage = con.execute(
    """
    SELECT
        (SELECT count(DISTINCT LOG_ID) FROM read_parquet(?)) AS pain_cases,
        (SELECT count(DISTINCT LOG_ID) FROM read_parquet(?)) AS opioid_cases,
        count(DISTINCT p.LOG_ID) AS linked_cases
    FROM (SELECT DISTINCT LOG_ID FROM read_parquet(?)) p
    INNER JOIN (SELECT DISTINCT LOG_ID FROM read_parquet(?)) o USING (LOG_ID)
    """,
    [str(PAIN), str(OPIOIDS), str(PAIN), str(OPIOIDS)],
).fetchdf()

print("MOVER POSTOPERATIVE PAIN/OPIOID EXTRACTION COMPLETED")
print("\nPAIN")
print(pain_summary.to_string(index=False))
print("\nOPIOIDS")
print(opioid_summary.to_string(index=False))
print("\nLINKAGE")
print(linkage.to_string(index=False))
print(f"\nPain: {PAIN}")
print(f"Opioids: {OPIOIDS}")
con.close()
