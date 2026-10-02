from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb


ROOT = MIMIC_ROOT.as_posix()
DERIVED = MIMIC_DERIVED
PAIN = DERIVED / "pain_scores_external_stage1.parquet"
OPIOIDS = DERIVED / "opioid_boluses_external_stage1.parquet"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA preserve_insertion_order=false")

con.execute(
    f"""
    COPY (
        SELECT
            subject_id::BIGINT AS subject_id,
            hadm_id::BIGINT AS hadm_id,
            stay_id::BIGINT AS stay_id,
            try_cast(charttime AS TIMESTAMP) AS charttime,
            try_cast(valuenum AS DOUBLE) AS pain_score
        FROM read_csv_auto(?, sample_size=500000, all_varchar=true)
        WHERE itemid = '223791'
          AND try_cast(valuenum AS DOUBLE) BETWEEN 0 AND 10
          AND try_cast(charttime AS TIMESTAMP) IS NOT NULL
          AND stay_id IS NOT NULL
    ) TO '{sql_path(PAIN)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
    """,
    [ROOT + "/icu/chartevents.csv.gz"],
)

con.execute(
    f"""
    COPY (
        SELECT
            subject_id::BIGINT AS subject_id,
            hadm_id::BIGINT AS hadm_id,
            stay_id::BIGINT AS stay_id,
            try_cast(starttime AS TIMESTAMP) AS administration_time,
            CASE itemid
                WHEN '221833' THEN 'hydromorphone'
                WHEN '225154' THEN 'morphine'
            END AS opioid,
            try_cast(amount AS DOUBLE) AS dose_mg,
            CASE itemid
                WHEN '221833' THEN try_cast(amount AS DOUBLE) * 20.0
                WHEN '225154' THEN try_cast(amount AS DOUBLE) * 3.0
            END AS mme
        FROM read_csv_auto(?, sample_size=500000, all_varchar=true)
        WHERE itemid IN ('221833', '225154')
          AND ordercategoryname = '05-Med Bolus'
          AND statusdescription = 'FinishedRunning'
          AND lower(coalesce(amountuom, '')) = 'mg'
          AND try_cast(starttime AS TIMESTAMP) IS NOT NULL
          AND try_cast(amount AS DOUBLE) > 0
          AND (
              (itemid = '221833' AND try_cast(amount AS DOUBLE) BETWEEN 0.05 AND 5)
              OR
              (itemid = '225154' AND try_cast(amount AS DOUBLE) BETWEEN 0.25 AND 30)
          )
    ) TO '{sql_path(OPIOIDS)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """,
    [ROOT + "/icu/inputevents.csv.gz"],
)

summary = con.execute(
    """
    SELECT
        (SELECT count(*) FROM read_parquet(?)) AS pain_records,
        (SELECT count(DISTINCT stay_id) FROM read_parquet(?)) AS pain_stays,
        (SELECT count(*) FROM read_parquet(?)) AS opioid_boluses,
        (SELECT count(DISTINCT stay_id) FROM read_parquet(?)) AS opioid_stays,
        count(DISTINCT p.stay_id) AS overlapping_stays
    FROM (SELECT DISTINCT stay_id FROM read_parquet(?)) p
    INNER JOIN (SELECT DISTINCT stay_id FROM read_parquet(?)) o USING (stay_id)
    """,
    [str(PAIN), str(PAIN), str(OPIOIDS), str(OPIOIDS), str(PAIN), str(OPIOIDS)],
).fetchdf()

print("MIMIC-IV EXTERNAL PAIN/OPIOID EXTRACTION COMPLETED")
print(summary.to_string(index=False))
print(f"Pain: {PAIN}")
print(f"Opioids: {OPIOIDS}")
con.close()
