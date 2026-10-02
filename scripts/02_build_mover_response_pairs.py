from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb


INFO = str(MOVER_ROOT / "EPIC_EMR" / "EMR" / "patient_information.csv")
PAIN = str(MOVER_DERIVED / "postop_pain_scores_stage1.parquet")
OPIOIDS = str(MOVER_DERIVED / "postop_opioid_administrations_stage1.parquet")
OUTPUT = MOVER_DERIVED / "postop_opioid_pain_response_pairs_stage2.parquet"
DOSES_OUTPUT = MOVER_DERIVED / "postop_opioid_doses_24h_stage2.parquet"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA preserve_insertion_order=false")

con.execute(
    """
    CREATE TEMP TABLE case_info AS
    SELECT * EXCLUDE (rn)
    FROM (
        SELECT
            LOG_ID,
            MRN,
            try_cast(BIRTH_DATE AS DOUBLE) AS age,
            SEX,
            ASA_RATING,
            PRIMARY_ANES_TYPE_NM,
            PRIMARY_PROCEDURE_NM,
            ICU_ADMIN_FLAG,
            try_cast(LOS AS DOUBLE) AS hospital_los_days,
            try_strptime(AN_STOP_DATETIME, '%m/%d/%y %H:%M') AS anesthesia_stop,
            row_number() OVER (
                PARTITION BY LOG_ID
                ORDER BY (AN_STOP_DATETIME IS NOT NULL)::INT DESC
            ) AS rn
        FROM read_csv_auto(?, sample_size=500000, all_varchar=true)
    )
    WHERE rn = 1 AND anesthesia_stop IS NOT NULL
    """,
    [INFO],
)

con.execute(
    """
    CREATE TEMP TABLE pain AS
    SELECT
        p.LOG_ID,
        p.recorded_time,
        avg(p.pain_score) AS pain_score
    FROM read_parquet(?) p
    INNER JOIN case_info i USING (LOG_ID)
    WHERE p.recorded_time BETWEEN i.anesthesia_stop - INTERVAL 60 MINUTE
                              AND i.anesthesia_stop + INTERVAL 26 HOUR
    GROUP BY p.LOG_ID, p.recorded_time
    """,
    [PAIN],
)

con.execute(
    """
    CREATE TEMP TABLE doses AS
    SELECT
        row_number() OVER (
            ORDER BY o.LOG_ID, o.administration_time, o.opioid,
                     o.administered_dose, o.DISPLAY_NAME
        ) AS dose_id,
        o.*,
        i.age,
        i.SEX,
        i.ASA_RATING,
        i.PRIMARY_ANES_TYPE_NM,
        i.PRIMARY_PROCEDURE_NM,
        i.ICU_ADMIN_FLAG,
        i.hospital_los_days,
        i.anesthesia_stop,
        date_diff('minute', i.anesthesia_stop, o.administration_time) AS postop_minute,
        CASE
            WHEN o.opioid = 'fentanyl' AND lower(o.dose_unit) = 'mcg'
                 AND o.administered_dose BETWEEN 1 AND 500 THEN o.administered_dose * 0.30
            WHEN o.opioid = 'hydromorphone' AND lower(o.dose_unit) = 'mg'
                 AND lower(o.route) LIKE '%venous%' AND o.administered_dose BETWEEN 0.05 AND 5
                 THEN o.administered_dose * 20.0
            WHEN o.opioid = 'morphine' AND lower(o.dose_unit) = 'mg'
                 AND lower(o.route) LIKE '%venous%' AND o.administered_dose BETWEEN 0.25 AND 30
                 THEN o.administered_dose * 3.0
            WHEN o.opioid = 'oxycodone' AND lower(o.dose_unit) = 'mg'
                 AND lower(o.route) LIKE '%oral%' AND o.administered_dose BETWEEN 1 AND 40
                 THEN o.administered_dose * 1.5
            WHEN o.opioid = 'hydrocodone' AND lower(o.dose_unit) = 'tablet'
                 AND lower(o.route) LIKE '%oral%' AND o.administered_dose BETWEEN 0.5 AND 2
                 THEN o.administered_dose *
                      CASE WHEN regexp_matches(lower(o.DISPLAY_NAME), '10-325') THEN 10.0 ELSE 5.0 END
            WHEN o.opioid = 'tramadol' AND lower(o.dose_unit) = 'mg'
                 AND lower(o.route) LIKE '%oral%' AND o.administered_dose BETWEEN 10 AND 200
                 THEN o.administered_dose * 0.10
            WHEN o.opioid = 'codeine' AND lower(o.dose_unit) = 'mg'
                 AND lower(o.route) LIKE '%oral%' AND o.administered_dose BETWEEN 5 AND 120
                 THEN o.administered_dose * 0.15
            WHEN o.opioid = 'meperidine' AND lower(o.dose_unit) = 'mg'
                 AND o.administered_dose BETWEEN 5 AND 150 THEN o.administered_dose * 0.10
            ELSE NULL
        END AS mme
    FROM read_parquet(?) o
    INNER JOIN case_info i USING (LOG_ID)
    WHERE o.opioid NOT IN ('methadone', 'buprenorphine')
      AND o.administered_dose > 0
      AND o.administration_time BETWEEN i.anesthesia_stop
                                    AND i.anesthesia_stop + INTERVAL 24 HOUR
    """,
    [OPIOIDS],
)

con.execute(
    f"""
    COPY (
        SELECT * FROM doses WHERE mme IS NOT NULL
    ) TO '{sql_path(DOSES_OUTPUT)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
)

con.execute(
    """
    CREATE TEMP TABLE pre AS
    SELECT
        d.dose_id,
        p.recorded_time AS pre_time,
        p.pain_score AS pre_score
    FROM doses d
    ASOF LEFT JOIN pain p
      ON d.LOG_ID = p.LOG_ID
     AND d.administration_time >= p.recorded_time
    """
)

con.execute(
    """
    CREATE TEMP TABLE post AS
    SELECT
        d.dose_id,
        p.recorded_time AS post_time,
        p.pain_score AS post_score
    FROM doses d
    ASOF LEFT JOIN pain p
      ON d.LOG_ID = p.LOG_ID
     AND d.administration_time + INTERVAL 15 MINUTE <= p.recorded_time
    """
)

con.execute(
    f"""
    COPY (
        SELECT
            d.*,
            pre.pre_time,
            pre.pre_score,
            post.post_time,
            post.post_score,
            date_diff('minute', pre.pre_time, d.administration_time) AS minutes_from_pre_score,
            date_diff('minute', d.administration_time, post.post_time) AS minutes_to_post_score,
            post.post_score - pre.pre_score AS pain_score_change,
            CASE WHEN pre.pre_score >= 4 AND
                      (pre.pre_score - post.post_score >= 2 OR post.post_score < 4)
                 THEN 1 ELSE 0 END AS clinically_meaningful_response,
            CASE WHEN pre.pre_score >= 4 AND post.post_score >= 4 AND
                      pre.pre_score - post.post_score < 2
                 THEN 1 ELSE 0 END AS inadequate_response,
            (
                SELECT count(*)
                FROM doses d2
                WHERE d2.LOG_ID = d.LOG_ID
                  AND d2.administration_time > d.administration_time
                  AND d2.administration_time <= post.post_time
            ) AS intervening_opioid_doses
        FROM doses d
        INNER JOIN pre USING (dose_id)
        INNER JOIN post USING (dose_id)
        WHERE d.mme IS NOT NULL
          AND date_diff('minute', pre.pre_time, d.administration_time) BETWEEN 0 AND 60
          AND date_diff('minute', d.administration_time, post.post_time) BETWEEN 15 AND 120
    ) TO '{sql_path(OUTPUT)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
)

overall = con.execute(
    """
    SELECT
        count(*) AS paired_doses,
        count(DISTINCT LOG_ID) AS cases,
        count(DISTINCT MRN) AS patients,
        count(*) FILTER (WHERE intervening_opioid_doses = 0) AS uncontaminated_doses,
        count(DISTINCT LOG_ID) FILTER (WHERE intervening_opioid_doses = 0) AS uncontaminated_cases,
        count(*) FILTER (WHERE pre_score >= 4 AND intervening_opioid_doses = 0) AS eligible_painful_doses,
        avg(clinically_meaningful_response) FILTER
            (WHERE pre_score >= 4 AND intervening_opioid_doses = 0) AS response_rate
    FROM read_parquet(?)
    """,
    [str(OUTPUT)],
).fetchdf()

by_drug = con.execute(
    """
    SELECT
        opioid,
        count(*) AS paired_doses,
        count(DISTINCT LOG_ID) AS cases,
        median(mme) AS median_mme,
        median(pre_score) AS median_pre,
        median(post_score) AS median_post,
        median(pain_score_change) AS median_change,
        avg(clinically_meaningful_response) FILTER (WHERE pre_score >= 4) AS response_rate
    FROM read_parquet(?)
    WHERE intervening_opioid_doses = 0
    GROUP BY opioid
    ORDER BY paired_doses DESC
    """,
    [str(OUTPUT)],
).fetchdf()

print("MOVER PAIN-OPIOID RESPONSE PAIRS STAGE 2 COMPLETED")
print("\nOVERALL")
print(overall.to_string(index=False))
print("\nBY OPIOID")
print(by_drug.to_string(index=False))
print(f"\nOutput: {OUTPUT}")
print(f"Doses: {DOSES_OUTPUT}")
con.close()
