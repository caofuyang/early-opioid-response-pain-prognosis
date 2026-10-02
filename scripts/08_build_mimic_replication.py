from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = MIMIC_ROOT.as_posix()
PAIN = str(MIMIC_DERIVED / "pain_scores_external_stage1.parquet")
OPIOIDS = str(MIMIC_DERIVED / "opioid_boluses_external_stage1.parquet")
COHORT = MIMIC_DERIVED / "early_analgesic_response_external_stage2.parquet"
OUT = RESULTS_ROOT / "mimic_pain_response_external"
OUT.mkdir(parents=True, exist_ok=True)


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA preserve_insertion_order=false")
con.execute(
    """
    CREATE TEMP TABLE stays AS
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        try_cast(i.intime AS TIMESTAMP) AS intime,
        try_cast(i.outtime AS TIMESTAMP) AS outtime,
        i.first_careunit,
        p.gender,
        try_cast(p.anchor_age AS INTEGER)
            + year(try_cast(i.intime AS TIMESTAMP))
            - try_cast(p.anchor_year AS INTEGER) AS age,
        a.race,
        a.admission_type
    FROM read_csv_auto(?, sample_size=500000, all_varchar=true) i
    INNER JOIN read_csv_auto(?, sample_size=500000, all_varchar=true) p USING (subject_id)
    INNER JOIN read_csv_auto(?, sample_size=500000, all_varchar=true) a USING (subject_id, hadm_id)
    """,
    [ROOT + "/icu/icustays.csv.gz", ROOT + "/hosp/patients.csv.gz", ROOT + "/hosp/admissions.csv.gz"],
)
con.execute(
    """
    CREATE TEMP TABLE pain AS
    SELECT stay_id, charttime, avg(pain_score) AS pain_score
    FROM read_parquet(?)
    GROUP BY stay_id, charttime
    """,
    [PAIN],
)
con.execute(
    """
    CREATE TEMP TABLE doses AS
    SELECT
        row_number() OVER (ORDER BY stay_id, administration_time, opioid, dose_mg) AS dose_id,
        o.*,
        s.intime,
        s.outtime,
        s.first_careunit,
        s.gender,
        s.age,
        s.race,
        s.admission_type,
        date_diff('minute', s.intime, o.administration_time) AS icu_minute
    FROM read_parquet(?) o
    INNER JOIN stays s USING (subject_id, hadm_id, stay_id)
    WHERE o.administration_time BETWEEN s.intime AND s.outtime
    """,
    [OPIOIDS],
)
con.execute(
    """
    CREATE TEMP TABLE pre AS
    SELECT d.dose_id, p.charttime AS pre_time, p.pain_score AS pre_score
    FROM doses d
    ASOF LEFT JOIN pain p
      ON d.stay_id = p.stay_id AND d.administration_time >= p.charttime
    """
)
con.execute(
    """
    CREATE TEMP TABLE post AS
    SELECT d.dose_id, p.charttime AS post_time, p.pain_score AS post_score
    FROM doses d
    ASOF LEFT JOIN pain p
      ON d.stay_id = p.stay_id
     AND d.administration_time + INTERVAL 15 MINUTE <= p.charttime
    """
)
con.execute(
    """
    CREATE TEMP TABLE paired AS
    SELECT
        d.*,
        pre.pre_time,
        pre.pre_score,
        post.post_time,
        post.post_score,
        post.post_score - pre.pre_score AS pain_score_change,
        CASE WHEN pre.pre_score >= 4 AND
                  (pre.pre_score - post.post_score >= 2 OR post.post_score < 4)
             THEN 1 ELSE 0 END AS clinically_meaningful_response,
        CASE WHEN pre.pre_score >= 4 AND post.post_score >= 4 AND
                  pre.pre_score - post.post_score < 2
             THEN 1 ELSE 0 END AS inadequate_response,
        (
            SELECT count(*) FROM doses d2
            WHERE d2.stay_id = d.stay_id
              AND d2.administration_time > d.administration_time
              AND d2.administration_time <= post.post_time
        ) AS intervening_opioid_doses
    FROM doses d
    INNER JOIN pre USING (dose_id)
    INNER JOIN post USING (dose_id)
    WHERE date_diff('minute', pre.pre_time, d.administration_time) BETWEEN 0 AND 60
      AND date_diff('minute', d.administration_time, post.post_time) BETWEEN 15 AND 120
    """
)
con.execute(
    f"""
    COPY (
        WITH first_response AS (
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY stay_id ORDER BY administration_time, dose_id
                ) AS rn
                FROM paired
                WHERE intervening_opioid_doses = 0
                  AND pre_score >= 4
                  AND icu_minute BETWEEN 0 AND 360
                  AND post_time <= intime + INTERVAL 6 HOUR
            ) WHERE rn = 1
        ),
        later_pain AS (
            SELECT
                f.stay_id,
                count(*) AS later_pain_count,
                avg(p.pain_score) AS later_pain_mean,
                max(p.pain_score) AS later_pain_max,
                avg((p.pain_score >= 7)::INT) AS later_pain_ge7_share
            FROM first_response f
            INNER JOIN pain p
              ON p.stay_id = f.stay_id
             AND p.charttime > f.intime + INTERVAL 6 HOUR
             AND p.charttime <= f.intime + INTERVAL 24 HOUR
            GROUP BY f.stay_id
        ),
        later_opioid AS (
            SELECT stay_id, count(*) AS later_opioid_doses, sum(mme) AS later_mme_6_24h
            FROM doses
            WHERE icu_minute > 360 AND icu_minute <= 1440
            GROUP BY stay_id
        )
        SELECT
            f.*,
            lp.* EXCLUDE (stay_id),
            coalesce(lo.later_opioid_doses, 0) AS later_opioid_doses,
            coalesce(lo.later_mme_6_24h, 0) AS later_mme_6_24h,
            CASE WHEN lp.later_pain_count >= 2 AND lp.later_pain_ge7_share >= 0.5
                 THEN 1 ELSE 0 END AS persistent_severe_pain_6_24h
        FROM first_response f
        LEFT JOIN later_pain lp USING (stay_id)
        LEFT JOIN later_opioid lo USING (stay_id)
    ) TO '{sql_path(COHORT)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
)

df = con.execute("SELECT * FROM read_parquet(?)", [str(COHORT)]).fetchdf()
con.close()
df["male"] = (df["gender"] == "M").astype(int)
df["age"] = df["age"].where(df["age"].between(18, 100))
df["log_mme"] = np.log(df["mme"])
df["inadequate_response"] = df["inadequate_response"].astype(int)
analysis = df[df["later_pain_count"].ge(2)].dropna(subset=["age"]).copy()
formula = (
    "persistent_severe_pain_6_24h ~ inadequate_response + pre_score + log_mme + "
    "age + male + C(opioid) + C(first_careunit) + C(admission_type)"
)
result = smf.glm(formula, data=analysis, family=sm.families.Binomial()).fit(
    cov_type="cluster", cov_kwds={"groups": analysis["subject_id"]}
)
ci = result.conf_int().loc["inadequate_response"]
model = pd.DataFrame(
    [
        {
            "n": len(analysis),
            "events": int(analysis["persistent_severe_pain_6_24h"].sum()),
            "event_rate": analysis["persistent_severe_pain_6_24h"].mean(),
            "odds_ratio": np.exp(result.params["inadequate_response"]),
            "ci_low": np.exp(ci.iloc[0]),
            "ci_high": np.exp(ci.iloc[1]),
            "p_value": result.pvalues["inadequate_response"],
        }
    ]
)
summary = pd.DataFrame(
    [
        {
            "cohort_cases": len(df),
            "cohort_patients": df["subject_id"].nunique(),
            "pain_evaluable": len(analysis),
            "inadequate_response_rate": df["inadequate_response"].mean(),
        }
    ]
)
summary.to_csv(OUT / "external_cohort_summary.csv", index=False)
model.to_csv(OUT / "external_adjusted_association.csv", index=False)
print("MIMIC-IV EARLY ANALGESIC RESPONSE EXTERNAL VALIDATION COMPLETED")
print(summary.to_string(index=False))
print(model.to_string(index=False))
print(f"Cohort: {COHORT}")
print(f"Results: {OUT}")
