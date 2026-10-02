from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


PAIRS = str(MOVER_DERIVED / "postop_opioid_pain_response_pairs_stage2.parquet")
PAIN = str(MOVER_DERIVED / "postop_pain_scores_stage1.parquet")
DOSES = str(MOVER_DERIVED / "postop_opioid_doses_24h_stage2.parquet")
COMPLICATIONS = str(MOVER_ROOT / "EPIC_EMR" / "EMR" / "patient_post_op_complications.csv")
COHORT = MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet"
OUT = RESULTS_ROOT / "pain_response_stage3"
OUT.mkdir(parents=True, exist_ok=True)


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA preserve_insertion_order=false")

con.execute(
    f"""
    COPY (
        WITH first_response AS (
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT
                    p.*,
                    row_number() OVER (
                        PARTITION BY LOG_ID
                        ORDER BY administration_time, dose_id
                    ) AS rn
                FROM read_parquet(?) p
                WHERE intervening_opioid_doses = 0
                  AND pre_score >= 4
                  AND postop_minute BETWEEN 0 AND 360
                  AND post_time <= anesthesia_stop + INTERVAL 6 HOUR
                  AND opioid IN ('oxycodone', 'hydromorphone', 'hydrocodone',
                                 'fentanyl', 'morphine', 'tramadol')
            )
            WHERE rn = 1
        ),
        later_pain AS (
            SELECT
                f.LOG_ID,
                count(*) AS later_pain_count,
                avg(p.pain_score) AS later_pain_mean,
                max(p.pain_score) AS later_pain_max,
                avg((p.pain_score >= 4)::INT) AS later_pain_ge4_share,
                avg((p.pain_score >= 7)::INT) AS later_pain_ge7_share,
                sum((p.pain_score >= 7)::INT) AS later_severe_pain_records
            FROM first_response f
            INNER JOIN read_parquet(?) p
              ON p.LOG_ID = f.LOG_ID
             AND p.recorded_time > f.anesthesia_stop + INTERVAL 6 HOUR
             AND p.recorded_time <= f.anesthesia_stop + INTERVAL 24 HOUR
            GROUP BY f.LOG_ID
        ),
        later_opioid AS (
            SELECT
                LOG_ID,
                count(*) AS later_opioid_doses,
                sum(mme) AS later_mme_6_24h
            FROM read_parquet(?)
            WHERE postop_minute > 360 AND postop_minute <= 1440
            GROUP BY LOG_ID
        ),
        complications AS (
            SELECT
                LOG_ID,
                max(regexp_matches(
                    lower(coalesce(Element_Name,'') || ' ' || coalesce(Element_abbr,'') || ' ' ||
                          coalesce(SMRTDTA_ELEM_VALUE,'')),
                    'respiratory|airway|hypox|reintub|unplanned postoperative vent|respiratory failure|apnea'
                )::INT) AS respiratory_airway_complication,
                max(regexp_matches(lower(coalesce(SMRTDTA_ELEM_VALUE,'')),
                    'naloxone administration')::INT) AS naloxone_recorded,
                max(regexp_matches(lower(coalesce(SMRTDTA_ELEM_VALUE,'')),
                    'unplanned icu admission')::INT) AS unplanned_icu_recorded
            FROM read_csv_auto(?, sample_size=500000, all_varchar=true)
            GROUP BY LOG_ID
        )
        SELECT
            f.*,
            lp.later_pain_count,
            lp.later_pain_mean,
            lp.later_pain_max,
            lp.later_pain_ge4_share,
            lp.later_pain_ge7_share,
            lp.later_severe_pain_records,
            coalesce(lo.later_opioid_doses, 0) AS later_opioid_doses,
            coalesce(lo.later_mme_6_24h, 0) AS later_mme_6_24h,
            CASE WHEN lp.later_pain_count >= 2 AND lp.later_pain_ge7_share >= 0.5
                 THEN 1 ELSE 0 END AS persistent_severe_pain_6_24h,
            coalesce(c.respiratory_airway_complication, 0) AS respiratory_airway_complication,
            coalesce(c.naloxone_recorded, 0) AS naloxone_recorded,
            coalesce(c.unplanned_icu_recorded, 0) AS unplanned_icu_recorded
        FROM first_response f
        LEFT JOIN later_pain lp USING (LOG_ID)
        LEFT JOIN later_opioid lo USING (LOG_ID)
        LEFT JOIN complications c USING (LOG_ID)
    ) TO '{sql_path(COHORT)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """,
    [PAIRS, PAIN, DOSES, COMPLICATIONS],
)

df = con.execute("SELECT * FROM read_parquet(?)", [str(COHORT)]).fetchdf()
con.close()

asa_map = {
    "Healthy": 1,
    "Mild Systemic Disease": 2,
    "Severe Systemic Disease": 3,
    "Incapacitating Disease": 4,
    "Moribund": 5,
    "Brain Dead": 6,
}
df["asa"] = df["ASA_RATING"].map(asa_map).fillna(0).astype(int)
df["male"] = (df["SEX"].str.lower() == "male").astype(int)
df["age"] = df["age"].where(df["age"].between(18, 100))
procedure = df["PRIMARY_PROCEDURE_NM"].fillna("").str.lower()
conditions = [
    procedure.str.contains("cardiac|coronary|valve|cabg|heart", regex=True),
    procedure.str.contains("vascular|artery|aortic|aneurysm|endarter", regex=True),
    procedure.str.contains("crani|brain|spine|laminect|fusion", regex=True),
    procedure.str.contains("hip|knee|shoulder|orthop|fracture", regex=True),
    procedure.str.contains("colon|bowel|gastr|liver|pancrea|abdomen|hernia", regex=True),
    procedure.str.contains("neph|ureter|bladder|prostat|urolog", regex=True),
]
df["surgery_group"] = np.select(
    conditions,
    ["cardiac", "vascular", "neuro_spine", "orthopedic", "abdominal", "urology"],
    default="other",
)
df["inadequate_response"] = df["inadequate_response"].astype(int)
df["log_later_mme"] = np.log1p(df["later_mme_6_24h"])

q75 = float(df["later_mme_6_24h"].quantile(0.75))
df["high_later_mme"] = (df["later_mme_6_24h"] >= q75).astype(int)

summary = pd.DataFrame(
    {
        "metric": [
            "cases", "patients", "inadequate_first_response", "inadequate_response_rate",
            "later_pain_evaluable", "persistent_severe_pain", "persistent_severe_pain_rate",
            "high_later_mme_threshold", "respiratory_airway_complications",
            "naloxone_recorded", "unplanned_icu_recorded",
        ],
        "value": [
            len(df), df["MRN"].nunique(), df["inadequate_response"].sum(),
            df["inadequate_response"].mean(), df["later_pain_count"].ge(2).sum(),
            df["persistent_severe_pain_6_24h"].sum(),
            df.loc[df["later_pain_count"].ge(2), "persistent_severe_pain_6_24h"].mean(),
            q75, df["respiratory_airway_complication"].sum(),
            df["naloxone_recorded"].sum(), df["unplanned_icu_recorded"].sum(),
        ],
    }
)

base = "pre_score + mme + age + male + C(asa) + C(opioid) + C(surgery_group)"
models = []
for name, outcome, subset in [
    (
        "persistent_severe_pain",
        "persistent_severe_pain_6_24h",
        df[df["later_pain_count"].ge(2)].dropna(subset=["age"]),
    ),
    ("high_later_mme", "high_later_mme", df.dropna(subset=["age"])),
]:
    result = smf.glm(
        f"{outcome} ~ inadequate_response + {base}",
        data=subset,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": subset["MRN"]})
    ci = result.conf_int().loc["inadequate_response"]
    models.append(
        {
            "outcome": name,
            "n": len(subset),
            "events": int(subset[outcome].sum()),
            "odds_ratio": np.exp(result.params["inadequate_response"]),
            "ci_low": np.exp(ci.iloc[0]),
            "ci_high": np.exp(ci.iloc[1]),
            "p_value": result.pvalues["inadequate_response"],
        }
    )

linear_subset = df.dropna(subset=["age"])
linear = smf.ols(
    f"log_later_mme ~ inadequate_response + {base}", data=linear_subset
).fit(cov_type="cluster", cov_kwds={"groups": linear_subset["MRN"]})
ci = linear.conf_int().loc["inadequate_response"]
models.append(
    {
        "outcome": "log_later_mme_continuous",
        "n": len(linear_subset),
        "events": np.nan,
        "odds_ratio": np.exp(linear.params["inadequate_response"]),
        "ci_low": np.exp(ci.iloc[0]),
        "ci_high": np.exp(ci.iloc[1]),
        "p_value": linear.pvalues["inadequate_response"],
    }
)

models = pd.DataFrame(models)
summary.to_csv(OUT / "cohort_summary.csv", index=False)
models.to_csv(OUT / "adjusted_pilot_models.csv", index=False)

print("MOVER EARLY ANALGESIC RESPONSE STAGE 3 COMPLETED")
print("\nCOHORT")
print(summary.to_string(index=False))
print("\nADJUSTED PILOT MODELS")
print(models.to_string(index=False))
print(f"\nCohort: {COHORT}")
print(f"Results: {OUT}")
