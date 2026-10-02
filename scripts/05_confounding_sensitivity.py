from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


COHORT = str(MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet")
HISTORY = str(MOVER_ROOT / "EPIC_EMR" / "EMR" / "patient_history.csv")
MEDS = str(MOVER_ROOT / "EPIC_EMR" / "EMR" / "patient_medications.csv")
OUT = RESULTS_ROOT / "pain_response_stage7_confounding"
OUT.mkdir(parents=True, exist_ok=True)


con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA preserve_insertion_order=false")
df = con.execute(
    """
    WITH cohort AS (
        SELECT * FROM read_parquet(?) WHERE later_pain_count >= 2
    ),
    history AS (
        SELECT
            mrn AS MRN,
            max(regexp_matches(lower(coalesce(dx_name,'')),
                'chronic pain|fibromyalgia|complex regional pain|reflex sympathetic|phantom limb|neuropathic pain|postlaminectomy|failed back syndrome')::INT
            ) AS chronic_pain_history,
            max(regexp_matches(lower(coalesce(dx_name,'')), 'opioid|opiate')::INT
            ) AS opioid_related_history
        FROM read_csv_auto(?, sample_size=500000, all_varchar=true)
        GROUP BY mrn
    ),
    medications AS (
        SELECT
            c.LOG_ID,
            max((regexp_matches(lower(coalesce(m.DISPLAY_NAME,'') || ' ' || coalesce(m.MEDICATION_NM,'')),
                'acetaminophen|paracetamol') AND try_cast(m.MED_ACTION_TIME AS TIMESTAMP) BETWEEN c.anesthesia_stop AND c.post_time)::INT
            ) AS acetaminophen_during_response,
            max((regexp_matches(lower(coalesce(m.DISPLAY_NAME,'') || ' ' || coalesce(m.MEDICATION_NM,'')),
                'ketorolac|ibuprofen|celecoxib|diclofenac|naproxen') AND try_cast(m.MED_ACTION_TIME AS TIMESTAMP) BETWEEN c.anesthesia_stop AND c.post_time)::INT
            ) AS nsaid_during_response,
            max((regexp_matches(lower(coalesce(m.DISPLAY_NAME,'') || ' ' || coalesce(m.MEDICATION_NM,'')),
                'gabapentin|pregabalin') AND try_cast(m.MED_ACTION_TIME AS TIMESTAMP) BETWEEN c.anesthesia_stop AND c.post_time)::INT
            ) AS gabapentinoid_during_response,
            max((regexp_matches(lower(coalesce(m.DISPLAY_NAME,'') || ' ' || coalesce(m.MEDICATION_NM,'')),
                'ketamine') AND try_cast(m.MED_ACTION_TIME AS TIMESTAMP) BETWEEN c.anesthesia_stop AND c.post_time)::INT
            ) AS ketamine_during_response,
            max((regexp_matches(lower(coalesce(m.DISPLAY_NAME,'') || ' ' || coalesce(m.MEDICATION_NM,'')),
                'dexmedetomidine') AND try_cast(m.MED_ACTION_TIME AS TIMESTAMP) BETWEEN c.anesthesia_stop AND c.post_time)::INT
            ) AS dexmedetomidine_during_response,
            max((
                regexp_matches(lower(coalesce(m.DISPLAY_NAME,'') || ' ' || coalesce(m.MEDICATION_NM,'')),
                    'morphine|hydromorphone|fentanyl|oxycodone|hydrocodone|tramadol|codeine|meperidine|methadone|buprenorphine')
                AND try_cast(m.START_DATE AS TIMESTAMP) < c.anesthesia_stop - INTERVAL 24 HOUR
                AND (try_cast(m.END_DATE AS TIMESTAMP) IS NULL OR try_cast(m.END_DATE AS TIMESTAMP) >= c.anesthesia_stop)
            )::INT) AS active_preoperative_opioid_order
        FROM cohort c
        LEFT JOIN read_csv_auto(?, sample_size=500000, all_varchar=true) m
          ON m.LOG_ID = c.LOG_ID
        GROUP BY c.LOG_ID
    )
    SELECT
        c.*,
        coalesce(h.chronic_pain_history,0) AS chronic_pain_history,
        coalesce(h.opioid_related_history,0) AS opioid_related_history,
        coalesce(m.acetaminophen_during_response,0) AS acetaminophen_during_response,
        coalesce(m.nsaid_during_response,0) AS nsaid_during_response,
        coalesce(m.gabapentinoid_during_response,0) AS gabapentinoid_during_response,
        coalesce(m.ketamine_during_response,0) AS ketamine_during_response,
        coalesce(m.dexmedetomidine_during_response,0) AS dexmedetomidine_during_response,
        coalesce(m.active_preoperative_opioid_order,0) AS active_preoperative_opioid_order
    FROM cohort c
    LEFT JOIN history h USING (MRN)
    LEFT JOIN medications m USING (LOG_ID)
    """,
    [COHORT, HISTORY, MEDS],
).fetchdf()
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
df["worse_response_per_point"] = df["pain_score_change"]
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
df["primary_share40"] = (df["later_pain_ge7_share"] >= 0.40).astype(int)
df["primary_share50"] = (df["later_pain_ge7_share"] >= 0.50).astype(int)
df["primary_share60"] = (df["later_pain_ge7_share"] >= 0.60).astype(int)
df["later_mean_ge4"] = (df["later_pain_mean"] >= 4).astype(int)
df["later_mean_ge7"] = (df["later_pain_mean"] >= 7).astype(int)
df["first_case_number"] = df.sort_values("anesthesia_stop").groupby("MRN").cumcount() + 1

base = (
    "worse_response_per_point + pre_score + mme + age + male + C(asa) + "
    "C(opioid) + C(surgery_group)"
)
extended = (
    base
    + " + chronic_pain_history + opioid_related_history + active_preoperative_opioid_order"
    + " + acetaminophen_during_response + nsaid_during_response"
    + " + gabapentinoid_during_response + ketamine_during_response"
    + " + dexmedetomidine_during_response"
)
multimodal_only = (
    base
    + " + acetaminophen_during_response + nsaid_during_response"
    + " + gabapentinoid_during_response + ketamine_during_response"
    + " + dexmedetomidine_during_response"
)


def fit(label, data, outcome, formula):
    use = data.dropna(subset=["age"]).copy()
    result = smf.glm(
        f"{outcome} ~ {formula}", data=use, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": use["MRN"]})
    ci = result.conf_int().loc["worse_response_per_point"]
    return {
        "analysis": label,
        "outcome": outcome,
        "n": len(use),
        "events": int(use[outcome].sum()),
        "event_rate": use[outcome].mean(),
        "or_per_1point_worse_response": np.exp(result.params["worse_response_per_point"]),
        "ci_low": np.exp(ci.iloc[0]),
        "ci_high": np.exp(ci.iloc[1]),
        "p_value": result.pvalues["worse_response_per_point"],
    }


rows = [
    fit("primary_base", df, "primary_share50", base),
    fit("primary_extended_confounders", df, "primary_share50", extended),
    fit("severe_share_40pct", df, "primary_share40", extended),
    fit("severe_share_60pct", df, "primary_share60", extended),
    fit("later_mean_pain_ge4", df, "later_mean_ge4", extended),
    fit("later_mean_pain_ge7", df, "later_mean_ge7", extended),
    fit("at_least_3_later_scores", df[df["later_pain_count"] >= 3], "primary_share50", extended),
    fit("first_case_per_patient", df[df["first_case_number"] == 1], "primary_share50", extended),
    fit(
        "exclude_chronic_pain_or_opioid_history",
        df[
            (df["chronic_pain_history"] == 0)
            & (df["opioid_related_history"] == 0)
            & (df["active_preoperative_opioid_order"] == 0)
        ],
        "primary_share50",
        multimodal_only,
    ),
]
results = pd.DataFrame(rows)
coverage = pd.DataFrame(
    [
        {
            "cases": len(df),
            "chronic_pain_history": int(df["chronic_pain_history"].sum()),
            "opioid_related_history": int(df["opioid_related_history"].sum()),
            "active_preoperative_opioid_order": int(df["active_preoperative_opioid_order"].sum()),
            "acetaminophen_during_response": int(df["acetaminophen_during_response"].sum()),
            "nsaid_during_response": int(df["nsaid_during_response"].sum()),
            "gabapentinoid_during_response": int(df["gabapentinoid_during_response"].sum()),
            "ketamine_during_response": int(df["ketamine_during_response"].sum()),
            "dexmedetomidine_during_response": int(df["dexmedetomidine_during_response"].sum()),
        }
    ]
)
coverage.to_csv(OUT / "confounder_coverage.csv", index=False)
results.to_csv(OUT / "confounding_and_outcome_sensitivities.csv", index=False)
print("MOVER CONFOUNDING AND OUTCOME SENSITIVITIES COMPLETED")
print("\nCOVERAGE")
print(coverage.to_string(index=False))
print("\nSENSITIVITY RESULTS")
print(results.to_string(index=False))
print(f"\nOutput: {OUT}")
