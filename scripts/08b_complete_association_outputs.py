"""Repair and complete association outputs required by the manuscript figures.

Run after scripts 01-08 from the reproducibility package.  This script:
1. estimates the prespecified continuous-response association in MIMIC-IV;
2. retains the binary inadequate-response association as a secondary analysis;
3. reconstructs the MIMIC-IV surgical-service subgroup using services.csv.gz;
4. estimates MOVER overall and opioid-specific continuous-response associations;
5. estimates the MOVER anesthesia-type sensitivity model; and
6. writes the three CSV files required by 11_main_figures.py.
"""

from config import MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


MOVER_COHORT = str(MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet")
MIMIC_COHORT = str(MIMIC_DERIVED / "early_analgesic_response_external_stage2.parquet")
MIMIC_SERVICES = str(MIMIC_ROOT / "hosp" / "services.csv.gz")

OUT_CONFOUNDING = RESULTS_ROOT / "pain_response_stage7_confounding"
OUT_CROSS = RESULTS_ROOT / "pain_response_stage6_sensitivity"
OUT_MIMIC = RESULTS_ROOT / "mimic_pain_response_external"
for path in (OUT_CONFOUNDING, OUT_CROSS, OUT_MIMIC):
    path.mkdir(parents=True, exist_ok=True)

OUTCOME = "persistent_severe_pain_6_24h"
SURGICAL_SERVICES = {
    "NSURG",
    "SURG",
    "TSURG",
    "CSURG",
    "VSURG",
    "ORTHO",
    "ENT",
    "PSURG",
    "GYN",
    "GU",
}


def fit_clustered_logistic(label, data, exposure, formula, cluster):
    use = data.dropna(subset=["age", exposure, OUTCOME, cluster]).copy()
    result = smf.glm(
        f"{OUTCOME} ~ {formula}",
        data=use,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": use[cluster]})
    ci = result.conf_int().loc[exposure]
    return {
        "analysis": label,
        "n": len(use),
        "events": int(use[OUTCOME].sum()),
        "event_rate": float(use[OUTCOME].mean()),
        "estimate": float(np.exp(result.params[exposure])),
        "ci_low": float(np.exp(ci.iloc[0])),
        "ci_high": float(np.exp(ci.iloc[1])),
        "p_value": float(result.pvalues[exposure]),
    }


def prepare_mover():
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT * FROM read_parquet(?)
        WHERE later_pain_count >= 2
        """,
        [MOVER_COHORT],
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
    df["male"] = (df["SEX"].fillna("").str.lower() == "male").astype(int)
    df["age"] = df["age"].where(df["age"].between(18, 100))
    df["worse_response_per_point"] = df["pain_score_change"]
    df["anesthesia_type"] = (
        df["PRIMARY_ANES_TYPE_NM"].fillna("Unknown").str.strip().replace("", "Unknown")
    )

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
    return df


def prepare_mimic_with_service():
    con = duckdb.connect()
    df = con.execute(
        """
        WITH cohort AS (
            SELECT * FROM read_parquet(?)
            WHERE later_pain_count >= 2
        ),
        services AS (
            SELECT
                subject_id::BIGINT AS subject_id,
                hadm_id::BIGINT AS hadm_id,
                try_cast(transfertime AS TIMESTAMP) AS transfertime,
                curr_service
            FROM read_csv_auto(?, sample_size=500000, all_varchar=true)
        )
        SELECT c.*, s.curr_service
        FROM cohort c
        ASOF LEFT JOIN services s
          ON c.subject_id = s.subject_id
         AND c.hadm_id = s.hadm_id
         AND c.intime >= s.transfertime
        """,
        [MIMIC_COHORT, MIMIC_SERVICES],
    ).fetchdf()
    con.close()
    df["male"] = (df["gender"].fillna("") == "M").astype(int)
    df["age"] = df["age"].where(df["age"].between(18, 100))
    df["log_mme"] = np.log(df["mme"])
    df["worse_response_per_point"] = df["pain_score_change"]
    df["inadequate_response"] = df["inadequate_response"].astype(int)
    for column in ["opioid", "first_careunit", "admission_type", "curr_service"]:
        df[column] = df[column].fillna("Unknown").astype(str)
    return df


mover = prepare_mover()
mover_base = (
    "worse_response_per_point + pre_score + mme + age + male + "
    "C(asa) + C(opioid) + C(surgery_group)"
)
mover_no_drug = (
    "worse_response_per_point + pre_score + mme + age + male + "
    "C(asa) + C(surgery_group)"
)

mover_rows = [
    fit_clustered_logistic(
        "MOVER overall", mover, "worse_response_per_point", mover_base, "MRN"
    )
]
for opioid in ["oxycodone", "hydromorphone", "hydrocodone", "fentanyl", "morphine"]:
    mover_rows.append(
        fit_clustered_logistic(
            f"MOVER {opioid}",
            mover[mover["opioid"] == opioid],
            "worse_response_per_point",
            mover_no_drug,
            "MRN",
        )
    )

anesthesia = fit_clustered_logistic(
    "MOVER anesthesia-type sensitivity",
    mover,
    "worse_response_per_point",
    mover_base + " + C(anesthesia_type)",
    "MRN",
)
pd.DataFrame(
    [
        {
            "n": anesthesia["n"],
            "events": anesthesia["events"],
            "or_per_point": anesthesia["estimate"],
            "ci_low": anesthesia["ci_low"],
            "ci_high": anesthesia["ci_high"],
            "p_value": anesthesia["p_value"],
        }
    ]
).to_csv(OUT_CONFOUNDING / "anesthesia_type_sensitivity.csv", index=False)

mimic = prepare_mimic_with_service()
mimic_continuous_formula = (
    "worse_response_per_point + pre_score + log_mme + age + male + "
    "C(opioid) + C(first_careunit) + C(admission_type)"
)
mimic_binary_formula = (
    "inadequate_response + pre_score + log_mme + age + male + "
    "C(opioid) + C(first_careunit) + C(admission_type)"
)

mimic_overall = fit_clustered_logistic(
    "MIMIC-IV overall",
    mimic,
    "worse_response_per_point",
    mimic_continuous_formula,
    "subject_id",
)
mimic_binary = fit_clustered_logistic(
    "MIMIC-IV binary inadequate response (secondary)",
    mimic,
    "inadequate_response",
    mimic_binary_formula,
    "subject_id",
)

surgical = mimic[mimic["curr_service"].isin(SURGICAL_SERVICES)].copy()
mimic_surgical = fit_clustered_logistic(
    "MIMIC-IV surgical service",
    surgical,
    "worse_response_per_point",
    mimic_continuous_formula,
    "subject_id",
)

if (mimic_surgical["n"], mimic_surgical["events"]) != (718, 206):
    raise RuntimeError(
        "Surgical-service cohort mismatch: expected n=718/events=206, "
        f"observed n={mimic_surgical['n']}/events={mimic_surgical['events']}"
    )

association_models = pd.DataFrame([mimic_overall, mimic_binary])
association_models.to_csv(OUT_MIMIC / "external_association_models.csv", index=False)

# Preserve the historical filename for the prespecified continuous-response primary model.
pd.DataFrame(
    [
        {
            "n": mimic_overall["n"],
            "events": mimic_overall["events"],
            "event_rate": mimic_overall["event_rate"],
            "odds_ratio": mimic_overall["estimate"],
            "ci_low": mimic_overall["ci_low"],
            "ci_high": mimic_overall["ci_high"],
            "p_value": mimic_overall["p_value"],
        }
    ]
).to_csv(OUT_MIMIC / "external_adjusted_association.csv", index=False)

pd.DataFrame(
    [
        {
            "n": mimic_binary["n"],
            "events": mimic_binary["events"],
            "event_rate": mimic_binary["event_rate"],
            "odds_ratio": mimic_binary["estimate"],
            "ci_low": mimic_binary["ci_low"],
            "ci_high": mimic_binary["ci_high"],
            "p_value": mimic_binary["p_value"],
        }
    ]
).to_csv(OUT_MIMIC / "external_binary_association_secondary.csv", index=False)

pd.DataFrame(
    [
        {
            "n": mimic_surgical["n"],
            "events": mimic_surgical["events"],
            "or_per_point": mimic_surgical["estimate"],
            "ci_low": mimic_surgical["ci_low"],
            "ci_high": mimic_surgical["ci_high"],
            "p_value": mimic_surgical["p_value"],
            "included_services": ",".join(sorted(SURGICAL_SERVICES)),
        }
    ]
).to_csv(OUT_MIMIC / "surgical_service_sensitivity.csv", index=False)

cross_rows = mover_rows[:1] + [mimic_overall] + mover_rows[1:]
cross = pd.DataFrame(
    [
        {
            "analysis": row["analysis"],
            "n": row["n"],
            "events": row["events"],
            "or_per_1point_worse_response": row["estimate"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "p_value": row["p_value"],
        }
        for row in cross_rows
    ]
)
cross.to_csv(OUT_CROSS / "continuous_effect_cross_database_and_drugs.csv", index=False)

print("COMPLETED ASSOCIATION OUTPUT REPAIR")
print("\nMOVER AND CROSS-DATABASE CONTINUOUS EFFECTS")
print(cross.to_string(index=False))
print("\nMOVER ANESTHESIA-TYPE SENSITIVITY")
print(pd.DataFrame([anesthesia]).to_string(index=False))
print("\nMIMIC-IV ASSOCIATION MODELS")
print(association_models.to_string(index=False))
print("\nMIMIC-IV SURGICAL-SERVICE SUBGROUP")
print(pd.DataFrame([mimic_surgical]).to_string(index=False))
print(f"\nOutputs: {OUT_CONFOUNDING}; {OUT_CROSS}; {OUT_MIMIC}")
