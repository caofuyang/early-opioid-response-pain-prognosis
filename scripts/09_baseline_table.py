from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


COHORT = str(MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet")
OUT = RESULTS_ROOT / "manuscript" / "baseline"
OUT.mkdir(parents=True, exist_ok=True)


def n_pct(series: pd.Series, value) -> str:
    valid = series.notna()
    n = int((series[valid] == value).sum())
    denominator = int(valid.sum())
    return f"{n:,} ({100 * n / denominator:.1f}%)" if denominator else "0 (NA)"


def med_iqr(series: pd.Series, digits: int = 1) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return "NA"
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
    return f"{med:.{digits}f} ({q1:.{digits}f}-{q3:.{digits}f})"


def format_missing(series: pd.Series) -> str:
    n = int(series.isna().sum())
    return f"{n:,} ({100 * n / len(series):.1f}%)"


con = duckdb.connect()
df = con.execute("SELECT * FROM read_parquet(?)", [COHORT]).fetchdf()
con.close()

# The primary analysis requires at least two pain observations in the 6-24 h window.
analysis = df.loc[df["later_pain_count"].ge(2)].copy()
analysis["outcome"] = analysis["persistent_severe_pain_6_24h"].astype(int)
analysis["age_eligible"] = analysis["age"].where(analysis["age"].between(18, 100))
analysis["female"] = analysis["SEX"].str.strip().str.lower().eq("female")
analysis["asa_group"] = analysis["ASA_RATING"].map(
    {
        "Healthy": "ASA I",
        "Mild Systemic Disease": "ASA II",
        "Severe Systemic Disease": "ASA III",
        "Incapacitating Disease": "ASA IV",
        "Moribund": "ASA V",
        "Brain Dead": "ASA VI",
    }
).fillna("Missing/other")

procedure = analysis["PRIMARY_PROCEDURE_NM"].fillna("").str.lower()
conditions = [
    procedure.str.contains("cardiac|coronary|valve|cabg|heart", regex=True),
    procedure.str.contains("vascular|artery|aortic|aneurysm|endarter", regex=True),
    procedure.str.contains("crani|brain|spine|laminect|fusion", regex=True),
    procedure.str.contains("hip|knee|shoulder|orthop|fracture", regex=True),
    procedure.str.contains("colon|bowel|gastr|liver|pancrea|abdomen|hernia", regex=True),
    procedure.str.contains("neph|ureter|bladder|prostat|urolog", regex=True),
]
analysis["surgery_group"] = np.select(
    conditions,
    ["Cardiac", "Vascular", "Neurosurgery/spine", "Orthopedic", "Abdominal", "Urology"],
    default="Other",
)
analysis["anesthesia_group"] = (
    analysis["PRIMARY_ANES_TYPE_NM"].fillna("Missing").str.strip().replace("", "Missing")
)
analysis["icu_admission"] = (
    analysis["ICU_ADMIN_FLAG"].astype("string").str.strip().str.lower().isin(["1", "y", "yes", "true"])
)

groups = {
    "Overall (n=6,038)": analysis,
    "No persistent severe pain (n=4,083)": analysis.loc[analysis.outcome.eq(0)],
    "Persistent severe pain (n=1,955)": analysis.loc[analysis.outcome.eq(1)],
}

rows = []


def add_cont(label: str, field: str, digits: int = 1):
    row = {"Characteristic": label}
    for name, part in groups.items():
        row[name] = med_iqr(part[field], digits)
    row["Missing, n (%)"] = format_missing(analysis[field])
    rows.append(row)


def add_cat_header(label: str):
    rows.append({"Characteristic": label, **{name: "" for name in groups}, "Missing, n (%)": ""})


def add_cat(label: str, field: str, value):
    row = {"Characteristic": f"  {label}"}
    for name, part in groups.items():
        row[name] = n_pct(part[field], value)
    row["Missing, n (%)"] = ""
    rows.append(row)


add_cont("Age, years", "age_eligible", 1)
add_cat_header("Sex")
add_cat("Female", "female", True)
add_cat("Male", "female", False)
add_cat_header("ASA physical status")
for level in ["ASA I", "ASA II", "ASA III", "ASA IV", "ASA V", "ASA VI", "Missing/other"]:
    add_cat(level, "asa_group", level)
add_cat_header("Surgery group")
for level in ["Cardiac", "Vascular", "Neurosurgery/spine", "Orthopedic", "Abdominal", "Urology", "Other"]:
    add_cat(level, "surgery_group", level)
add_cat_header("Primary anesthesia type")
for level in ["General", "Regional", "Monitored Anesthesia Care (MAC)", "Epidural", "Spinal", "Missing"]:
    display = level.strip() if level != "Missing" else level
    add_cat(display, "anesthesia_group", level)
add_cat_header("Index opioid")
for level in ["oxycodone", "hydromorphone", "hydrocodone", "fentanyl", "morphine", "tramadol"]:
    add_cat(level.capitalize(), "opioid", level)
add_cont("Index opioid dose, MME", "mme", 1)
add_cont("Pre-dose pain score", "pre_score", 1)
add_cont("Post-dose pain score", "post_score", 1)
add_cont("Pain-score change (post minus pre)", "pain_score_change", 1)
add_cont("Minutes after anesthesia end at index dose", "postop_minute", 0)
add_cont("Minutes from dose to post-dose pain score", "minutes_to_post_score", 0)
add_cont("Pain assessments during 6-24 h, n", "later_pain_count", 0)
add_cont("Hospital length of stay, days", "hospital_los_days", 1)
add_cat_header("ICU admission recorded")
add_cat("Yes", "icu_admission", True)
add_cat("No", "icu_admission", False)

table = pd.DataFrame(rows)
table.to_csv(OUT / "mover_baseline_characteristics.csv", index=False)

profile_rows = []
for column in analysis.columns:
    s = analysis[column]
    profile_rows.append(
        {
            "column": column,
            "dtype": str(s.dtype),
            "rows": len(s),
            "missing_n": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 3),
            "distinct_n": int(s.nunique(dropna=True)),
        }
    )
pd.DataFrame(profile_rows).to_csv(OUT / "mover_analysis_cohort_profile.csv", index=False)

key_checks = pd.DataFrame(
    [
        {"check": "source cohort rows", "value": len(df)},
        {"check": "analysis cohort rows", "value": len(analysis)},
        {"check": "analysis unique LOG_ID", "value": analysis.LOG_ID.nunique()},
        {"check": "analysis duplicate LOG_ID rows", "value": int(analysis.LOG_ID.duplicated().sum())},
        {"check": "analysis unique dose_id", "value": analysis.dose_id.nunique()},
        {"check": "analysis duplicate dose_id rows", "value": int(analysis.dose_id.duplicated().sum())},
        {"check": "analysis unique patients", "value": analysis.MRN.nunique()},
        {"check": "primary outcome events", "value": int(analysis.outcome.sum())},
        {"check": "age missing in source", "value": int(analysis.age.isna().sum())},
        {"check": "age outside frozen 18-100 range", "value": int(analysis.age_eligible.isna().sum())},
        {"check": "pain scores outside 0-10", "value": int((~analysis.pre_score.between(0, 10) | ~analysis.post_score.between(0, 10)).sum())},
        {"check": "negative MME", "value": int(analysis.mme.lt(0).sum())},
        {"check": "post-dose score precedes administration", "value": int(analysis.minutes_to_post_score.lt(0).sum())},
        {"check": "intervening opioid doses >0", "value": int(analysis.intervening_opioid_doses.gt(0).sum())},
    ]
)
key_checks.to_csv(OUT / "mover_key_quality_checks.csv", index=False)

with (OUT / "baseline_data_quality_report.md").open("w", encoding="utf-8") as fh:
    fh.write("# MOVER baseline-table data quality report\n\n")
    fh.write("## Dataset and grain\n\n")
    fh.write(f"The source cohort contains {len(df):,} first eligible administration records, one intended record per surgical case. ")
    fh.write(f"The primary outcome-evaluable cohort contains {len(analysis):,} cases from {analysis.MRN.nunique():,} patients.\n\n")
    fh.write("## Key checks\n\n")
    fh.write("| Check | Value |\n|---|---:|\n")
    for item in key_checks.itertuples(index=False):
        fh.write(f"| {item.check} | {item.value} |\n")
    fh.write("\n\n## Interpretation\n\n")
    fh.write("The analysis cohort preserves the intended case-level grain if duplicate LOG_ID and dose_id counts are zero. ")
    fh.write("The primary table reports medians and interquartile ranges for continuous variables and counts with column percentages for categorical variables. ")
    fh.write("Percentages use nonmissing values as denominators; missingness is reported separately. No hypothesis-test P values are used for descriptive baseline comparisons.\n")

print(table.to_string(index=False))
print("\nKEY CHECKS")
print(key_checks.to_string(index=False))
