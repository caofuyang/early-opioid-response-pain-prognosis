from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COHORT = str(MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet")
OUT = RESULTS_ROOT / "pain_response_stage4"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260827)


con = duckdb.connect()
df = con.execute(
    """
    SELECT *
    FROM read_parquet(?)
    WHERE later_pain_count >= 2
      AND year(anesthesia_stop) BETWEEN 2018 AND 2021
    """,
    [COHORT],
).fetchdf()
con.close()

asa_map = {
    "Healthy": "ASA1",
    "Mild Systemic Disease": "ASA2",
    "Severe Systemic Disease": "ASA3",
    "Incapacitating Disease": "ASA4",
    "Moribund": "ASA5",
    "Brain Dead": "ASA6",
}
df["asa_group"] = df["ASA_RATING"].map(asa_map).fillna("Unknown")
df["sex_group"] = df["SEX"].fillna("Unknown")
df["surgery_year"] = pd.to_datetime(df["anesthesia_stop"]).dt.year

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

development = df[df["surgery_year"] <= 2019].copy()
validation = df[df["surgery_year"] == 2020].copy()
locked = df[df["surgery_year"] == 2021].copy()

categorical = ["sex_group", "asa_group", "opioid", "surgery_group"]
baseline_numeric = ["age", "pre_score", "mme", "postop_minute"]
feature_sets = {
    "pretreatment_baseline": categorical + baseline_numeric,
    "response_indicator": categorical + baseline_numeric + ["inadequate_response"],
    "continuous_response": categorical + baseline_numeric + ["pain_score_change"],
}
outcome = "persistent_severe_pain_6_24h"


def make_model(features):
    cats = [c for c in categorical if c in features]
    nums = [c for c in features if c not in cats]
    preprocess = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cats,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                nums,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )


def calibration(y, p):
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    x = logit(clipped)
    design = np.column_stack([np.ones(len(x)), x])
    # Stable Newton iterations for logistic recalibration.
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = expit(design @ beta)
        w = np.clip(mu * (1 - mu), 1e-8, None)
        hessian = design.T @ (design * w[:, None])
        score = design.T @ (y - mu)
        step = np.linalg.solve(hessian, score)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta[0], beta[1]


predictions = validation[["LOG_ID", "MRN", outcome]].copy()
rows = []
for name, features in feature_sets.items():
    model = make_model(features)
    model.fit(development[features], development[outcome].astype(int))
    pred = model.predict_proba(validation[features])[:, 1]
    predictions[name] = pred
    intercept, slope = calibration(validation[outcome].to_numpy(), pred)
    rows.append(
        {
            "model": name,
            "development_n": len(development),
            "validation_n": len(validation),
            "validation_events": int(validation[outcome].sum()),
            "prevalence": validation[outcome].mean(),
            "auroc": roc_auc_score(validation[outcome], pred),
            "auprc": average_precision_score(validation[outcome], pred),
            "brier": brier_score_loss(validation[outcome], pred),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }
    )

metrics = pd.DataFrame(rows)

# Patient-cluster bootstrap for the prespecified comparison.
predictions = predictions.reset_index(drop=True)
patients = predictions["MRN"].dropna().unique()
patient_rows = {
    patient: np.flatnonzero(predictions["MRN"].to_numpy() == patient)
    for patient in patients
}
boot = []
for _ in range(1000):
    sampled = RNG.choice(patients, size=len(patients), replace=True)
    idx = np.concatenate([patient_rows[patient] for patient in sampled])
    y = predictions[outcome].to_numpy(dtype=int)[idx]
    if np.unique(y).size < 2:
        continue
    base = predictions["pretreatment_baseline"].to_numpy()[idx]
    response = predictions["response_indicator"].to_numpy()[idx]
    continuous = predictions["continuous_response"].to_numpy()[idx]
    boot.append(
        {
            "auroc_difference": roc_auc_score(y, response) - roc_auc_score(y, base),
            "auprc_difference": average_precision_score(y, response)
            - average_precision_score(y, base),
            "brier_improvement": brier_score_loss(y, base)
            - brier_score_loss(y, response),
            "continuous_auroc_difference": roc_auc_score(y, continuous)
            - roc_auc_score(y, base),
            "continuous_auprc_difference": average_precision_score(y, continuous)
            - average_precision_score(y, base),
            "continuous_brier_improvement": brier_score_loss(y, base)
            - brier_score_loss(y, continuous),
        }
    )
boot = pd.DataFrame(boot)
point = metrics.set_index("model")
differences = []
for metric, estimate in {
    "auroc_difference": point.loc["response_indicator", "auroc"]
    - point.loc["pretreatment_baseline", "auroc"],
    "auprc_difference": point.loc["response_indicator", "auprc"]
    - point.loc["pretreatment_baseline", "auprc"],
    "brier_improvement": point.loc["pretreatment_baseline", "brier"]
    - point.loc["response_indicator", "brier"],
    "continuous_auroc_difference": point.loc["continuous_response", "auroc"]
    - point.loc["pretreatment_baseline", "auroc"],
    "continuous_auprc_difference": point.loc["continuous_response", "auprc"]
    - point.loc["pretreatment_baseline", "auprc"],
    "continuous_brier_improvement": point.loc["pretreatment_baseline", "brier"]
    - point.loc["continuous_response", "brier"],
}.items():
    differences.append(
        {
            "metric": metric,
            "estimate": estimate,
            "ci_low": boot[metric].quantile(0.025),
            "ci_high": boot[metric].quantile(0.975),
        }
    )
differences = pd.DataFrame(differences)

metrics.to_csv(OUT / "temporal_validation_metrics.csv", index=False)
differences.to_csv(OUT / "paired_increment_bootstrap.csv", index=False)
predictions.to_parquet(OUT / "validation_predictions.parquet", index=False)

print("MOVER ANALGESIC RESPONSE INCREMENT VALIDATION COMPLETED")
print(
    f"Development 2018-2019: n={len(development)}, events={int(development[outcome].sum())}; "
    f"Validation 2020: n={len(validation)}, events={int(validation[outcome].sum())}; "
    f"Locked 2021: n={len(locked)}, events={int(locked[outcome].sum())}"
)
print("\nTEMPORAL VALIDATION")
print(metrics.to_string(index=False))
print("\nPAIRED INCREMENT WITH PATIENT-CLUSTER BOOTSTRAP")
print(differences.to_string(index=False))
print("\nThe 2021 cohort remains locked and unevaluated.")
print(f"Output: {OUT}")
