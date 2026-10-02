from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COHORT = str(MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet")
OUT = RESULTS_ROOT / "pain_response_stage5_locked"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260827)
OUTCOME = "persistent_severe_pain_6_24h"


con = duckdb.connect()
df = con.execute(
    """
    SELECT * FROM read_parquet(?)
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

development = df[df["surgery_year"] <= 2020].copy()
test = df[df["surgery_year"] == 2021].copy()
categorical = ["sex_group", "asa_group", "opioid", "surgery_group"]
baseline_numeric = ["age", "pre_score", "mme", "postop_minute"]
feature_sets = {
    "pretreatment_baseline": categorical + baseline_numeric,
    "continuous_response": categorical + baseline_numeric + ["pain_score_change"],
}


def make_model(features):
    cats = [c for c in categorical if c in features]
    nums = [c for c in features if c not in cats]
    transformer = ColumnTransformer(
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
            ("preprocess", transformer),
            ("model", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )


def calibration(y, p):
    x = logit(np.clip(p, 1e-6, 1 - 1e-6))
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = expit(design @ beta)
        weights = np.clip(mu * (1 - mu), 1e-8, None)
        hessian = design.T @ (design * weights[:, None])
        score = design.T @ (y - mu)
        step = np.linalg.solve(hessian, score)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta[0], beta[1]


models = {}
predictions = test[["LOG_ID", "MRN", OUTCOME]].reset_index(drop=True).copy()
rows = []
for name, features in feature_sets.items():
    model = make_model(features)
    model.fit(development[features], development[OUTCOME].astype(int))
    pred = model.predict_proba(test[features])[:, 1]
    predictions[name] = pred
    intercept, slope = calibration(test[OUTCOME].to_numpy(dtype=int), pred)
    rows.append(
        {
            "model": name,
            "development_n": len(development),
            "test_n": len(test),
            "test_events": int(test[OUTCOME].sum()),
            "test_prevalence": test[OUTCOME].mean(),
            "auroc": roc_auc_score(test[OUTCOME], pred),
            "auprc": average_precision_score(test[OUTCOME], pred),
            "brier": brier_score_loss(test[OUTCOME], pred),
            "log_loss": log_loss(test[OUTCOME], pred),
            "mean_predicted_risk": pred.mean(),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }
    )
    models[name] = model
metrics = pd.DataFrame(rows)

# Patient-cluster bootstrap, preserving repeated surgical cases per patient.
patients = predictions["MRN"].dropna().unique()
patient_rows = {
    patient: np.flatnonzero(predictions["MRN"].to_numpy() == patient)
    for patient in patients
}
boot = []
for _ in range(2000):
    sampled = RNG.choice(patients, size=len(patients), replace=True)
    idx = np.concatenate([patient_rows[p] for p in sampled])
    y = predictions[OUTCOME].to_numpy(dtype=int)[idx]
    if np.unique(y).size < 2:
        continue
    base = predictions["pretreatment_baseline"].to_numpy()[idx]
    response = predictions["continuous_response"].to_numpy()[idx]
    boot.append(
        {
            "baseline_auroc": roc_auc_score(y, base),
            "response_auroc": roc_auc_score(y, response),
            "auroc_difference": roc_auc_score(y, response) - roc_auc_score(y, base),
            "baseline_auprc": average_precision_score(y, base),
            "response_auprc": average_precision_score(y, response),
            "auprc_difference": average_precision_score(y, response)
            - average_precision_score(y, base),
            "brier_improvement": brier_score_loss(y, base)
            - brier_score_loss(y, response),
        }
    )
boot = pd.DataFrame(boot)
point = metrics.set_index("model")
estimates = {
    "baseline_auroc": point.loc["pretreatment_baseline", "auroc"],
    "response_auroc": point.loc["continuous_response", "auroc"],
    "auroc_difference": point.loc["continuous_response", "auroc"]
    - point.loc["pretreatment_baseline", "auroc"],
    "baseline_auprc": point.loc["pretreatment_baseline", "auprc"],
    "response_auprc": point.loc["continuous_response", "auprc"],
    "auprc_difference": point.loc["continuous_response", "auprc"]
    - point.loc["pretreatment_baseline", "auprc"],
    "brier_improvement": point.loc["pretreatment_baseline", "brier"]
    - point.loc["continuous_response", "brier"],
}
cis = pd.DataFrame(
    [
        {
            "metric": metric,
            "estimate": estimate,
            "ci_low": boot[metric].quantile(0.025),
            "ci_high": boot[metric].quantile(0.975),
        }
        for metric, estimate in estimates.items()
    ]
)

# Decision-curve analysis for the response model versus baseline and default strategies.
y = predictions[OUTCOME].to_numpy(dtype=int)
dca_rows = []
for threshold in np.arange(0.10, 0.61, 0.01):
    odds = threshold / (1 - threshold)
    for name in ["pretreatment_baseline", "continuous_response"]:
        positive = predictions[name].to_numpy() >= threshold
        tp = np.sum(positive & (y == 1))
        fp = np.sum(positive & (y == 0))
        net_benefit = tp / len(y) - fp / len(y) * odds
        dca_rows.append({"threshold": threshold, "model": name, "net_benefit": net_benefit})
    dca_rows.append(
        {
            "threshold": threshold,
            "model": "treat_all",
            "net_benefit": np.mean(y) - (1 - np.mean(y)) * odds,
        }
    )
    dca_rows.append({"threshold": threshold, "model": "treat_none", "net_benefit": 0.0})
dca = pd.DataFrame(dca_rows)

metrics.to_csv(OUT / "locked_2021_metrics.csv", index=False)
cis.to_csv(OUT / "locked_2021_bootstrap_ci.csv", index=False)
dca.to_csv(OUT / "locked_2021_decision_curve.csv", index=False)
predictions.to_parquet(OUT / "locked_2021_predictions.parquet", index=False)
joblib.dump(models, OUT / "locked_2021_models.joblib")

print("MOVER LOCKED 2021 PAIN-RESPONSE VALIDATION COMPLETED")
print(f"Development 2018-2020: n={len(development)}, events={int(development[OUTCOME].sum())}")
print(f"Locked test 2021: n={len(test)}, events={int(test[OUTCOME].sum())}")
print("\nLOCKED TEST METRICS")
print(metrics.to_string(index=False))
print("\nPATIENT-CLUSTER BOOTSTRAP 95% CI")
print(cis.to_string(index=False))
print(f"\nOutput: {OUT}")
