from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path
from math import erf, sqrt

import duckdb
import numpy as np
import pandas as pd


COHORT = str(MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet")
PREDICTIONS = str(RESULTS_ROOT / "pain_response_stage5_locked" / "locked_2021_predictions.parquet")
OUT = RESULTS_ROOT / "manuscript" / "submission_statistical_checks"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260827


def normal_two_sided_p(z):
    return 1.0 - erf(abs(float(z)) / sqrt(2.0))


def logistic_fit(X, y, clusters, max_iter=200, tolerance=1e-8):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        hessian = X.T @ (X * w[:, None])
        step = np.linalg.pinv(hessian) @ (X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < tolerance:
            break
    else:
        if np.max(np.abs(X.T @ (y - p))) > 1e-5:
            raise RuntimeError(f"logistic regression did not converge; max step={np.max(np.abs(step)):.3g}")

    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1.0 - p), 1e-9, None)
    bread = np.linalg.pinv(X.T @ (X * w[:, None]))
    scores = X * (y - p)[:, None]
    grouped = pd.DataFrame(scores).assign(cluster=np.asarray(clusters)).groupby("cluster", sort=False).sum().to_numpy()
    meat = grouped.T @ grouped
    n, k = X.shape
    g = len(grouped)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = correction * bread @ meat @ bread
    return beta, covariance, p


def rcs_basis(x, knots):
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    last = knots[-1]
    penultimate = knots[-2]
    scale = (last - knots[0]) ** 2
    columns = []
    for knot in knots[:-2]:
        term = (
            np.maximum(x - knot, 0) ** 3
            - np.maximum(x - penultimate, 0) ** 3 * (last - knot) / (last - penultimate)
            + np.maximum(x - last, 0) ** 3 * (penultimate - knot) / (last - penultimate)
        ) / scale
        columns.append(term)
    return np.column_stack(columns)


def wald_joint(beta, covariance, indices):
    values = beta[indices]
    subcov = covariance[np.ix_(indices, indices)]
    statistic = float(values.T @ np.linalg.solve(subcov, values))
    # With two nonlinear terms, chi-square survival is exp(-x/2).
    if len(indices) != 2:
        raise RuntimeError("this implementation expects two nonlinear spline terms")
    return statistic, float(np.exp(-statistic / 2.0))


con = duckdb.connect()
df = con.execute(
    "SELECT * FROM read_parquet(?) WHERE later_pain_count >= 2 AND age BETWEEN 18 AND 100",
    [COHORT],
).fetchdf()
pred = con.execute("SELECT * FROM read_parquet(?)", [PREDICTIONS]).fetchdf()
con.close()

asa_map = {
    "Healthy": "1",
    "Mild Systemic Disease": "2",
    "Severe Systemic Disease": "3",
    "Incapacitating Disease": "4",
    "Moribund": "5",
    "Brain Dead": "6",
}
df["asa"] = df["ASA_RATING"].map(asa_map).fillna("unknown")
df["male"] = (df["SEX"].fillna("").str.lower() == "male").astype(int)
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

r = df["pain_score_change"].to_numpy(float)
r_centered = r - r.mean()
y = df["persistent_severe_pain_6_24h"].to_numpy(float)
continuous = pd.DataFrame(
    {
        "pre_score": df["pre_score"].astype(float),
        "mme": df["mme"].astype(float),
        "age": df["age"].astype(float),
        "male": df["male"].astype(float),
    }
)
categorical = pd.get_dummies(
    df[["asa", "opioid", "surgery_group"]].fillna("unknown").astype(str),
    drop_first=True,
    dtype=float,
)
base = pd.concat([continuous.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
base = base.fillna(base.median(numeric_only=True))
for column in continuous.columns:
    base[column] = (base[column] - base[column].mean()) / base[column].std(ddof=0)
base = base.loc[:, base.std(ddof=0) > 0]
base_matrix = base.to_numpy(float)
intercept = np.ones((len(df), 1))

X_linear = np.column_stack([intercept, r_centered, base_matrix])
beta_linear, cov_linear, _ = logistic_fit(X_linear, y, df["MRN"])
linear_se = sqrt(cov_linear[1, 1])

X_quadratic = np.column_stack([intercept, r_centered, r_centered**2, base_matrix])
beta_quadratic, cov_quadratic, _ = logistic_fit(X_quadratic, y, df["MRN"])
quadratic_z = beta_quadratic[2] / sqrt(cov_quadratic[2, 2])

knots = np.quantile(r, [0.05, 0.35, 0.65, 0.95])
spline = rcs_basis(r, knots)
X_spline = np.column_stack([intercept, r_centered, spline, base_matrix])
beta_spline, cov_spline, _ = logistic_fit(X_spline, y, df["MRN"])
spline_stat, spline_p = wald_joint(beta_spline, cov_spline, [2, 3])

reference_response = -3.0
curve_rows = []
observed_counts = df.groupby("pain_score_change")["persistent_severe_pain_6_24h"].agg(["size", "sum", "mean"])
for response_value in range(int(r.min()), int(r.max()) + 1):
    design_value = np.concatenate(([response_value - r.mean()], rcs_basis([response_value], knots)[0]))
    design_reference = np.concatenate(([reference_response - r.mean()], rcs_basis([reference_response], knots)[0]))
    contrast = design_value - design_reference
    response_beta = beta_spline[1:4]
    response_cov = cov_spline[1:4, 1:4]
    log_or = float(contrast @ response_beta)
    se = sqrt(float(contrast @ response_cov @ contrast))
    count = int(observed_counts.loc[response_value, "size"]) if response_value in observed_counts.index else 0
    events = int(observed_counts.loc[response_value, "sum"]) if response_value in observed_counts.index else 0
    curve_rows.append(
        {
            "pain_score_change": response_value,
            "cases": count,
            "events": events,
            "event_rate": events / count if count else np.nan,
            "adjusted_or_vs_minus3": np.exp(log_or),
            "ci_low": np.exp(log_or - 1.96 * se),
            "ci_high": np.exp(log_or + 1.96 * se),
        }
    )
curve = pd.DataFrame(curve_rows)
curve.to_csv(OUT / "restricted_cubic_spline_curve.csv", index=False)

linearity = pd.DataFrame(
    [
        {
            "analysis": "primary linear effect",
            "n": len(df),
            "events": int(y.sum()),
            "estimate": float(np.exp(beta_linear[1])),
            "ci_low": float(np.exp(beta_linear[1] - 1.96 * linear_se)),
            "ci_high": float(np.exp(beta_linear[1] + 1.96 * linear_se)),
            "test_statistic": float(beta_linear[1] / linear_se),
            "p_value": normal_two_sided_p(beta_linear[1] / linear_se),
            "notes": "OR per 1-point less favourable response; patient-cluster robust SE",
        },
        {
            "analysis": "quadratic nonlinearity check",
            "n": len(df),
            "events": int(y.sum()),
            "estimate": float(beta_quadratic[2]),
            "ci_low": float(beta_quadratic[2] - 1.96 * sqrt(cov_quadratic[2, 2])),
            "ci_high": float(beta_quadratic[2] + 1.96 * sqrt(cov_quadratic[2, 2])),
            "test_statistic": float(quadratic_z),
            "p_value": normal_two_sided_p(quadratic_z),
            "notes": "coefficient for centred response squared; patient-cluster robust Wald test",
        },
        {
            "analysis": "restricted cubic spline nonlinearity check",
            "n": len(df),
            "events": int(y.sum()),
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "test_statistic": spline_stat,
            "p_value": spline_p,
            "notes": "joint Wald test of two nonlinear terms; knots at response percentiles " + ", ".join(f"{v:g}" for v in knots),
        },
    ]
)
linearity.to_csv(OUT / "linearity_checks.csv", index=False)


def net_benefit(y_values, probabilities, threshold):
    positive = probabilities >= threshold
    odds = threshold / (1.0 - threshold)
    return (np.sum(positive & (y_values == 1)) / len(y_values)) - (np.sum(positive & (y_values == 0)) / len(y_values)) * odds


rng = np.random.default_rng(SEED)
patients = pred["MRN"].dropna().unique()
patient_rows = {patient: np.flatnonzero(pred["MRN"].to_numpy() == patient) for patient in patients}
thresholds = np.round(np.arange(0.10, 0.61, 0.01), 2)
bootstrap_differences = np.empty((2000, len(thresholds)))
y_pred = pred["persistent_severe_pain_6_24h"].to_numpy(int)
p_base = pred["pretreatment_baseline"].to_numpy(float)
p_response = pred["continuous_response"].to_numpy(float)
for b in range(2000):
    sampled_patients = rng.choice(patients, size=len(patients), replace=True)
    rows = np.concatenate([patient_rows[patient] for patient in sampled_patients])
    for t_index, threshold in enumerate(thresholds):
        bootstrap_differences[b, t_index] = net_benefit(y_pred[rows], p_response[rows], threshold) - net_benefit(y_pred[rows], p_base[rows], threshold)

dca_rows = []
for t_index, threshold in enumerate(thresholds):
    difference = net_benefit(y_pred, p_response, threshold) - net_benefit(y_pred, p_base, threshold)
    lower, upper = np.quantile(bootstrap_differences[:, t_index], [0.025, 0.975])
    dca_rows.append(
        {
            "threshold": threshold,
            "net_benefit_difference_response_minus_pretreatment": difference,
            "ci_low": lower,
            "ci_high": upper,
            "bootstrap_replicates": 2000,
            "cluster_unit": "patient",
        }
    )
dca = pd.DataFrame(dca_rows)
dca.to_csv(OUT / "locked_2021_decision_curve_uncertainty.csv", index=False)

selected = dca[dca["threshold"].isin([0.20, 0.30, 0.40, 0.50])]
report = [
    "SUBMISSION STATISTICAL CHECKS",
    "",
    "Linearity:",
    linearity.to_string(index=False),
    "",
    "Restricted cubic spline curve (reference response = -3):",
    curve.to_string(index=False),
    "",
    "Decision-curve net-benefit differences at selected thresholds:",
    selected.to_string(index=False),
]
(OUT / "submission_statistical_checks.txt").write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))
