"""
Script 26 — Headline models (v3.1, corrected bootstrap and model reporting)
==========================================================================
Fixes:
  * patient-cluster bootstrap now accounts for how many times each patient is
    drawn (weighted mean), instead of silently de-duplicating with .isin()
  * the adjusted model is a logistic regression with cluster-robust standard
    errors; it is no longer called a GEE
  * covariate set, sample size and decomposition n are exported from the
    fitted data

Outputs: RESULTS_ROOT/pain_response_v2/models_v31.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import RESULTS_ROOT, MOVER_DERIVED

OUT = RESULTS_ROOT / "pain_response_v2"
MOVER = OUT / "rescue_episodes_v2.parquet"
MIMIC = OUT / "mimic_v2.parquet"
COHORT = MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet"


def cluster_bootstrap(df, cat_col, outcome, n_boot=1000, seed=11):
    """Weighted patient-cluster bootstrap: each patient drawn k times contributes k."""
    rng = np.random.default_rng(seed)
    ids = df["MRN"].unique()
    out = {k: [] for k in range(4)}
    groups = {k: g for k, g in df.groupby(cat_col)}
    for _ in range(n_boot):
        draw = rng.choice(ids, len(ids), replace=True)
        counts = pd.Series(draw).value_counts()
        for k, g in groups.items():
            w = g["MRN"].map(counts).fillna(0).values
            if w.sum() > 0:
                out[k].append(float(np.average(g[outcome].values, weights=w)))
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in out.items()}


def main() -> None:
    mov = pd.read_parquet(MOVER)
    cov = pd.read_parquet(COHORT)[["LOG_ID", "age", "SEX", "ASA_RATING"]]
    mov = mov.merge(cov, on="LOG_ID", how="left")
    mov["mcid"] = mov["success"]
    rows = []

    boot = cluster_bootstrap(mov, "prev_fail_cat", "mcid")
    for k in range(4):
        g = mov[mov.prev_fail_cat == k]
        rows.append({"analysis": "MOVER success", "group": k, "n_episodes": len(g),
                     "estimate": round(float(g.mcid.mean()), 3),
                     "ci_low": round(boot[k][1], 3), "ci_high": round(boot[k][2], 3)})

    sub = mov.dropna(subset=["pre_score", "mme", "age", "SEX", "ASA_RATING"]).copy()
    sub["pcat"] = pd.Categorical(sub["prev_fail_cat"])
    sub["mins"] = (pd.to_datetime(sub["dose_time"]) - pd.to_datetime(sub["anesthesia_stop"])).dt.total_seconds() / 60
    model = smf.glm("mcid ~ pcat + mins + pre_score + mme + age + C(SEX) + C(ASA_RATING)",
                    data=sub, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": sub["MRN"]})
    for k in ["pcat[T.1]", "pcat[T.2]", "pcat[T.3]"]:
        ci = model.conf_int().loc[k]
        rows.append({"analysis": "MOVER adjusted OR", "group": k, "n_episodes": len(sub),
                     "n_patients": sub.MRN.nunique(),
                     "estimate": round(float(np.exp(model.params[k])), 3),
                     "ci_low": round(float(np.exp(ci[0])), 3),
                     "ci_high": round(float(np.exp(ci[1])), 3)})

    mov["seq"] = mov.groupby("LOG_ID").cumcount() + 1
    fixed = mov[mov.seq <= 3].pivot_table(index="LOG_ID", columns="seq", values="mcid", aggfunc="first").dropna()
    for k in [1, 2, 3]:
        rows.append({"analysis": "MOVER fixed cohort", "group": k, "n_episodes": len(fixed),
                     "estimate": round(float(fixed[k].mean()), 3)})

    mov["x"] = np.log2(mov["seq"])
    s = mov[mov.groupby("LOG_ID").LOG_ID.transform("size") >= 2].copy()
    for c in ["mcid", "x"]:
        s[c + "_w"] = s[c] - s.groupby("MRN")[c].transform("mean")
    fe = sm.OLS(s["mcid_w"], s[["x_w"]]).fit(cov_type="cluster", cov_kwds={"groups": s["MRN"]})
    ci = fe.conf_int().loc["x_w"]
    rows.append({"analysis": "MOVER within-patient", "group": "per doubling",
                 "n_episodes": len(s), "n_cases": s.LOG_ID.nunique(), "n_patients": s.MRN.nunique(),
                 "estimate": round(float(fe.params["x_w"]), 3),
                 "ci_low": round(float(ci[0]), 3), "ci_high": round(float(ci[1]), 3)})
    gm = s.groupby("MRN").agg(m=("mcid", "mean"), x=("x", "mean"))
    be = sm.OLS(gm["m"], sm.add_constant(gm["x"])).fit(cov_type="HC1")
    ci = be.conf_int().loc["x"]
    rows.append({"analysis": "MOVER between-patient", "group": "per doubling",
                 "n_patients": len(gm),
                 "estimate": round(float(be.params["x"]), 3),
                 "ci_low": round(float(ci[0]), 3), "ci_high": round(float(ci[1]), 3)})

    mim = pd.read_parquet(MIMIC)
    mim["mcid"] = mim["success"]
    mim["cat"] = mim["prev"].clip(upper=3)
    rng = np.random.default_rng(5)
    mids = mim.subject_id.unique()
    mdraws = {k: [] for k in range(4)}
    for _ in range(1000):
        draw = rng.choice(mids, len(mids), replace=True)
        cnt = pd.Series(draw).value_counts()
        for k, g in mim.groupby("cat"):
            w = g.subject_id.map(cnt).fillna(0).values
            if w.sum() > 0:
                mdraws[k].append(float(np.average(g.mcid.values, weights=w)))
    for k in range(4):
        g = mim[mim.cat == k]
        v = np.array(mdraws[k])
        rows.append({"analysis": "MIMIC success", "group": k, "n_episodes": len(g),
                     "n_stays": g.stay_id.nunique(), "n_patients": g.subject_id.nunique(),
                     "estimate": round(float(g.mcid.mean()), 3),
                     "ci_low": round(float(np.percentile(v, 2.5)), 3),
                     "ci_high": round(float(np.percentile(v, 97.5)), 3)})

    mim["seq"] = mim.groupby("stay_id").cumcount() + 1
    mim["x"] = np.log2(mim["seq"])
    ms = mim[mim.groupby("stay_id").stay_id.transform("size") >= 2].copy()
    for c in ["mcid", "x"]:
        ms[c + "_w"] = ms[c] - ms.groupby("subject_id")[c].transform("mean")
    mf = sm.OLS(ms["mcid_w"], ms[["x_w"]]).fit(cov_type="cluster", cov_kwds={"groups": ms["subject_id"]})
    ci = mf.conf_int().loc["x_w"]
    rows.append({"analysis": "MIMIC within-patient", "group": "per doubling",
                 "n_episodes": len(ms), "n_stays": ms.stay_id.nunique(), "n_patients": ms.subject_id.nunique(),
                 "estimate": round(float(mf.params["x_w"]), 3),
                 "ci_low": round(float(ci[0]), 3), "ci_high": round(float(ci[1]), 3)})
    mg = ms.groupby("subject_id").agg(m=("mcid", "mean"), x=("x", "mean"))
    mb = sm.OLS(mg["m"], sm.add_constant(mg["x"])).fit(cov_type="HC1")
    ci = mb.conf_int().loc["x"]
    rows.append({"analysis": "MIMIC between-patient", "group": "per doubling",
                 "n_patients": len(mg),
                 "estimate": round(float(mb.params["x"]), 3),
                 "ci_low": round(float(ci[0]), 3), "ci_high": round(float(ci[1]), 3)})

    # interval analysis on the same subsample: base model vs model with gap
    mov["gap_h"] = mov.groupby("LOG_ID").dose_time.diff().dt.total_seconds() / 3600
    iv = mov.dropna(subset=["gap_h", "pre_score", "mme", "age", "SEX", "ASA_RATING"]).copy()
    iv["pcat"] = pd.Categorical(iv["prev_fail_cat"])
    iv["mins"] = (pd.to_datetime(iv["dose_time"]) - pd.to_datetime(iv["anesthesia_stop"])).dt.total_seconds() / 60
    for label, formula in [
        ("MOVER same-sample base OR", "mcid ~ pcat + mins + pre_score + mme + age + C(SEX) + C(ASA_RATING)"),
        ("MOVER same-sample +gap OR", "mcid ~ pcat + mins + gap_h + pre_score + mme + age + C(SEX) + C(ASA_RATING)"),
    ]:
        mi = smf.glm(formula, data=iv, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": iv["MRN"]})
        for k in ["pcat[T.1]", "pcat[T.2]", "pcat[T.3]"]:
            ci = mi.conf_int().loc[k]
            rows.append({"analysis": label, "group": k, "n_episodes": len(iv),
                         "estimate": round(float(np.exp(mi.params[k])), 3),
                         "ci_low": round(float(np.exp(ci[0])), 3),
                         "ci_high": round(float(np.exp(ci[1])), 3)})

    # MIMIC adjusted model
    mim2 = mim.dropna(subset=["pre2", "mme"]).copy()
    mim2["mins"] = (pd.to_datetime(mim2["dose_time"]) - pd.to_datetime(mim2["intime"])).dt.total_seconds() / 60 if "intime" in mim2 else np.nan
    mim2["pcat"] = pd.Categorical(mim2["cat"])
    mm = smf.glm("mcid ~ pcat + pre2 + mme", data=mim2, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": mim2["subject_id"]})
    for k in ["pcat[T.1]", "pcat[T.2]", "pcat[T.3]"]:
        ci = mm.conf_int().loc[k]
        rows.append({"analysis": "MIMIC adjusted OR", "group": k, "n_episodes": len(mim2),
                     "n_patients": mim2.subject_id.nunique(),
                     "estimate": round(float(np.exp(mm.params[k])), 3),
                     "ci_low": round(float(np.exp(ci[0])), 3),
                     "ci_high": round(float(np.exp(ci[1])), 3)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "models_v31.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
