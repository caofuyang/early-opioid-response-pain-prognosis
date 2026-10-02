from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = RESULTS_ROOT
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#164B73",
    "blue": "#2B78A6",
    "teal": "#2A9D8F",
    "gold": "#E9A23B",
    "red": "#C84C4C",
    "gray": "#6B7280",
    "light": "#E8EEF3",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Figure 1: study design and cohort formation
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 7)
ax.axis("off")

boxes = [
    (0.35, 5.25, 2.25, 1.0, "MOVER source data\n5,292,529 pain records\n20,274 surgical cases"),
    (3.0, 5.25, 2.25, 1.0, "Paired administrations\n34,687 candidate doses\n10,901 cases"),
    (5.65, 5.25, 2.25, 1.0, "Eligible index event\n19,766 doses after rules\nfirst event per case"),
    (8.3, 5.25, 2.25, 1.0, "Primary cohort\n7,238 cases\n6,231 patients"),
    (1.0, 2.6, 2.65, 1.1, "Exploratory association\n6,037 complete cases\n1,955 events"),
    (4.15, 2.6, 2.65, 1.1, "Temporal evaluation\n2018–2019 development: 2,426\n2020 validation: 3,307"),
    (7.3, 2.6, 2.65, 1.1, "Frozen test\n2018–2020 refit: 5,733\n2021 test: 305 (99 events)"),
    (4.15, 0.5, 2.65, 1.1, "MIMIC-IV replication\n1,624 ICU stays\n1,448 outcome-evaluable"),
]
for i, (x, y, w, h, text) in enumerate(boxes):
    color = COLORS["navy"] if i < 4 else (COLORS["teal"] if i == 7 else COLORS["blue"])
    rect = plt.Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#17212B", linespacing=1.3)

for x1, x2 in [(2.6, 3.0), (5.25, 5.65), (7.9, 8.3)]:
    ax.annotate("", xy=(x2, 5.75), xytext=(x1, 5.75), arrowprops=dict(arrowstyle="->", lw=1.5, color=COLORS["gray"]))
ax.plot([9.425, 9.425], [5.25, 4.55], color=COLORS["gray"], lw=1.25)
ax.plot([2.325, 8.625], [4.55, 4.55], color=COLORS["gray"], lw=1.25)
for x in [2.325, 5.475, 8.625]:
    ax.annotate("", xy=(x, 3.72), xytext=(x, 4.55), arrowprops=dict(arrowstyle="->", lw=1.25, color=COLORS["gray"]))
ax.text(5.475, 2.02, "Independent cross-setting replication", ha="center", va="center", fontsize=8, color=COLORS["gray"])
ax.annotate("", xy=(5.475, 1.62), xytext=(5.475, 1.9), arrowprops=dict(arrowstyle="->", lw=1.25, color=COLORS["gray"]))
ax.text(5.5, 6.75, "Administration-anchored postoperative pain-response study", ha="center", va="center", fontsize=14, fontweight="bold", color=COLORS["navy"])
ax.text(5.5, 4.55, "First opioid dose within 0–6 h after anesthesia end; pre-dose pain ≥4; post-dose score at 15–120 min; no intervening opioid", ha="center", va="center", fontsize=8.5, color=COLORS["gray"])
save(fig, "figure1_study_flow.png")


# Figure 2: association sensitivity forest plot
sens = pd.read_csv(ROOT / "pain_response_stage7_confounding" / "confounding_and_outcome_sensitivities.csv")
anes = pd.read_csv(ROOT / "pain_response_stage7_confounding" / "anesthesia_type_sensitivity.csv")
sens = pd.concat([sens, pd.DataFrame([{
    "analysis": "anesthesia_type_adjustment",
    "outcome": "primary_share50",
    "n": int(anes.loc[0, "n"]),
    "events": int(anes.loc[0, "events"]),
    "event_rate": int(anes.loc[0, "events"]) / int(anes.loc[0, "n"]),
    "or_per_1point_worse_response": anes.loc[0, "or_per_point"],
    "ci_low": anes.loc[0, "ci_low"],
    "ci_high": anes.loc[0, "ci_high"],
    "p_value": anes.loc[0, "p_value"],
}])], ignore_index=True)
labels = {
    "primary_base": "Primary model",
    "primary_extended_confounders": "Extended recorded confounders",
    "severe_share_40pct": "Severe scores ≥40%",
    "severe_share_60pct": "Severe scores ≥60%",
    "later_mean_pain_ge4": "Later mean pain ≥4",
    "later_mean_pain_ge7": "Later mean pain ≥7",
    "at_least_3_later_scores": "At least 3 later scores",
    "first_case_per_patient": "First case per patient",
    "exclude_chronic_pain_or_opioid_history": "Exclude recorded pain/opioid history",
    "anesthesia_type_adjustment": "Additional anesthesia-type adjustment",
}
sens["label"] = sens["analysis"].map(labels)
sens = sens.iloc[::-1].reset_index(drop=True)
y = np.arange(len(sens))
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.errorbar(sens["or_per_1point_worse_response"], y,
            xerr=[sens["or_per_1point_worse_response"] - sens["ci_low"], sens["ci_high"] - sens["or_per_1point_worse_response"]],
            fmt="o", color=COLORS["navy"], ecolor=COLORS["blue"], capsize=3, markersize=5)
ax.axvline(1, color=COLORS["gray"], linestyle="--", linewidth=1)
ax.set_yticks(y, sens["label"])
ax.set_xlabel("Adjusted odds ratio per 1-point less favorable response (95% CI)")
ax.set_xlim(0.98, 1.31)
ax.grid(axis="x", color=COLORS["light"], linewidth=0.8)
for yi, row in sens.iterrows():
    ax.text(1.235, yi, f"{row['or_per_1point_worse_response']:.3f} ({row['ci_low']:.3f}–{row['ci_high']:.3f})", va="center", fontsize=8)
ax.set_title("MOVER association was stable across sensitivity analyses", loc="left", fontweight="bold")
save(fig, "figure2_association_sensitivity.png")


# Figure 3: temporal performance and decision curves
temporal = pd.read_csv(ROOT / "pain_response_stage4" / "temporal_validation_metrics.csv")
locked = pd.read_csv(ROOT / "pain_response_stage5_locked" / "locked_2021_metrics.csv")
dca = pd.read_csv(ROOT / "pain_response_stage5_locked" / "locked_2021_decision_curve.csv")

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.35]})

t = temporal[temporal.model.isin(["pretreatment_baseline", "continuous_response"])].copy()
for i, metric in enumerate(["auroc", "auprc"]):
    ax = axes[i]
    vals = t[metric].to_numpy()
    ax.bar([0, 1], vals, color=[COLORS["gray"], COLORS["teal"]], width=.62)
    ax.set_xticks([0, 1], ["Pretreatment", "+ continuous response"])
    ax.set_ylim(0.45 if metric == "auprc" else 0.62, 0.74)
    ax.set_ylabel(metric.upper())
    ax.set_title(("A. 2020 temporal AUROC" if i == 0 else "B. 2020 temporal AUPRC"), loc="left", fontweight="bold")
    for j, v in enumerate(vals):
        ax.text(j, v + .006, f"{v:.3f}", ha="center", fontweight="bold")
    ax.grid(axis="y", color=COLORS["light"], linewidth=.8)

ax = axes[2]
for model, color, label in [
    ("pretreatment_baseline", COLORS["gray"], "Pretreatment"),
    ("continuous_response", COLORS["teal"], "+ continuous response"),
    ("treat_all", COLORS["red"], "Treat all"),
    ("treat_none", "#111111", "Treat none"),
]:
    z = dca[dca.model == model]
    ax.plot(z.threshold, z.net_benefit, color=color, lw=2 if "response" in model else 1.4, label=label)
ax.set_xlim(.10, .60)
ax.set_ylim(min(-.12, dca.net_benefit.min() * 1.05), .31)
ax.set_xlabel("Risk threshold")
ax.set_ylabel("Net benefit")
ax.set_title("C. Frozen 2021 decision curve", loc="left", fontweight="bold")
ax.axhline(0, color="#111111", lw=.7)
ax.grid(color=COLORS["light"], linewidth=.7)
ax.legend(frameon=False, fontsize=8, loc="lower left")

fig.suptitle("Incremental prognostic performance across time", fontsize=13, fontweight="bold", color=COLORS["navy"], y=1.02)
save(fig, "figure3_prediction_performance.png")


# Figure 4: cross-database and drug subgroup associations
cross = pd.read_csv(ROOT / "pain_response_stage6_sensitivity" / "continuous_effect_cross_database_and_drugs.csv")
extra = pd.read_csv(ROOT / "mimic_pain_response_external" / "surgical_service_sensitivity.csv")
extra_row = pd.DataFrame([{
    "analysis": "MIMIC-IV surgical service",
    "n": int(extra.loc[0, "n"]),
    "events": int(extra.loc[0, "events"]),
    "or_per_1point_worse_response": extra.loc[0, "or_per_point"],
    "ci_low": extra.loc[0, "ci_low"],
    "ci_high": extra.loc[0, "ci_high"],
}])
cross = pd.concat([cross, extra_row], ignore_index=True)
cross["label"] = cross["analysis"].str.replace("MOVER ", "MOVER: ", regex=False).str.replace("MIMIC-IV ", "MIMIC-IV: ", regex=False)
cross = cross.iloc[::-1].reset_index(drop=True)
y = np.arange(len(cross))
colors = [COLORS["teal"] if s.startswith("MIMIC-IV") else COLORS["navy"] for s in cross["label"]]
fig, ax = plt.subplots(figsize=(8.4, 5.0))
for yi, row in cross.iterrows():
    ax.errorbar(row["or_per_1point_worse_response"], yi,
                xerr=[[row["or_per_1point_worse_response"] - row["ci_low"]], [row["ci_high"] - row["or_per_1point_worse_response"]]],
                fmt="o", color=colors[yi], ecolor=colors[yi], capsize=3, markersize=5)
ax.axvline(1, color=COLORS["gray"], linestyle="--", linewidth=1)
ax.set_yticks(y, cross["label"])
ax.set_xlabel("Adjusted odds ratio per 1-point less favorable response (95% CI)")
ax.set_xlim(.98, 1.56)
ax.grid(axis="x", color=COLORS["light"], linewidth=.8)
for yi, row in cross.iterrows():
    ax.text(1.425, yi, f"{row['or_per_1point_worse_response']:.3f} ({row['ci_low']:.3f}–{row['ci_high']:.3f})", va="center", fontsize=7.6)
ax.set_title("Directionally consistent associations across databases and opioid subgroups", loc="left", fontweight="bold")
save(fig, "figure4_cross_database_consistency.png")

print(f"Created 4 figures in {OUT}")
