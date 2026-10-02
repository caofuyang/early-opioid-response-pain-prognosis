from config import MOVER_ROOT, MOVER_DERIVED, MIMIC_ROOT, MIMIC_DERIVED, RESULTS_ROOT
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BASE = RESULTS_ROOT / "manuscript"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CHECKS = BASE / "submission_statistical_checks"
TEMPORAL = str(RESULTS_ROOT / "pain_response_stage4" / "validation_predictions.parquet")
LOCKED = str(RESULTS_ROOT / "pain_response_stage5_locked" / "locked_2021_predictions.parquet")
OUTCOME = "persistent_severe_pain_6_24h"
RNG = np.random.default_rng(20260827)


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default(size=size)


INK = "#243247"
BLUE = "#2E6F9E"
ORANGE = "#C9782A"
GREY = "#A9B2BD"
LIGHT_BLUE = "#DCEAF4"
GRID = "#D9DEE5"
WHITE = "#FFFFFF"


def text_center(draw, xy, label, text_font, fill=INK):
    box = draw.textbbox((0, 0), label, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), label, font=text_font, fill=fill)


def create_spline_figure():
    curve = pd.read_csv(CHECKS / "restricted_cubic_spline_curve.csv")
    width, height = 2200, 1500
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((130, 55), "Adjusted association across pain-score response values", font=font(54, True), fill=INK)
    draw.text((130, 125), "MOVER outcome-evaluable cohort; reference response = -3 points; n=6037", font=font(31), fill=INK)

    left, right = 190, 2070
    top, bottom = 245, 1070
    hist_top, hist_bottom = 1160, 1370
    xmin, xmax = curve.pain_score_change.min(), curve.pain_score_change.max()
    ymin, ymax = np.log(0.25), np.log(10.0)

    def xmap(value):
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    def ymap(value):
        return bottom - (np.log(value) - ymin) / (ymax - ymin) * (bottom - top)

    for tick in [0.25, 0.5, 1, 2, 4, 8]:
        y = ymap(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((80, y - 16), f"{tick:g}", font=font(27), fill=INK)
    draw.line((left, ymap(1), right, ymap(1)), fill=INK, width=4)
    for tick in range(int(xmin), int(xmax) + 1, 2):
        x = xmap(tick)
        draw.line((x, bottom, x, bottom + 12), fill=INK, width=3)
        text_center(draw, (x, bottom + 18), str(tick), font(25))

    upper = [(xmap(row.pain_score_change), ymap(row.ci_high)) for row in curve.itertuples()]
    lower = [(xmap(row.pain_score_change), ymap(row.ci_low)) for row in reversed(list(curve.itertuples()))]
    draw.polygon(upper + lower, fill=LIGHT_BLUE + "B3")
    points = [(xmap(row.pain_score_change), ymap(row.adjusted_or_vs_minus3)) for row in curve.itertuples()]
    draw.line(points, fill=BLUE, width=7, joint="curve")
    for x, y in points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=WHITE, outline=BLUE, width=4)
    draw.line((left, top, left, bottom), fill=INK, width=4)
    draw.line((left, bottom, right, bottom), fill=INK, width=4)
    draw.text((left, 190), "Adjusted odds ratio (log scale)", font=font(30, True), fill=INK)
    text_center(draw, ((left + right) / 2, 1415), "Pain-score change after opioid administration (post-dose minus pre-dose)", font(31, True))

    max_count = curve.cases.max()
    bar_width = (right - left) / len(curve) * 0.68
    for row in curve.itertuples():
        x = xmap(row.pain_score_change)
        bar_height = row.cases / max_count * (hist_bottom - hist_top)
        draw.rectangle((x - bar_width / 2, hist_bottom - bar_height, x + bar_width / 2, hist_bottom), fill=GREY, outline=INK, width=1)
    draw.text((left, hist_top - 45), "Observed cases by response value", font=font(27, True), fill=INK)
    draw.text((right - 590, top + 20), "Line: adjusted OR   Shading: 95% CI", font=font(27), fill=INK)
    image.save(FIG / "figureS1_spline_functional_form.png", dpi=(300, 300))


def calibration_table(data, model, bins, replicates=2000):
    work = data[["MRN", OUTCOME, model]].dropna().copy()
    work["bin"] = pd.qcut(work[model], bins, labels=False, duplicates="drop")
    summary = work.groupby("bin", sort=True).agg(
        n=(OUTCOME, "size"),
        mean_predicted=(model, "mean"),
        observed=(OUTCOME, "mean"),
    ).reset_index()
    patients = work["MRN"].unique()
    rows_by_patient = {patient: np.flatnonzero(work["MRN"].to_numpy() == patient) for patient in patients}
    boot = np.full((replicates, len(summary)), np.nan)
    values = work[OUTCOME].to_numpy(float)
    groups = work["bin"].to_numpy(int)
    for replicate in range(replicates):
        sampled = RNG.choice(patients, len(patients), replace=True)
        rows = np.concatenate([rows_by_patient[patient] for patient in sampled])
        for position, group in enumerate(summary["bin"]):
            selected = rows[groups[rows] == group]
            if len(selected):
                boot[replicate, position] = values[selected].mean()
    summary["ci_low"] = np.nanquantile(boot, 0.025, axis=0)
    summary["ci_high"] = np.nanquantile(boot, 0.975, axis=0)
    summary["model"] = model
    return summary


def create_calibration_figure():
    con = duckdb.connect()
    temporal = con.execute("SELECT * FROM read_parquet(?)", [TEMPORAL]).fetchdf()
    locked = con.execute("SELECT * FROM read_parquet(?)", [LOCKED]).fetchdf()
    con.close()
    tables = []
    for label, data, bins in [("2020 temporal evaluation", temporal, 10), ("2021 frozen test", locked, 5)]:
        for model in ["pretreatment_baseline", "continuous_response"]:
            table = calibration_table(data, model, bins)
            table["cohort"] = label
            tables.append(table)
    calibration = pd.concat(tables, ignore_index=True)
    calibration.to_csv(CHECKS / "calibration_plot_data.csv", index=False)

    width, height = 2400, 1280
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((135, 45), "Calibration of postoperative pain-risk predictions", font=font(53, True), fill=INK)
    draw.text((135, 115), "Points show grouped observed risk; vertical bars show 95% patient-cluster bootstrap intervals", font=font(30), fill=INK)

    panel_specs = [
        ("2020 temporal evaluation", 135, 1140, "A", "n=3307; 10 equal-frequency groups"),
        ("2021 frozen test", 1260, 2265, "B", "n=305; 5 equal-frequency groups"),
    ]
    plot_top, plot_bottom = 270, 1080
    axis_max = 0.8
    for cohort, left, right, letter, subtitle in panel_specs:
        draw.text((left, 180), f"{letter}. {cohort}", font=font(35, True), fill=INK)
        draw.text((left, 225), subtitle, font=font(25), fill=INK)

        def xmap(value):
            return left + value / axis_max * (right - left)

        def ymap(value):
            return plot_bottom - value / axis_max * (plot_bottom - plot_top)

        for tick in np.arange(0, axis_max + 0.001, 0.2):
            x, y = xmap(tick), ymap(tick)
            draw.line((left, y, right, y), fill=GRID, width=2)
            draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=2)
            draw.text((left - 66, y - 15), f"{tick:.1f}", font=font(24), fill=INK)
            text_center(draw, (x, plot_bottom + 16), f"{tick:.1f}", font(24))
        draw.line((left, plot_bottom, right, plot_top), fill=INK, width=4)
        draw.line((left, plot_top, left, plot_bottom), fill=INK, width=4)
        draw.line((left, plot_bottom, right, plot_bottom), fill=INK, width=4)

        cohort_data = calibration[calibration["cohort"] == cohort]
        for model, color, square in [("pretreatment_baseline", ORANGE, True), ("continuous_response", BLUE, False)]:
            model_data = cohort_data[cohort_data["model"] == model].sort_values("mean_predicted")
            points = [(xmap(row.mean_predicted), ymap(row.observed)) for row in model_data.itertuples()]
            draw.line(points, fill=color, width=6)
            for row in model_data.itertuples():
                x, y = xmap(row.mean_predicted), ymap(row.observed)
                low, high = ymap(row.ci_low), ymap(row.ci_high)
                draw.line((x, high, x, low), fill=color, width=4)
                draw.line((x - 10, high, x + 10, high), fill=color, width=4)
                draw.line((x - 10, low, x + 10, low), fill=color, width=4)
                if square:
                    draw.rectangle((x - 8, y - 8, x + 8, y + 8), fill=WHITE, outline=color, width=5)
                else:
                    draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=WHITE, outline=color, width=5)
        text_center(draw, ((left + right) / 2, 1135), "Mean predicted risk", font(29, True))

    draw.text((18, 610), "Observed risk", font=font(30, True), fill=INK)
    draw.rectangle((770, 1190, 792, 1212), fill=WHITE, outline=ORANGE, width=5)
    draw.text((810, 1183), "Pretreatment", font=font(27), fill=INK)
    draw.ellipse((1125, 1189, 1149, 1213), fill=WHITE, outline=BLUE, width=5)
    draw.text((1168, 1183), "Pretreatment + continuous response", font=font(27), fill=INK)
    draw.line((1680, 1201, 1740, 1201), fill=INK, width=4)
    draw.text((1760, 1183), "Ideal calibration", font=font(27), fill=INK)
    image.save(FIG / "figureS2_calibration.png", dpi=(300, 300))


create_spline_figure()
create_calibration_figure()
print(FIG / "figureS1_spline_functional_form.png")
print(FIG / "figureS2_calibration.png")
