"""
Script 30 — Route-specific reassessment window sensitivity (MOVER)
=================================================================
Rebuilds episodes from the derived tables with the reassessment window matched
to the route:
  * intravenous: first score 15-60 minutes after the dose
  * oral:        first score 60-120 minutes after the dose
All other rules are those of script 20.

Output: RESULTS_ROOT/pain_response_v2/route_window_episodes.parquet
        RESULTS_ROOT/pain_response_v2/route_window_summary.csv
"""

import duckdb
import numpy as np
import pandas as pd

from config import MOVER_DERIVED, RESULTS_ROOT

OUT = RESULTS_ROOT / "pain_response_v2"
OUT.mkdir(parents=True, exist_ok=True)

COHORT = MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet"
PAIN = MOVER_DERIVED / "postop_pain_scores_stage1.parquet"
DOSES = MOVER_DERIVED / "postop_opioid_doses_24h_stage2.parquet"


def sql_path(p):
    return str(p).replace("\\", "/").replace("'", "''")


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    con.execute(f"""
    CREATE TEMP TABLE cand AS
    SELECT d.LOG_ID, c.MRN, c.anesthesia_stop, d.administration_time AS dose_time,
           d.opioid, d.route, d.mme
    FROM read_parquet('{sql_path(DOSES)}') d
    JOIN read_parquet('{sql_path(COHORT)}') c USING (LOG_ID)
    WHERE d.mme IS NOT NULL
      AND d.administration_time > c.post_time
      AND d.administration_time <= c.anesthesia_stop + INTERVAL 24 HOUR
    """)

    con.execute("""
    CREATE TEMP TABLE dedup AS
    WITH o AS (SELECT *, lag(dose_time) OVER (PARTITION BY LOG_ID ORDER BY dose_time) AS p FROM cand)
    SELECT * EXCLUDE (p) FROM o
    WHERE p IS NULL OR date_diff('minute', p, dose_time) >= 15
    """)

    con.execute(f"""
    CREATE TEMP TABLE pre AS
    SELECT d.*, pre.pain_score AS pre_score, pre.recorded_time AS pre_time
    FROM dedup d
    ASOF LEFT JOIN read_parquet('{sql_path(PAIN)}') pre
      ON d.LOG_ID = pre.LOG_ID AND d.dose_time >= pre.recorded_time
    WHERE pre.pain_score >= 4 AND date_diff('minute', pre.recorded_time, d.dose_time) <= 60
    """)

    # route-specific post window
    con.execute(f"""
    CREATE TEMP TABLE post_iv AS
    SELECT p.*, post.pain_score AS post_score, post.recorded_time AS post_time
    FROM pre p
    ASOF LEFT JOIN read_parquet('{sql_path(PAIN)}') post
      ON p.LOG_ID = post.LOG_ID AND p.dose_time + INTERVAL 15 MINUTE <= post.recorded_time
    WHERE lower(p.route) LIKE '%venous%'
      AND date_diff('minute', p.dose_time, post.recorded_time) BETWEEN 15 AND 60
    """)

    con.execute(f"""
    CREATE TEMP TABLE post_oral AS
    SELECT p.*, post.pain_score AS post_score, post.recorded_time AS post_time
    FROM pre p
    ASOF LEFT JOIN read_parquet('{sql_path(PAIN)}') post
      ON p.LOG_ID = post.LOG_ID AND p.dose_time + INTERVAL 60 MINUTE <= post.recorded_time
    WHERE lower(p.route) NOT LIKE '%venous%'
      AND date_diff('minute', p.dose_time, post.recorded_time) BETWEEN 60 AND 120
    """)

    con.execute(f"""
    CREATE TEMP TABLE ep AS
    SELECT * FROM post_iv
    UNION ALL
    SELECT * FROM post_oral
    """)

    con.execute(f"""
    CREATE TEMP TABLE clean AS
    SELECT e.* FROM ep e
    WHERE NOT EXISTS (
      SELECT 1 FROM read_parquet('{sql_path(DOSES)}') d2
      WHERE d2.LOG_ID = e.LOG_ID
        AND d2.administration_time > e.dose_time
        AND d2.administration_time < e.post_time
        AND d2.mme IS NOT NULL)
    """)

    df = con.execute("SELECT * FROM clean").fetchdf()
    con.close()

    df = df.sort_values(["LOG_ID", "dose_time"]).reset_index(drop=True)
    df["change"] = df["post_score"] - df["pre_score"]
    df["success"] = (df["change"] <= -2).astype(int)

    prev = []
    for _, g in df.groupby("LOG_ID", sort=False):
        done = []
        out = []
        for r in g.itertuples():
            known = [s for t, s in done if t < r.dose_time]
            out.append(sum(1 for s in known if s == 0))
            done.append((r.post_time, r.success))
        prev.extend(out)
    df["prev"] = prev
    df["prev_cat"] = df["prev"].clip(upper=3)

    df.to_parquet(OUT / "route_window_episodes.parquet", index=False)
    summary = df.groupby("prev_cat").success.agg(["size", "sum", "mean"]).round(3)
    summary.to_csv(OUT / "route_window_summary.csv")
    print(summary.to_string())
    print("episodes", len(df), "cases", df.LOG_ID.nunique(), "patients", df.MRN.nunique())


if __name__ == "__main__":
    main()
