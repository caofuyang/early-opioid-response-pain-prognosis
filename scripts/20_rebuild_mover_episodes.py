"""
Script 20 — Rebuild MOVER rescue episodes with audited rules
===========================================================
Fixes identified in the 2026-10-17 audit:
  * episodes start at the index reassessment (not a 6-hour lower bound)
  * doses within 15 minutes of a preceding dose are merged (first kept)
  * no opioid may be given between the dose and its reassessment
  * a previous failure counts only if its reassessment was complete before the
    current dose was given (no look-ahead)
  * failure is the complement of success (fall <2 points), stated explicitly
  * outputs carry MRN for patient-level clustering

Output: MOVER_DERIVED/rescue_episodes_v2.parquet
        RESULTS_ROOT/pain_response_v2/episode_audit.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

from config import MOVER_DERIVED, RESULTS_ROOT

COHORT = MOVER_DERIVED / "early_analgesic_response_cohort_stage3.parquet"
PAIN = MOVER_DERIVED / "postop_pain_scores_stage1.parquet"
DOSES = MOVER_DERIVED / "postop_opioid_doses_24h_stage2.parquet"
OUT = RESULTS_ROOT / "pain_response_v2"
OUT.mkdir(parents=True, exist_ok=True)


def sql_path(p) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # 1. candidate doses after the index reassessment, within 24 h of anesthesia end
    con.execute(f"""
    CREATE TEMP TABLE cand AS
    SELECT d.LOG_ID, c.MRN, c.anesthesia_stop, c.post_time AS index_post_time,
           d.administration_time AS dose_time, d.opioid, d.route, d.mme
    FROM read_parquet('{sql_path(DOSES)}') d
    INNER JOIN read_parquet('{sql_path(COHORT)}') c USING (LOG_ID)
    WHERE d.mme IS NOT NULL
      AND d.administration_time > c.post_time
      AND d.administration_time <= c.anesthesia_stop + INTERVAL 24 HOUR
    """)

    # 2. merge doses less than 15 minutes apart (keep the first of each cluster)
    con.execute("""
    CREATE TEMP TABLE dedup AS
    WITH ordered AS (
        SELECT *, lag(dose_time) OVER (PARTITION BY LOG_ID ORDER BY dose_time) AS prev_time
        FROM cand
    )
    SELECT * EXCLUDE (prev_time) FROM ordered
    WHERE prev_time IS NULL OR date_diff('minute', prev_time, dose_time) >= 15
    """)

    # 3. pre-dose and post-dose pain, no intervening opioid
    con.execute(f"""
    CREATE TEMP TABLE episodes AS
    SELECT d.*,
           pre.pain_score  AS pre_score,
           pre.recorded_time AS pre_time,
           post.pain_score AS post_score,
           post.recorded_time AS post_time,
           date_diff('minute', pre.recorded_time, d.dose_time) AS minutes_from_pre,
           date_diff('minute', d.dose_time, post.recorded_time) AS minutes_to_post
    FROM dedup d
    ASOF LEFT JOIN read_parquet('{sql_path(PAIN)}') pre
      ON d.LOG_ID = pre.LOG_ID AND d.dose_time >= pre.recorded_time
    ASOF LEFT JOIN read_parquet('{sql_path(PAIN)}') post
      ON d.LOG_ID = post.LOG_ID AND d.dose_time + INTERVAL 15 MINUTE <= post.recorded_time
    WHERE pre.pain_score >= 4
      AND date_diff('minute', pre.recorded_time, d.dose_time) <= 60
      AND date_diff('minute', d.dose_time, post.recorded_time) BETWEEN 15 AND 120
    """)

    # 4. exclude any opioid given between the dose and its reassessment
    con.execute(f"""
    CREATE TEMP TABLE clean AS
    SELECT e.* FROM episodes e
    WHERE NOT EXISTS (
        SELECT 1 FROM read_parquet('{sql_path(DOSES)}') d2
        WHERE d2.LOG_ID = e.LOG_ID
          AND d2.administration_time > e.dose_time
          AND d2.administration_time < e.post_time
          AND d2.mme IS NOT NULL
    )
    """)

    df = con.execute("SELECT * FROM clean").fetchdf()
    con.close()

    df = df.sort_values(["LOG_ID", "dose_time"]).reset_index(drop=True)
    df["change"] = df["post_score"] - df["pre_score"]
    df["success"] = (df["change"] <= -2).astype(int)

    # 5. previous non-success episodes known at the time of this dose
    prev_fail = []
    for _, g in df.groupby("LOG_ID", sort=False):
        done_times, done_success = [], []
        out = []
        for row in g.itertuples():
            known = [s for t, s in zip(done_times, done_success) if t < row.dose_time]
            out.append(sum(1 for s in known if s == 0))
            done_times.append(row.post_time)
            done_success.append(row.success)
        prev_fail.extend(out)
    df["prev_fail"] = prev_fail
    df["prev_fail_cat"] = df["prev_fail"].clip(upper=3)

    df.to_parquet(OUT / "rescue_episodes_v2.parquet", index=False)

    audit = pd.DataFrame({
        "metric": ["episodes", "cases", "patients", "events_before_6h",
                   "median_minutes_after_anesthesia"],
        "value": [len(df), df.LOG_ID.nunique(), df.MRN.nunique(),
                  int((((df.dose_time - df.anesthesia_stop).dt.total_seconds() / 60) < 360).sum()),
                  float(((df.dose_time - df.anesthesia_stop).dt.total_seconds() / 60).median())],
    })
    audit.to_csv(OUT / "episode_audit.csv", index=False)

    print(audit.to_string(index=False))
    print(df.groupby("prev_fail_cat").success.agg(["size", "mean"]).round(3).to_string())
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
