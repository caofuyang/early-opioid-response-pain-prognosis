import duckdb, pandas as pd, numpy as np
from config import MIMIC_DERIVED, RESULTS_ROOT
D=str(MIMIC_DERIVED)
con=duckdb.connect(); con.execute("PRAGMA threads=8")
con.execute(f"""
CREATE TEMP TABLE cand AS
SELECT d.stay_id, c.subject_id, c.post_time AS index_post, c.intime,
       d.administration_time AS dose_time, d.opioid, d.mme
FROM read_parquet('{D}/opioid_boluses_external_stage1.parquet') d
JOIN read_parquet('{D}/early_analgesic_response_external_stage2.parquet') c USING (stay_id)
WHERE d.mme IS NOT NULL AND d.administration_time > c.post_time
  AND d.administration_time <= c.intime + INTERVAL 24 HOUR
""")
con.execute("""
CREATE TEMP TABLE dedup AS
WITH o AS (SELECT *, lag(dose_time) OVER (PARTITION BY stay_id ORDER BY dose_time) p FROM cand)
SELECT * EXCLUDE (p) FROM o WHERE p IS NULL OR date_diff('minute',p,dose_time)>=15
""")
con.execute(f"""
CREATE TEMP TABLE ep AS
SELECT d.*, pre.pain_score pre2, post.pain_score post2, post.charttime AS post_time, pre.charttime AS pre_time
FROM dedup d
ASOF LEFT JOIN read_parquet('{D}/pain_scores_external_stage1.parquet') pre
  ON d.stay_id=pre.stay_id AND d.dose_time>=pre.charttime
ASOF LEFT JOIN read_parquet('{D}/pain_scores_external_stage1.parquet') post
  ON d.stay_id=post.stay_id AND d.dose_time + INTERVAL 15 MINUTE <= post.charttime
WHERE pre.pain_score>=4 AND date_diff('minute',pre.charttime,d.dose_time)<=60
  AND date_diff('minute',d.dose_time,post.charttime) BETWEEN 15 AND 120
""")
con.execute(f"""
CREATE TEMP TABLE clean AS
SELECT e.* FROM ep e WHERE NOT EXISTS (
  SELECT 1 FROM read_parquet('{D}/opioid_boluses_external_stage1.parquet') d2
  WHERE d2.stay_id=e.stay_id AND d2.administration_time>e.dose_time
    AND d2.administration_time<e.post_time AND d2.mme IS NOT NULL)
""")
df=con.execute("SELECT * FROM clean").fetchdf()
con.close()
df["change"]=df.post2-df.pre2; df["success"]=(df.change<=-2).astype(int)
df=df.sort_values(["stay_id","dose_time"]).reset_index(drop=True)
prev=[]
for _,g in df.groupby("stay_id",sort=False):
    done=[];out=[]
    for r in g.itertuples():
        known=[s for t,s in done if t<r.dose_time]
        out.append(sum(1 for s in known if s==0))
        done.append((r.post_time, r.success))
    prev.extend(out)
df["prev"]=prev
print("episodes",len(df),"stays",df.stay_id.nunique())
print(df.groupby(df.prev.clip(upper=3)).success.agg(["size","mean"]).round(3).to_string())
OUT = RESULTS_ROOT / "pain_response_v2"
OUT.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT / "mimic_v2.parquet", index=False)
