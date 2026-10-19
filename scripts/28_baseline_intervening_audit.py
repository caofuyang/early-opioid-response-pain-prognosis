import duckdb, pandas as pd, numpy as np
from pathlib import Path
from config import RESULTS_ROOT
R = RESULTS_ROOT / "pain_response_v2"
con=duckdb.connect()
def audit(df,idc,timec,prec,postc,success,doses,dose_time_col,dose_id):
    con.register("ep", df[[idc,timec,prec]].rename(columns={timec:"dose_time",prec:"pre_time"}))
    flag=con.execute(f"""SELECT ep.{idc}, ep.dose_time, count(*) FILTER (
      WHERE d.{dose_time_col}>=ep.pre_time AND d.{dose_time_col}<ep.dose_time AND d.mme IS NOT NULL) AS n
      FROM ep LEFT JOIN read_parquet('{doses}') d USING({idc}) GROUP BY 1,2""").fetchdf()
    out=df.merge(flag,on=[idc,"dose_time"] if timec=="dose_time" else [idc,timec],how="left")
    out["n"]=out["n"].fillna(0); out["clean"]=out["n"]==0
    out=out.sort_values([idc,timec])
    hist={}
    for cid,g in out.groupby(idc,sort=False):
        done=[];vals=[]
        for r in g.itertuples():
            d=r._asdict()
            known=[s for t,s in done if t<d[timec]]
            vals.append(sum(1 for s in known if s==0))
            if d["clean"]:
                done.append((d[postc],d[success]))
        hist.update({idx:v for idx,v in zip(g.index,vals)})
    out["prev_clean"]=pd.Series(hist)
    clean=out[out.clean]
    tab=clean.groupby(clean.prev_clean.clip(upper=3))[success].agg(["size","sum","mean"]).round(3)
    return int((~out.clean).sum()), tab
from config import MOVER_DERIVED
D=str(MOVER_DERIVED)
m=pd.read_parquet(R/"rescue_episodes_v2.parquet")
n1,t1=audit(m,"LOG_ID","dose_time","pre_time","post_time","success",f"{D}/postop_opioid_doses_24h_stage2.parquet","administration_time","dose_id")
print("MOVER excluded",n1); print(t1.to_string()); t1.to_csv(R/"baseline_audit_mover.csv")
from config import MIMIC_DERIVED
D2=str(MIMIC_DERIVED)
mm=pd.read_parquet(R/"mimic_v2.parquet")
n2,t2=audit(mm,"stay_id","dose_time","pre_time","post_time","success",f"{D2}/opioid_boluses_external_stage1.parquet","administration_time","dose_id")
print("MIMIC excluded",n2); print(t2.to_string()); t2.to_csv(R/"baseline_audit_mimic.csv")
