import pandas as pd, numpy as np
from pathlib import Path
import sys
from config import RESULTS_ROOT
R = RESULTS_ROOT / "pain_response_v2"
d=pd.read_parquet(R/"rescue_episodes_v2.parquet")
d["Pain"]=pd.cut(d.pre_score,[3,5,6,7,8,10],labels=["4–5","6","7","8","9–10"])
d["Route"]=np.where(d.route.str.lower().str.contains("venous"),"Intravenous","Oral")
d["Opioid"]=d.opioid.where(~d.opioid.isin(["codeine","meperidine"]),"Other")
d["Prev"]=d.prev_fail_cat.map({0:"0",1:"1",2:"2",3:"≥3"})
rng=np.random.default_rng(21)
def boot_cell(df,by):
    # per-patient sums and counts per cell
    g=df.groupby(["MRN"]+by).success.agg(["sum","size"]).reset_index()
    ids=g.MRN.unique(); idx={p:i for i,p in enumerate(ids)}
    cells=g.groupby(by)
    out={}
    for key,c in list(cells):
        v=np.zeros(len(ids)); v[[idx[p] for p in c.MRN]]=c["sum"].values
        n=np.zeros(len(ids)); n[[idx[p] for p in c.MRN]]=c["size"].values
        est=v.sum()/n.sum(); reps=[]
        for _ in range(1000):
            w=rng.multinomial(len(ids),np.ones(len(ids))/len(ids))
            den=(n*w).sum()
            if den>0: reps.append((v*w).sum()/den)
        out[key]=(est*100,np.percentile(reps,2.5)*100,np.percentile(reps,97.5)*100,int(n.sum()),int(v.sum()))
    return out
def table(by,label):
    res=boot_cell(d,by); rows=[]
    for k,(p,lo,hi,n,s) in res.items():
        rows.append({**dict(zip(by,k if isinstance(k,tuple) else (k,))),"Episodes":n,"Responses":s,"Percent":round(p,1),"95% CI":f"{lo:.1f}–{hi:.1f}"})
    t=pd.DataFrame(rows); print(label); print(t.to_string(index=False)); t.to_csv(R/f"table_{label}.csv",index=False)
table(["Pain","Prev"],"S1"); table(["Opioid","Prev"],"S2"); table(["Route","Prev"],"S3")
# Panel B cluster CI
d2=d.sort_values(["LOG_ID","dose_time"]).copy(); d2["seq"]=d2.groupby("LOG_ID").cumcount()+1
piv=d2[d2.seq<=3].pivot_table(index="LOG_ID",columns="seq",values="success",aggfunc="first").dropna()
mrn=d2.drop_duplicates("LOG_ID").set_index("LOG_ID").MRN
piv=piv.join(mrn)
ids=piv.MRN.unique(); idx={p:i for i,p in enumerate(ids)}
res={}
for k in [1,2,3]:
    v=np.zeros(len(ids)); n=np.zeros(len(ids))
    g=piv.groupby("MRN")[k].agg(["sum","size"])
    v[[idx[p] for p in g.index]]=g["sum"].values; n[[idx[p] for p in g.index]]=g["size"].values
    reps=[]
    for _ in range(1000):
        w=rng.multinomial(len(ids),np.ones(len(ids))/len(ids)); reps.append((v*w).sum()/(n*w).sum())
    res[k]=(v.sum()/n.sum()*100,np.percentile(reps,2.5)*100,np.percentile(reps,97.5)*100)
print("PanelB",res); pd.DataFrame([{"episode":k,"pct":v[0],"lo":v[1],"hi":v[2]} for k,v in res.items()]).to_csv(R/"panelB_cluster.csv",index=False)
