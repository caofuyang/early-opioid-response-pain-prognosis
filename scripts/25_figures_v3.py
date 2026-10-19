import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = str(ROOT / "results" / "figures")
C={"MOVER":"#1f3b73","MIMIC-IV":"#b23a2e"}
m=pd.read_csv(ROOT / "results" / "pain_response_v2" / "models_v31.csv")
mov=m[m.analysis=="MOVER success"].sort_values("group")
mim=m[m.analysis=="MIMIC success"].sort_values("group")
fig,axes=plt.subplots(1,2,figsize=(9.2,3.6))
ax=axes[0]
ax.errorbar(mov.group,mov.estimate*100,yerr=[(mov.estimate-mov.ci_low)*100,(mov.ci_high-mov.estimate)*100],
            marker="o",ms=5,lw=1.8,capsize=3,color=C["MOVER"],label="MOVER")
ax.errorbar(mim.group,mim.estimate*100,
            yerr=[(mim.estimate-mim.ci_low)*100,(mim.ci_high-mim.estimate)*100],
            marker="o",ms=5,lw=1.8,capsize=3,color=C["MIMIC-IV"],label="MIMIC-IV")
ax.set_xticks(range(4)); ax.set_xticklabels(["0","1","2","≥3"]); ax.set_xlabel("Previous failed episodes"); ax.set_ylabel("Response (%)"); ax.set_ylim(0,100)
ax.set_title("A. Response by previous failures",loc="left",fontsize=10); ax.legend(frameon=False,fontsize=8)
ax=axes[1]
d=pd.read_parquet(ROOT / "results" / "pain_response_v2" / "rescue_episodes_v2.parquet").sort_values(["LOG_ID","dose_time"])
mrn = d.drop_duplicates("LOG_ID").set_index("LOG_ID").MRN
d["seq"]=d.groupby("LOG_ID").cumcount()+1; d["mcid"]=d.success
piv=d[d.seq<=3].pivot_table(index="LOG_ID",columns="seq",values="mcid",aggfunc="first").dropna()
pb=pd.read_csv(ROOT / "results" / "pain_response_v2" / "panelB_cluster.csv").sort_values("episode")
p=pb.set_index("episode")["pct"]/100; n=len(piv)
ax.errorbar([1,2,3],pb["pct"],yerr=[pb["pct"]-pb["lo"],pb["hi"]-pb["pct"]],marker="s",ms=6,lw=1.8,capsize=3,color=C["MOVER"])
ax.set_xticks([1,2,3]); ax.set_xlabel("Episode number"); ax.set_ylim(0,100); ax.set_title("B. Cases with ≥3 episodes",loc="left",fontsize=10)
ax.text(0.02,0.04,f"n = {n:,} cases, {mrn.loc[piv.index].nunique():,} patients",transform=ax.transAxes,fontsize=8,color="0.4")
for a in axes: a.spines[["top","right"]].set_visible(False); a.grid(axis="y",color="0.9",lw=0.6)
fig.tight_layout()
for ext,kw in [("png",dict(dpi=300)),("pdf",{}),("tiff",dict(dpi=600,pil_kwargs={"compression":"tiff_lzw"}))]:
    fig.savefig(f"{OUT}/Figure_2_Response_Probability.{ext}",**kw)
print("saved")
