# fig_training.py
"""
Figure C.2: overall rank of each explainer with the production models (reference)
and with the models retrained with seeds 11, 33 and 55 (scope of source A).
Writes Figure_C2_training_seed_ranks.{pdf,png} and Figure_C2_training_seed_ranks_data.csv.

Usage (from the repository root):
    python experiments/appendix/ranking_uncertainty/fig_training.py --data-dir <uncertainty_final>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd

import uncertainty as AZ
from scope import SCOPE_A, ST, rm

OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "r1_9_uncertainty_20261016" / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    T0 = AZ.Tables(args.data_dir)
    ACR = T0.c["acronyms"]

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.edgecolor":"black","text.color":"black","axes.labelcolor":"black","axes.linewidth":0.6,
                         "xtick.color":"black","ytick.color":"black","xtick.major.width":0.6,"ytick.major.width":0})
    ORDER=["Ecoating","FD001","CWRU_12k","Hydraulic"]; TITLE={"Ecoating":"Ecoating","FD001":"C-MAPSS","CWRU_12k":"CWRU","Hydraulic":"Hydraulic"}
    ST_={"prod":dict(c="#1f1f1d",m="D",s=22,lab="Production models (reference)"),
         "s11":dict(c="#2a78d6",m="o",s=16,lab="Seed 11"),"s33":dict(c="#eb6834",m="s",s=15,lab="Seed 33"),"s55":dict(c="#1baf7a",m="^",s=18,lab="Seed 55")}
    OFF={"prod":0.0,"s11":-0.22,"s33":0.0,"s55":0.22}
    fig,axs=plt.subplots(1,4,figsize=(7.2,3.1),gridspec_kw=dict(wspace=0.85))
    rows=[]
    for ax,ds in zip(axs,ORDER):
        T=ST(T0,ds,**SCOPE_A[ds]); W_={t: rm(T,ds,(lambda m,e,t=t: T0.get("A",ds,m,t,e))) for t in ["prod","s11","s33","s55"]}
        E=T.E; n=len(E); ref=W_["prod"]["overall"].rank
        order=sorted(range(n),key=lambda j: ref[j])
        ypos={j:i for i,j in enumerate(order)}
        ax.axvspan(0.5,3.5,color="#ecebe6",zorder=0,lw=0)
        for i in range(n): ax.axhline(i,color="#e3e2dc",lw=0.5,zorder=0.5)
        for t,st in ST_.items():
            rk=W_[t]["overall"].rank
            ax.scatter([rk[j] for j in order],[ypos[j]+OFF[t] for j in order],c=st["c"],marker=st["m"],s=st["s"],
                       edgecolors="white",linewidths=0.5,zorder=3 if t!="prod" else 4,label=st["lab"])
            for j in order: rows.append(dict(dataset=ds,explainer=ACR[E[j]],world=t,rank=int(rk[j])))
        ax.set_yticks(range(n)); ax.set_yticklabels([ACR[E[j]].replace("PermutationSHAP","SHAP") for j in order])
        ax.set_ylim(n-0.5,-0.5); ax.set_xlim(0.4,n+0.6); ax.set_xticks([1,5,10,n] if n>=14 else [1,5,10,n])
        ax.set_title(TITLE[ds],fontsize=7.5,color="black",pad=4)
        ax.set_xlabel("Overall rank",color="black"); ax.tick_params(axis="y",length=0)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
    h,l=axs[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.02),handletextpad=0.3,columnspacing=1.4,labelcolor="black")
    fig.subplots_adjust(left=0.105,right=0.99,top=0.92,bottom=0.2)
    O = args.out_dir; O.mkdir(parents=True, exist_ok=True)
    fig.savefig(O / "Figure_C2_training_seed_ranks.pdf"); fig.savefig(O / "Figure_C2_training_seed_ranks.png", dpi=220)
    pd.DataFrame(rows).to_csv(O / "Figure_C2_training_seed_ranks_data.csv", index=False)


if __name__ == "__main__":
    main()
