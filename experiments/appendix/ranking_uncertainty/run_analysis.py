# run_analysis.py
"""
Uncertainty of the overall and pillar rankings (Appendix C.3), each source against
the reference ranking of the production models:

  A  training seeds 11/33/55           C1 independent FC subsets (M = 25..400)
  B  explainer seeds (B2, B3)          C2 independent robustness draws (M = 5..20)
  D  evaluation samples (500 paired resamples; Ecoating: leave-one-out)
  F1 FC subset size q                  F2 robustness noise scale sigma

--data-dir holds metrics/, metric_mc/, parameter_sensitivity/ and data/<ds>/sample_manifest.csv
of the uncertainty study. Writes 0_reference_vs_frozen_manuscript.json,
summary_all_sources.csv, rankings_per_world.csv and checks.json.

Usage (from the repository root):
    python experiments/appendix/ranking_uncertainty/run_analysis.py --data-dir <uncertainty_final>
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import uncertainty as AZ
from scope import DS, SCOPE_A, SCOPE_B, ST, names, rm

_ROOT = Path(__file__).resolve().parents[3]
PUBLISHED = _ROOT / "results" / "main_benchmark_14_promethee" / "output"
OUT_DIR = _ROOT / "results" / "r1_9_uncertainty_20261016" / "outputs"


def check(T, ds, getters):
    miss = []
    for m in T.models(ds):
        for e in T.E:
            if T.sup(ds, m, e):
                for name, g in getters.items():
                    if g(m, e) is None: miss.append((name, m, e))
    return miss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--published", type=Path, default=PUBLISHED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    UF, OUT = args.data_dir, str(args.out_dir)
    os.makedirs(OUT, exist_ok=True)

    T0 = AZ.Tables(UF)
    ACR = T0.c["acronyms"]; STOCH = T0.c["stochastic_explainers"]
    log = {}; SUMM = []; RANKS = []

    def add(source, ds, T, ref, worlds, labels, scope_note, conditional=False):
        acc, pw, s = AZ.summarize(ref, worlds, T.E, source, ds, conditional)
        for r in AZ.RANKINGS:
            SUMM.append({"source": source, "dataset": ds, "ranking": r, "scope": scope_note, "n_explainers": len(T.E),
                         "n_models": len(T.models(ds)), "n_worlds": len(worlds), **s[r]})
        for lab, w in zip(labels, worlds):
            for r in AZ.RANKINGS:
                RANKS.append({"source": source, "dataset": ds, "world": lab, "ranking": r, "order": " > ".join(names(T, w, ACR, r))})

    # ---------- 0. reference (production models, full scope) vs the published ranking
    integ = {}
    for ds in DS:
        T = ST(T0, ds); ref = rm(T, ds, lambda m, e: T0.get("A", ds, m, "prod", e))
        pub = pd.read_csv(args.published / f"promethee_ranking_{ds}.csv")
        pub = pub.set_index("Explainer").promethee_net_flow.reindex(T.E)
        integ[ds] = {"controlled_reference_order": names(T, ref, ACR), "frozen_manuscript_order": [ACR[e] for e in pub.sort_values(ascending=False).index],
                     "same_order": names(T, ref, ACR) == [ACR[e] for e in pub.sort_values(ascending=False).index],
                     "max_abs_netflow_diff": float(np.nanmax(np.abs(ref["overall"].phi - pub.to_numpy()))),
                     "tau_b": float(kendalltau(ref["overall"].phi, pub.to_numpy()).statistic)}
    json.dump(integ, open(f"{OUT}/0_reference_vs_frozen_manuscript.json", "w"), indent=1)
    print(json.dumps({d: {k: v for k, v in x.items() if k in ("same_order", "max_abs_netflow_diff", "tau_b")} for d, x in integ.items()}, indent=0))

    # ---------- A training seeds
    for ds in DS:
        T = ST(T0, ds, **SCOPE_A[ds]); g = {t: (lambda m, e, t=t: T0.get("A", ds, m, t, e)) for t in ["prod", "s11", "s33", "s55"]}
        miss = check(T, ds, g); assert not miss, (ds, miss[:5])
        ref = rm(T, ds, g["prod"]); ws = [rm(T, ds, g[t]) for t in ["s11", "s33", "s55"]]
        add("A_training_seed", ds, T, ref, ws, ["s11", "s33", "s55"], json.dumps(SCOPE_A[ds]))
        RANKS.append({"source": "A_training_seed", "dataset": ds, "world": "prod (reference)", "ranking": "overall", "order": " > ".join(names(T, ref, ACR))})

    # ---------- B explainer seeds (B1 = production maps)
    for ds in DS:
        T = ST(T0, ds, **SCOPE_B[ds]); prod = lambda m, e: T0.get("A", ds, m, "prod", e)
        gb = {b: (lambda m, e, b=b: T0.get(b, ds, m, "prod", e) if e in STOCH else prod(m, e)) for b in ["B2", "B3"]}
        miss = check(T, ds, gb); assert not miss, (ds, miss[:5])
        ref = rm(T, ds, prod); ws = [rm(T, ds, gb[b]) for b in ["B2", "B3"]]
        add("B_explainer_rng", ds, T, ref, ws, ["B2", "B3"], json.dumps(SCOPE_B[ds]))

    # ---------- C metric random draws and F metric parameters (production maps, all 14 explainers)
    for ds in DS:
        T = ST(T0, ds); prod = lambda m, e: T0.get("A", ds, m, "prod", e); ref = rm(T, ds, prod)
        fc = pd.concat([pd.read_csv(p) for p in sorted((UF / "metric_mc" / "fc" / ds).glob("*.csv"))])
        rb = pd.concat([pd.read_csv(p) for p in sorted((UF / "metric_mc" / "robust" / ds).glob("*.csv"))])
        fq = pd.concat([pd.read_csv(p) for p in sorted((UF / "parameter_sensitivity" / "fc_q" / ds).glob("*.csv"))])
        sg = pd.concat([pd.read_csv(p) for p in sorted((UF / "parameter_sensitivity" / "sigma" / ds).glob("*.csv"))])
        fcg = {k: g.sort_values("sample_index").FC_abs.to_numpy() for k, g in fc.groupby(["model", "explainer", "realization", "M"])}
        rbg = {k: g.sort_values("sample_index")[["AvgSen", "Cont"]].to_numpy() for k, g in rb.groupby(["model", "explainer", "realization", "M"])}
        qg = {k: g.sort_values("sample_index").FC_abs.to_numpy() for k, g in fq.groupby(["model", "explainer", "q"])}
        sgg = {k: g.sort_values("sample_index")[["AvgSen", "Cont"]].to_numpy() for k, g in sg.groupby(["model", "explainer", "sigma"])}
        # realization 1 with M=100 is the production FC
        dev = max(float(np.nanmax(np.abs(fcg[(m, e, 1, 100)] - prod(m, e)[:, 0]))) for m in T.models(ds) for e in T.E if T.sup(ds, m, e))
        log[f"{ds}_C1_r1M100_vs_prod_FC_maxabs"] = dev
        def sub(cols, arr):
            def f(m, e):
                v = prod(m, e).copy(); v[:, cols] = arr(m, e); return v
            return f
        for real in (1, 2):
            for M_ in (25, 50, 100, 200, 400):
                add(f"C1_FC_subsets_M{M_}", ds, T, ref, [rm(T, ds, sub(0, lambda m, e: fcg[(m, e, real, M_)]))], [f"r{real}"], "full")
            for M_ in (5, 10, 20):
                add(f"C2_robust_draws_M{M_}", ds, T, ref, [rm(T, ds, sub([2, 3], lambda m, e: rbg[(m, e, real, M_)]))], [f"r{real}"], "full")
        for q in sorted(fq.q.unique()):
            add(f"F1_q{q}", ds, T, ref, [rm(T, ds, sub(0, lambda m, e: qg[(m, e, q)]))], [f"q={q}"], "full")
        for s_ in sorted(sg.sigma.unique()):
            add(f"F2_sigma{s_}", ds, T, ref, [rm(T, ds, sub([2, 3], lambda m, e: sgg[(m, e, s_)]))], [f"sigma={s_}"], "full")

    # ---------- D evaluation samples (paired resampling, B = 500)
    rng = np.random.default_rng(AZ.RNG_D); B = 500
    for ds in ["Ecoating", "Hydraulic", "FD001", "CWRU_12k"]:
        T = ST(T0, ds); prod = lambda m, e: T0.get("A", ds, m, "prod", e); ref = rm(T, ds, prod)
        grp = np.array([r["group"] for r in csv.DictReader(open(UF / "data" / ds / "sample_manifest.csv"))])
        y = np.array([float(r["target"]) for r in csv.DictReader(open(UF / "data" / ds / "sample_manifest.csv"))])
        N = len(grp)
        if ds == "Ecoating":
            idxs = [np.array([j for j in range(N) if j != i]) for i in range(N)]; kind = "leave-one-out (N=5)"
        elif ds == "FD001":
            ug = np.unique(grp); pos = {g: np.where(grp == g)[0] for g in ug}
            idxs = [np.concatenate([pos[g] for g in rng.choice(ug, size=len(ug), replace=True)]) for _ in range(B)]; kind = f"engine-cluster bootstrap B={B}"
        elif ds == "Hydraulic":
            ug = np.unique(grp); pos = {g: np.where(grp == g)[0] for g in ug}; by = {}
            for g in ug: by.setdefault(int(y[pos[g][0]]), []).append(g)
            idxs = [np.concatenate([pos[g] for v in by.values() for g in rng.choice(v, size=len(v), replace=True)]) for _ in range(B)]; kind = f"class-stratified block bootstrap B={B}"
        else:
            L = 4; st = np.arange(0, N - L + 1)
            idxs = [np.concatenate([np.arange(s, s + L) for s in rng.choice(st, size=int(np.ceil(N / L)), replace=True)])[:N] for _ in range(B)]; kind = f"moving-block bootstrap (4 windows) B={B}"
        ws = [rm(T, ds, prod, idx=ix) for ix in idxs]
        add("D_sample_composition", ds, T, ref, ws, [f"b{i}" for i in range(len(ws))], kind)
        log[f"{ds}_D_groups"] = int(len(np.unique(grp)))

    pd.DataFrame(SUMM).to_csv(f"{OUT}/summary_all_sources.csv", index=False)
    pd.DataFrame(RANKS).to_csv(f"{OUT}/rankings_per_world.csv", index=False)
    json.dump(log, open(f"{OUT}/checks.json", "w"), indent=1); print(json.dumps(log, indent=0))


if __name__ == "__main__":
    main()
