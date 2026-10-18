# uncertainty.py
"""
Ranking functions of the uncertainty analysis (Appendix C.3).

A world's metric matrix is the skip-NaN mean over models of the per-model,
direction-aligned sample median of each metric; it is ranked with the PROMETHEE II
code of experiments/analysis/promethee_ranking.py (overall: equal pillar weights;
pillar ranking: all weight on that pillar), ties broken by explainer name.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

_ANALYSIS = Path(__file__).resolve().parents[2] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

STUDY_CONFIG = Path(__file__).resolve().parent / "study_config.json"
METRICS = ["FC_abs", "PF_AUC", "AvgSen", "Cont", "Spars_E", "Spars_C", "Compl_E", "Compl_C"]
SIGN = np.array([1, -1, -1, -1, 1, 1, -1, -1], dtype=float)   # FC+, PF-, AvgSen-, Cont-, Spars-E+, Spars-C+, Compl-E-, Compl-C-
RANKINGS = ["overall", "pillar_Faithfulness", "pillar_Robustness", "pillar_Complexity"]
RNG_D = 20261011
_W = None


def pr():
    import promethee_ranking as P
    return P


class RV:
    __slots__ = ("phi", "rank")

    def __init__(self, phi, rank):
        self.phi, self.rank = phi, rank


def rank_matrix(M: np.ndarray, E: list) -> dict:
    global _W
    P = pr()
    if _W is None:
        _W = {}
        for r in RANKINGS:
            pw = dict(P.DEFAULT_PILLAR_WEIGHTS) if r == "overall" else {k: (1.0 if r == f"pillar_{k}" else 0.0) for k in P.DEFAULT_PILLAR_WEIGHTS}
            _W[r] = P.pillar_weights_to_metric_weights(pw, P.COLS)
    df = pd.DataFrame(M, index=E, columns=P.COLS)
    out = {}
    for r in RANKINGS:
        phi = P.promethee_net_flow(df, _W[r]).to_numpy()
        order = sorted(range(len(E)), key=lambda j: (-phi[j], E[j]))
        rank = np.empty(len(E), dtype=np.int16)
        rank[order] = np.arange(1, len(E) + 1)
        out[r] = RV(phi, rank)
    return out


def tau(a, b, r):
    return float(kendalltau(a[r].phi, b[r].phi).statistic)


def leader(w, r, E):
    return E[int(np.argmin(w[r].rank))]


def topk(w, r, E, k=3):
    return {E[j] for j in np.where(w[r].rank <= k)[0]}


# ------------------------------------------------------------------ data
class Tables:
    def __init__(self, data_dir: Path):
        self.c = json.load(open(STUDY_CONFIG))
        self.data_dir = Path(data_dir)
        self.E = sorted(self.c["explainers"])
        self.EI = {e: j for j, e in enumerate(self.E)}
        self.cache = {}

    def get(self, regime, ds, model, tag, e):
        k = (regime, ds, model, tag, e)
        if k not in self.cache:
            p = self.data_dir / "metrics" / regime / ds / model / tag / f"{e}.csv"
            self.cache[k] = pd.read_csv(p)[METRICS].to_numpy(dtype=float) if p.exists() else None
        return self.cache[k]

    def models(self, ds):
        return self.c["datasets"][ds]["models"]

    def sup(self, ds, m, e):
        return e in self.c["support"][ds][m]


def med(vals, idx=None):
    v = vals if idx is None else vals[idx]
    with np.errstate(all="ignore"):
        return SIGN * np.nanmedian(v, axis=0)


def model_matrix(T, ds, m, getter, idx=None):
    row = np.full((len(T.E), 8), np.nan)
    for j, e in enumerate(T.E):
        if T.sup(ds, m, e):
            v = getter(m, e)
            if v is None:
                raise RuntimeError(f"missing scores {ds}/{m}/{e}")
            row[j] = med(v, idx)
    return row


def world_matrix(T, ds, getter, idx=None):
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack([model_matrix(T, ds, m, getter, idx) for m in T.models(ds)]), axis=0)


# ------------------------------------------------------------------ summaries
def summarize(ref, worlds, E, label, ds, conditional=False):
    acc, pw, summ = [], [], {}
    w = np.ones(len(worlds)) / len(worlds)
    for r in RANKINGS:
        R = np.stack([wd[r].rank for wd in worlds]).astype(int)
        Rr = ref[r].rank.astype(int)
        t3 = topk(ref, r, E)
        ld = leader(ref, r, E)
        same_t3 = np.array([topk(wd, r, E) == t3 for wd in worlds])
        same_ld = R[:, E.index(ld)] == 1
        taus = np.array([tau(ref, wd, r) for wd in worlds])
        disp = np.abs(R - Rr[None, :])
        for j, e in enumerate(E):
            acc.append({"source": label, "dataset": ds, "ranking": r, "explainer": e, "reference_rank": int(Rr[j]),
                        "P_rank_1": float(w @ (R[:, j] == 1)), "P_rank_le_3": float(w @ (R[:, j] <= 3)), "expected_rank": float(w @ R[:, j]),
                        "min_rank": int(R[:, j].min()), "max_rank": int(R[:, j].max()), "n_worlds": len(worlds),
                        "conditional_on_observed_realizations": conditional})
        if r == "overall":
            for a, b in itertools.combinations(range(len(E)), 2):
                pw.append({"source": label, "dataset": ds, "explainer_A": E[a], "explainer_B": E[b], "P_A_above_B": float(w @ (R[:, a] < R[:, b])),
                           "reference_A_above_B": bool(Rr[a] < Rr[b]), "conditional_on_observed_realizations": conditional})
        p1 = (R == 1).mean(axis=0)
        summ[r] = {"kendall_tau_b_vs_reference_mean": float(np.nanmean(taus)), "kendall_tau_b_vs_reference_min": float(np.nanmin(taus)),
                   "P_leader_retained": float(same_ld.mean()), "P_top3_set_unchanged": float(same_t3.mean()),
                   "mean_rank_displacement": float(disp.mean()), "max_rank_displacement": int(disp.max()),
                   "leading_group": json.dumps(sorted(E[j] for j in np.where(p1 > 0)[0])), "reference_leader": ld,
                   "reference_top3": json.dumps(sorted(t3))}
        if len(worlds) <= 50:
            t = [tau(a, b, r) for a, b in itertools.combinations(worlds, 2)]
            summ[r].update(pairwise_tau_b_between_replicates_mean=float(np.mean(t)) if t else None,
                           pairwise_tau_b_between_replicates_min=float(np.min(t)) if t else None)
    return acc, pw, summ
