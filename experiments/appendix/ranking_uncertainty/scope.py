# scope.py
"""
Scope of the uncertainty analysis: nine models per dataset, all 14 explainers, with
the restrictions forced by incomplete runs of the stochastic SHAP explainers. Every
restriction is applied to the reference and to the compared worlds.

  A (training seeds):   FD001 without TCN_REGRESSOR; CWRU_12k without KernelSHAP
  B (explainer seeds):  FD001 without TCN_REGRESSOR; CWRU_12k without KernelSHAP and PermutationSHAP
"""
import numpy as np

import uncertainty as AZ

DS = ["Hydraulic", "CWRU_12k", "FD001", "Ecoating"]
SCOPE_A = {"Hydraulic": {}, "Ecoating": {}, "FD001": {"drop_models": ["TCN_REGRESSOR"]}, "CWRU_12k": {"drop_expl": ["kernelshap"]}}
SCOPE_B = {"Hydraulic": {}, "Ecoating": {}, "FD001": {"drop_models": ["TCN_REGRESSOR"]}, "CWRU_12k": {"drop_expl": ["kernelshap", "shap_permutation"]}}


class ST:
    """Tables restricted to a subset of models and explainers of one dataset."""
    def __init__(s, T0, ds, drop_models=(), drop_expl=()):
        s.T0 = T0; s.c = T0.c; s.E = [e for e in T0.E if e not in drop_expl]; s.ds = ds
        s._m = [m for m in primary_models(T0, ds) if m not in drop_models]; s.cache = T0.cache
    def get(s, *a): return s.T0.get(*a)
    def models(s, ds): return s._m
    def sup(s, ds, m, e): return s.T0.sup(ds, m, e) and e in s.E


def primary_models(T0, ds):
    return [m for m in T0.models(ds) if not (ds == "Hydraulic" and m == "MCD_CNN")]


def rm(T, ds, getter, idx=None):
    return AZ.rank_matrix(AZ.world_matrix(T, ds, getter, idx), T.E)


def names(T, w, acr, r="overall"):
    return [acr[T.E[j]] for j in np.argsort(w[r].rank)]
