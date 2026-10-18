# seed_detail.py
"""
Training-seed detail (source A): per dataset and ranking, Kendall tau_b between the
three seeds and between the production models and each seed, the leaders and the
top three of each realization. Writes A_training_seed_detail.csv.

Usage (from the repository root):
    python experiments/appendix/ranking_uncertainty/seed_detail.py --data-dir <uncertainty_final>
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import uncertainty as AZ
from scope import DS, SCOPE_A, ST, rm

OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "r1_9_uncertainty_20261016" / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    T0 = AZ.Tables(args.data_dir)
    ACR = T0.c["acronyms"]
    rows = []
    for ds in DS:
        T = ST(T0, ds, **SCOPE_A[ds]); W_ = {t: rm(T, ds, (lambda m, e, t=t: T0.get("A", ds, m, t, e))) for t in ["prod", "s11", "s33", "s55"]}
        for r in AZ.RANKINGS:
            lead = {t: ACR[AZ.leader(W_[t], r, T.E)] for t in W_}
            top3 = {t: sorted(ACR[x] for x in AZ.topk(W_[t], r, T.E)) for t in W_}
            ss = ["s11", "s33", "s55"]
            tss = [AZ.tau(W_[a], W_[b], r) for a, b in itertools.combinations(ss, 2)]
            tps = [AZ.tau(W_["prod"], W_[s], r) for s in ss]
            t3 = [len(set(top3[a]) & set(top3[b])) for a, b in itertools.combinations(ss, 2)]
            rows.append(dict(dataset=ds, ranking=r, n_expl=len(T.E), n_models=len(T.models(ds)),
                             tau_between_seeds_min=round(min(tss), 3), tau_between_seeds_max=round(max(tss), 3), tau_between_seeds_mean=round(sum(tss) / 3, 3),
                             tau_prod_vs_seed_min=round(min(tps), 3), tau_prod_vs_seed_max=round(max(tps), 3),
                             leaders=json.dumps(lead), n_distinct_seed_leaders=len(set(lead[s] for s in ss)),
                             top3_overlap_between_seeds_min=min(t3), top3=json.dumps(top3)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "A_training_seed_detail.csv", index=False)


if __name__ == "__main__":
    main()
