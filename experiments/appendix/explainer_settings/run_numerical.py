# run_numerical.py
"""
Numerical settings: attribution maps of the step, sample and background-size grids.

  IG n_steps            {10,25,50,100,200}
  SmoothGrad n_samples  {5,10,20,50,100} x 5 seeds
  GradientSHAP n_samples{5,10,20,50,100} x 5 seeds, background M=50
  EG steps              {10,25,50,100,200}, background M=50
  EG background M       {10,25,50,100,200}, steps=50
  DeepSHAP background M {10,25,50,100,200}

Cached maps are reused. Run from experiments/ (dataset paths are relative to it):
    python appendix/explainer_settings/run_numerical.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import common as C
from xai_methods import ExplainerFactory

IG_STEPS_GRID = [10, 25, 50, 100, 200]
SG_N_GRID = [5, 10, 20, 50, 100]
GS_N_GRID = [5, 10, 20, 50, 100]
EG_STEPS_GRID = [10, 25, 50, 100, 200]
M_GRID = [10, 25, 50, 100, 200]


def run_model(model: str, results_dir: Path):
    print(f"=== {model} ===", flush=True)
    attr_dir = results_dir / "attributions"
    exp_model, xai_model, task, train_loader = C.load_model_cpu(model)
    X, y = C.load_hydraulic_30(results_dir)

    bg50, baseline_mean50 = C.build_background(train_loader, kmeans_k=50)

    ig = ExplainerFactory.get("integrated_gradients", xai_model)
    sg = ExplainerFactory.get("smooth_gradient", xai_model)
    gs = ExplainerFactory.get("gradientshap", xai_model)
    eg = ExplainerFactory.get("expected_gradients", xai_model)
    dls = ExplainerFactory.get("deepliftshap", xai_model)

    for steps in IG_STEPS_GRID:
        C.get_or_compute(
            attr_dir / f"{model}__ig__steps{steps}.npy",
            lambda s=steps: ig.explain(x_val=X, task=task, device="cpu", steps=s, baseline=baseline_mean50),
        )

    for seed in C.SEEDS:
        for n in SG_N_GRID:
            def _compute(n=n, seed=seed):
                torch.manual_seed(seed)
                return sg.explain(x_val=X, task=task, device="cpu", smooth_samples=n, noise_std=0.10)
            C.get_or_compute(attr_dir / f"{model}__smoothgrad__n{n}__seed{seed}.npy", _compute)

    for seed in C.SEEDS:
        for n in GS_N_GRID:
            def _compute(n=n, seed=seed):
                torch.manual_seed(seed)
                return gs.explain(x_val=X, task=task, device="cpu", background_data=bg50,
                                  smooth_samples=n, noise_std=0.10)
            C.get_or_compute(attr_dir / f"{model}__gradientshap__n{n}__seed{seed}.npy", _compute)

    for steps in EG_STEPS_GRID:
        C.get_or_compute(
            attr_dir / f"{model}__eg_steps__steps{steps}.npy",
            lambda s=steps: eg.explain(x_val=X, task=task, device="cpu", background_data=bg50, steps=s),
        )

    # each background size is a different set of KMeans centroids, shared by EG and DeepSHAP
    bg_by_M = {50: bg50}
    for M in M_GRID:
        if M not in bg_by_M:
            bg_by_M[M], _ = C.build_background(train_loader, kmeans_k=M)

    for M in M_GRID:
        C.get_or_compute(
            attr_dir / f"{model}__eg_M__M{M}.npy",
            lambda m=M: eg.explain(x_val=X, task=task, device="cpu", background_data=bg_by_M[m], steps=50),
        )
    for M in M_GRID:
        C.get_or_compute(
            attr_dir / f"{model}__deepshap_M__M{M}.npy",
            lambda m=M: dls.explain(x_val=X, task=task, device="cpu", background_data=bg_by_M[m]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR)
    args = parser.parse_args()
    (args.results_dir / "attributions").mkdir(parents=True, exist_ok=True)
    for model in C.MODELS:
        run_model(model, args.results_dir)
