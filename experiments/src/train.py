# train.py
import argparse
import importlib
import json
import logging
import os
from typing import Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    import joblib
except Exception:
    joblib = None
import pickle

from ml_models.experiment_model import ExperimentModel
from ml_models.model_registry import ModelRegistry
from ml_models.model_builder import build_model_with_params, load_arch_params
from performance_metrics import Metrics

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
logging.info(f"[trainer] Using device: {DEVICE}")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_yaml(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def import_callable(path: str):
    module_path, func_name = path.split(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def standardize_adapter_return(ret):
    from torch.utils.data import DataLoader as _DL
    tl = vl = te = None
    nb_classes = None
    if isinstance(ret, dict):
        tl = ret.get("train_loader")
        vl = ret.get("val_loader")
        te = ret.get("test_loader")
        nb_classes = ret.get("nb_classes")
    elif isinstance(ret, (tuple, list)):
        dls = [x for x in ret if isinstance(x, _DL)]
        if len(dls) >= 1:
            tl = dls[0]
        if len(dls) >= 2:
            vl = dls[1]
        if len(dls) >= 3:
            te = dls[2]
        for x in ret:
            if isinstance(x, (int, np.integer)):
                nb_classes = int(x)
                break
    elif isinstance(ret, DataLoader):
        tl = ret
    if tl is None or vl is None:
        raise ValueError("Adapter did not return at least (train_loader, val_loader).")
    return tl, vl, te, nb_classes


def get_loaders(dataset_name: str, datasets_cfg: dict):
    if dataset_name not in datasets_cfg:
        raise ValueError(f"Dataset '{dataset_name}' not in configs/datasets_config.yaml")
    entry = datasets_cfg[dataset_name]
    adapter = import_callable(entry["adapter"])
    args = entry.get("args", {}) or {}
    ret = adapter(**args)
    train_loader, val_loader, test_loader, nb_classes = standardize_adapter_return(ret)
    task = entry["task"]
    if task.lower() == "classification" and nb_classes is None:
        y = getattr(train_loader.dataset, "y", None)
        if isinstance(y, np.ndarray):
            nb_classes = int(y.max()) + 1 if y.ndim == 1 else int(y[:, 0].max()) + 1
        else:
            b = next(iter(train_loader))
            nb_classes = int(b["label"].max().item()) + 1
    return train_loader, val_loader, test_loader, nb_classes, task


def infer_input_shape(loader: DataLoader) -> Tuple[int, int]:
    batch = next(iter(loader))
    x = batch["sequence"]
    if x.dim() != 3:
        raise ValueError(f"expected (B,C,T), got {tuple(x.shape)}")
    _, C, T = x.shape
    return (T, C)


def collect_labels(loader: DataLoader):
    ys = []
    for batch in loader:
        y = batch["label"]
        y = y.numpy() if isinstance(y, np.ndarray) else y.cpu().numpy()
        if y.ndim > 1:
            y = y[:, 0]
        ys.append(y)
    return np.concatenate(ys, axis=0)


def save_model(exp_model: ExperimentModel, out_dir: str, model_name: str):
    os.makedirs(out_dir, exist_ok=True)
    model = exp_model.model
    if getattr(exp_model, "is_sklearn", False):
        path = os.path.join(out_dir, f"{model_name}.pkl")
        if joblib is not None:
            joblib.dump(model, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(model, f)
    elif isinstance(model, list):
        for i, net in enumerate(model):
            path = os.path.join(out_dir, f"{model_name}_ensemble{i}.pt")
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, path)
    else:
        path = os.path.join(out_dir, f"{model_name}.pt")
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)


def evaluate_and_log(exp_model, split, train_loader, val_loader, test_loader, out_dir, model_name):
    loader = test_loader if (split == "test" and test_loader is not None) else val_loader
    y_true = collect_labels(loader)
    y_pred = exp_model.predict(loader)
    result = Metrics().calculate_metrics(exp_model.model_type, y_true, y_pred)
    with open(os.path.join(out_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)


def train_dataset(dataset_name, models_cfg, datasets_cfg, trainer_cfg):
    train_loader, val_loader, test_loader, nb_classes, task = get_loaders(dataset_name, datasets_cfg)
    input_shape = infer_input_shape(train_loader)
    logging.info(f"[{dataset_name}] task={task} nb_classes={nb_classes} input_shape={input_shape}")

    out_dir      = trainer_cfg.get("output_root", "../saved_models")
    params_root  = trainer_cfg.get("params_root",  "../saved_models")
    metrics_split = trainer_cfg.get("metrics_split", "val")
    default_args = trainer_cfg.get("default_train_args", {})
    early_default = default_args.get("early_stopping")

    for model_name, per_model_cfg in models_cfg.items():
        try:
            model_enum = ModelRegistry.get_enum_by_name(model_name)
        except ValueError as e:
            logging.error(f"[{dataset_name}] {e}")
            continue

        logging.info(f"[{dataset_name}] Training {model_name}")

        arch_params = load_arch_params(params_root, dataset_name, model_name) or {}

        try:
            model_obj = build_model_with_params(model_enum, input_shape, nb_classes, arch_params, task)
        except NotImplementedError:
            logging.warning(f"[{dataset_name}/{model_name}] No parameterized build; using ExperimentModel defaults.")
            model_obj = None

        exp_model = ExperimentModel(model_type=model_enum, input_shape=input_shape, nb_classes=nb_classes)
        if model_obj is not None:
            exp_model.model = model_obj
            if not isinstance(model_obj, torch.nn.Module) and not isinstance(model_obj, list):
                exp_model.is_sklearn = True
                exp_model.expects_3d = bool(getattr(model_obj, "expects_3d", False))

        train_args = dict(default_args)
        train_args.update(per_model_cfg.get("train_args", {}) or {})
        epochs = int(arch_params.get("epochs", train_args.get("epochs", 10)))
        lr     = float(arch_params.get("learning_rate", train_args.get("lr", 1e-3)))
        early_stopping = train_args.get("early_stopping", early_default)

        try:
            exp_model.fit(train_loader, epochs=epochs, lr=lr,
                          val_loader=val_loader, early_stopping=early_stopping)
        except Exception as e:
            logging.error(f"[{dataset_name}/{model_name}] Training failed: {e}")
            continue

        flat_name = f"{dataset_name}_{model_name}"
        try:
            save_model(exp_model, out_dir, flat_name)
        except Exception as e:
            logging.error(f"[{dataset_name}/{model_name}] Save failed: {e}")

        if arch_params:
            params_out = os.path.join(out_dir, f"{flat_name}.json")
            if not os.path.exists(params_out):
                with open(params_out, "w") as f:
                    json.dump({"dataset": dataset_name, "model": model_name,
                               "source": "defaults", "params": arch_params}, f, indent=2)

        try:
            evaluate_and_log(exp_model, metrics_split, train_loader, val_loader,
                             test_loader, out_dir, flat_name)
        except Exception as e:
            logging.error(f"[{dataset_name}/{model_name}] Metrics failed: {e}")

        logging.info(f"[{dataset_name}] {model_name} done → {out_dir}/{flat_name}")


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_config", default="configs/datasets_config.yaml")
    ap.add_argument("--model_config",    default="configs/datasets_model_config.yaml")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Dataset names to run (whitelist)")
    args = ap.parse_args()

    datasets_cfg = load_yaml(args.datasets_config)
    model_cfg    = load_yaml(args.model_config)
    trainer_cfg  = model_cfg.get("trainer", {})
    datasets_block = model_cfg.get("datasets", {})

    for dataset_name, cfg in datasets_block.items():
        if args.only and dataset_name not in args.only:
            continue
        models_cfg = cfg.get("models", {})
        if not models_cfg:
            logging.info(f"[{dataset_name}] No models configured; skipping.")
            continue
        logging.info(f"\n=== {dataset_name} ===")
        train_dataset(dataset_name, models_cfg, datasets_cfg, trainer_cfg)

    logging.info("All done.")


if __name__ == "__main__":
    main()
