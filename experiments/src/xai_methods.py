import numpy as np
import torch
import warnings
from typing import Optional, List, Union

from captum.attr import (
    FeatureAblation,
    InputXGradient,
    DeepLiftShap,
)

Tensor = torch.Tensor
NDArray = np.ndarray

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer as _LimeTabularExplainer
    _HAVE_LIME = True
except Exception:
    _HAVE_LIME = False
    _LimeTabularExplainer = None  # type: ignore


# ---------------- Base ----------------
class BaseExplainer:
    def __init__(self, model: Union[torch.nn.Module, List[torch.nn.Module]]):
        self.model = model
        self.is_ensemble = isinstance(model, (list, tuple, torch.nn.ModuleList))

    @staticmethod
    def _resolve_device(model, device: Optional[str]):
        if device in (None, "auto"):
            m0 = model[0] if isinstance(model, (list, tuple, torch.nn.ModuleList)) else model
            try:
                return next(m0.parameters()).device
            except Exception:
                return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _prepare_inputs(x_val: NDArray, device: torch.device) -> Tensor:
        return torch.as_tensor(x_val, dtype=torch.float32, device=device)

    @staticmethod
    def _to_numpy(x: Tensor) -> NDArray:
        return x.detach().to("cpu").numpy()

    @staticmethod
    def _select_targets(
        m: torch.nn.Module,
        inputs: Tensor,
        task: str,
        target_class: Optional[int],
    ) -> Optional[Tensor]:
        task = (task or "classification").lower()
        with torch.no_grad():
            out = m(inputs)
            if isinstance(out, (tuple, list)):
                out = out[0]

        if task == "classification":
            if target_class is None:
                return out.argmax(dim=1)
            B = inputs.size(0)
            return torch.full((B,), int(target_class), dtype=torch.long, device=inputs.device)
        else:
            if out.dim() == 1 or (out.dim() == 2 and out.size(1) == 1):
                return None
            return torch.zeros(inputs.size(0), dtype=torch.long, device=inputs.device)

    @staticmethod
    def _expand_baseline_to_batch(inputs: Tensor, base: Tensor) -> Tensor:
        base = base.to(device=inputs.device, dtype=inputs.dtype)
        if base.dim() == 0 or (base.numel() == 1 and base.dim() <= 1):
            return base.view(1).expand_as(inputs)
        if base.dim() == inputs.dim() - 1 and base.shape == inputs.shape[1:]:
            return base.unsqueeze(0).expand(inputs.size(0), *base.shape)
        if base.dim() == inputs.dim() and base.shape[0] == 1 and base.shape[1:] == inputs.shape[1:]:
            return base.expand_as(inputs)
        if base.dim() == inputs.dim() and base.shape == inputs.shape:
            return base
        raise ValueError(
            f"Baseline shape {tuple(base.shape)} is incompatible with input {tuple(inputs.shape)}"
        )


# ---------------- Feature Ablation ----------------
class FeatureAblationExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        baseline: Optional[NDArray] = None,
        group_mode: str = "none",
        time_window: int = 10,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)

        if baseline is None:
            base = torch.zeros_like(inputs[0])
        else:
            base = torch.as_tensor(baseline, dtype=torch.float32, device=device)
        base = self._expand_baseline_to_batch(inputs, base)

        C, T = inputs.shape[1], inputs.shape[2]
        feature_mask = None
        gm = (group_mode or "none").lower()
        if gm == "channel":
            feature_mask = torch.arange(C, device=device, dtype=torch.long).view(C, 1).expand(C, T)
        elif gm == "time_window":
            tw = max(1, int(time_window))
            time_bins = torch.arange(T, device=device, dtype=torch.long) // tw
            feature_mask = time_bins.view(1, T).expand(C, T)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            fabl = FeatureAblation(m)
            targets = self._select_targets(m, inputs, task, target_class)
            a = fabl.attribute(inputs, target=targets, baselines=base, feature_mask=feature_mask)
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


# ---------------- Gradient x Input ----------------
class GradientXInputExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            gxi = InputXGradient(m)
            targets = self._select_targets(m, inputs, task, target_class)
            a = gxi.attribute(inputs, target=targets)
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


# ---------------- DeepLiftShap (DeepSHAP) ----------------
class DeepLiftShapExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        background_data: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        if background_data is None:
            raise ValueError("DeepLiftShap requires background_data.")
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)
        baselines = torch.as_tensor(background_data, dtype=torch.float32, device=device)
        if baselines.dim() == 2:
            baselines = baselines.unsqueeze(0)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            targets = self._select_targets(m, inputs, task, target_class)
            dls = DeepLiftShap(m)
            a = dls.attribute(inputs, baselines=baselines, target=targets)
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


# ---------------- LIME ----------------
class LimeTabularTimeseriesExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        background_data: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        num_features: Optional[int] = None,
        num_samples: int = 1000,
        class_names: Optional[List[str]] = None,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        if not _HAVE_LIME:
            warnings.warn("LIME is not installed; returning zeros.")
            return np.zeros_like(x_val, dtype=np.float32)

        if background_data is None:
            raise ValueError("LimeTabularTimeseriesExplainer requires background_data.")

        x_np: NDArray = np.asarray(x_val, dtype=np.float32)
        bg_np: NDArray = np.asarray(background_data, dtype=np.float32)

        if x_np.ndim != 3:
            raise ValueError(f"Expected x_val shape (B,C,T), got {x_np.shape}")
        B, C, T = x_np.shape
        D = C * T

        X = x_np.reshape(B, D)
        BG = bg_np.reshape(bg_np.shape[0], D) if bg_np.ndim == 3 else bg_np.reshape(1, -1)
        if BG.shape[1] != D:
            raise ValueError("background_data must be (M,C,T) to match input (B,C,T).")

        mode = "classification" if (task or "classification").lower() == "classification" else "regression"
        num_features = int(num_features or D)

        device_model = self._resolve_device(self.model, device)
        models = self.model if self.is_ensemble else [self.model]

        def _predict_fn(Xtab: NDArray) -> NDArray:
            Xb = torch.from_numpy(Xtab.astype(np.float32)).to(device_model).view(-1, C, T)
            with torch.no_grad():
                outs = []
                for m in models:
                    m.to(device_model).eval()
                    out = m(Xb)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    outs.append(out)
                out = torch.stack([o if o.dim() > 1 else o.unsqueeze(1) for o in outs], dim=0).mean(dim=0)
                if mode == "classification":
                    if out.size(-1) > 1:
                        prob = torch.softmax(out, dim=1)
                    else:
                        p1 = torch.sigmoid(out)
                        prob = torch.cat([1 - p1, p1], dim=1)
                    return prob.detach().cpu().numpy()
                else:
                    return out.detach().cpu().numpy()

        explainer = _LimeTabularExplainer(
            BG,
            mode=mode,
            feature_names=[f"f{i}" for i in range(D)],
            class_names=class_names,
            discretize_continuous=False,
            verbose=False,
        )

        attributions = np.zeros((B, D), dtype=np.float32)
        for i in range(B):
            xi = X[i]
            if mode == "classification":
                if target_class is None:
                    probs = _predict_fn(xi.reshape(1, -1))
                    cls = int(probs.argmax(axis=1).item())
                else:
                    cls = int(target_class)
                exp = explainer.explain_instance(
                    xi, _predict_fn, labels=(cls,), num_features=num_features, num_samples=int(num_samples)
                )
                weights = np.zeros(D, dtype=np.float32)
                for j, w in exp.local_exp[cls]:
                    weights[j] = w
            else:
                exp = explainer.explain_instance(
                    xi, _predict_fn, num_features=num_features, num_samples=int(num_samples)
                )
                weights = np.zeros(D, dtype=np.float32)
                for j, w in exp.local_exp[0]:
                    weights[j] = w

            attributions[i] = np.abs(weights) if return_abs else weights

        return attributions.reshape(B, C, T)


# ---------------- Factory ----------------
class ExplainerFactory:
    _REGISTRY = {
        "feature_ablation": FeatureAblationExplainer,
        "gradient_x_input": GradientXInputExplainer,
        "deepliftshap":     DeepLiftShapExplainer,
        "lime_tabular":     LimeTabularTimeseriesExplainer,
    }

    @classmethod
    def get(cls, method_name: str, model):
        key = (method_name or "").lower().strip()
        if key not in cls._REGISTRY:
            raise ValueError(f"Unknown XAI method: {method_name}")
        return cls._REGISTRY[key](model)
