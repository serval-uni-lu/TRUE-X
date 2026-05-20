# xai_methods.py
import numpy as np
import torch
import warnings
from typing import Optional, List, Union

from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    Saliency,
    GuidedBackprop,
    FeatureAblation,
    Occlusion,
    InputXGradient,          # Gradient x Input
    ShapleyValueSampling,    # for shapley_sampling
)

Tensor = torch.Tensor
NDArray = np.ndarray

# --- Optional extras (LIME / SHAP) ---
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer as _LimeTabularExplainer
    _HAVE_LIME = True
except Exception:
    _HAVE_LIME = False
    _LimeTabularExplainer = None  # type: ignore

try:
    import shap as _shap
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False
    _shap = None  # type: ignore


# --- optional wrapper for VAE_RUL (if present) ---
try:
    from ml_models.vae_rul_wrapper import VAE_RUL_Wrapper  # type: ignore
    _HAVE_VAE_WRAPPER = True
except Exception:
    VAE_RUL_Wrapper = None  # type: ignore
    _HAVE_VAE_WRAPPER = False


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
        # never pass requires_grad in as_tensor
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
        """
        For classification, choose argmax per-sample if target_class is None.
        For regression, return an index tensor if the model outputs (B, K>=1), else None.
        """
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
            # regression — single output: captum expects target=None
            if out.dim() == 1 or (out.dim() == 2 and out.size(1) == 1):
                return None
            # multi-output regression: default to output index 0
            return torch.zeros(inputs.size(0), dtype=torch.long, device=inputs.device)

    @staticmethod
    def _expand_baseline_to_batch(inputs: Tensor, base: Tensor) -> Tensor:
        """
        Ensure baseline shape is (B, C, T).
        Accepts scalar, (C,T), (1,C,T), or (B,C,T) and returns (B,C,T).
        """
        base = base.to(device=inputs.device, dtype=inputs.dtype)

        # Scalar -> expand to full inputs shape
        if base.dim() == 0 or (base.numel() == 1 and base.dim() <= 1):
            return base.view(1).expand_as(inputs)

        # (C,T) -> (1,C,T) -> (B,C,T)
        if base.dim() == inputs.dim() - 1 and base.shape == inputs.shape[1:]:
            return base.unsqueeze(0).expand(inputs.size(0), *base.shape)

        # (1,C,T) -> (B,C,T)
        if base.dim() == inputs.dim() and base.shape[0] == 1 and base.shape[1:] == inputs.shape[1:]:
            return base.expand_as(inputs)

        # (B,C,T) already okay
        if base.dim() == inputs.dim() and base.shape == inputs.shape:
            return base

        raise ValueError(
            f"Baseline shape {tuple(base.shape)} is incompatible with input {tuple(inputs.shape)}"
        )

    # Optional hook for SHAP additivity spot-checks (override in subclasses)
    def additivity_check(self):
        return None


# ---------------- Expected Gradients ----------------
class ExpectedGradientsExplainer(BaseExplainer):
    def explain(
        self,
        x_val,
        background_data,            # (M, C, T) or (C, T)
        baseline=None,              # unused by EG (kept for API compat)
        task='classification',
        target_class=None,          # int or None
        steps=50,
        device='auto',              # 'auto' => follow model device
        *,
        return_abs: bool = True,
        **_
    ):
        """
        Expected Gradients: average Integrated Gradients across multiple baselines.
        - x_val: (B, C, T) numpy array
        - background_data: (M, C, T) (recommended) or (C, T)
        """
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)  # (B,C,T)
        if background_data is None:
            raise ValueError("ExpectedGradients requires background_data with multiple baselines.")
        baselines = torch.as_tensor(background_data, dtype=torch.float32, device=device)
        if baselines.dim() == 2:  # (C,T) -> (M=1,C,T)
            baselines = baselines.unsqueeze(0)
        M = baselines.size(0)
        B = inputs.size(0)

        models = self.model if self.is_ensemble else [self.model]
        model_attrs = []

        for m in models:
            if getattr(m, "__class__", None) and m.__class__.__name__ == "VAE_RUL" and _HAVE_VAE_WRAPPER:
                m = VAE_RUL_Wrapper(m)  # type: ignore
            m.to(device).eval()

            targets = self._select_targets(m, inputs, task, target_class)
            ig = IntegratedGradients(m)
            attrs_this_model = torch.zeros_like(inputs)

            for i in range(B):
                x_i_rep = inputs[i:i+1].expand(M, -1, -1)  # (M,C,T)
                tgt = None
                if isinstance(targets, torch.Tensor):
                    tgt = int(targets.item()) if targets.numel() == 1 else int(targets[i].item())
                a_i = ig.attribute(
                    inputs=x_i_rep,
                    baselines=baselines,
                    target=tgt,
                    n_steps=int(steps),
                ).mean(dim=0, keepdim=True)  # (1,C,T)
                attrs_this_model[i:i+1] = a_i

            model_attrs.append(self._to_numpy(attrs_this_model))

        attr = np.mean(model_attrs, axis=0)
        return np.abs(attr) if return_abs else attr


# ---------------- Captum explainers ----------------
class IntegratedGradientsExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        background_data: Optional[NDArray] = None,
        baseline: Optional[NDArray] = None,
        task: str = "classification",
        target_class: Optional[int] = None,
        steps: int = 50,
        device: str = "auto",
        *,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)

        # choose baseline: explicit > background mean > zeros
        if baseline is not None:
            base = torch.as_tensor(baseline, dtype=torch.float32, device=device)
        elif background_data is not None:
            bg = torch.as_tensor(background_data, dtype=torch.float32, device=device)
            base = bg.mean(dim=0) if bg.dim() == 3 else bg  # (C,T) or (1,C,T)/(B,C,T)
        else:
            base = torch.zeros_like(inputs[0])              # (C,T)

        base = self._expand_baseline_to_batch(inputs, base)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            targets = self._select_targets(m, inputs, task, target_class)
            ig = IntegratedGradients(m)
            a = ig.attribute(inputs, baselines=base, target=targets, n_steps=int(steps))
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


class DeepLiftExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        background_data: Optional[NDArray] = None,
        baseline: Optional[NDArray] = None,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)

        if baseline is not None:
            base = torch.as_tensor(baseline, dtype=torch.float32, device=device)
        elif background_data is not None:
            bg = torch.as_tensor(background_data, dtype=torch.float32, device=device)
            base = bg.mean(dim=0) if bg.dim() == 3 else bg
        else:
            base = torch.zeros_like(inputs[0])
        base = self._expand_baseline_to_batch(inputs, base)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            targets = self._select_targets(m, inputs, task, target_class)
            dl = DeepLift(m)
            a = dl.attribute(inputs, baselines=base, target=targets)
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


class DeepLiftShapExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        background_data: NDArray,   # required
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
            baselines = baselines.unsqueeze(0)  # (M,C,T)

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


class SaliencyExplainer(BaseExplainer):
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
        raw = self._prepare_inputs(x_val, device)
        # lock targets off clean forward (stability)
        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            with torch.no_grad():
                targets = self._select_targets(m, raw, task, target_class)
            x = raw.clone().detach().requires_grad_(True)
            sal = Saliency(m)
            a = sal.attribute(x, target=targets)
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


class GuidedBackPropExplainer(BaseExplainer):
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
        raw = self._prepare_inputs(x_val, device)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            with torch.no_grad():
                targets = self._select_targets(m, raw, task, target_class)
            x = raw.clone().detach().requires_grad_(True)
            gbp = GuidedBackprop(m)
            a = gbp.attribute(x, target=targets)
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


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


class GradientShapExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        background_data: NDArray,     # required
        task: str = "classification",
        target_class: Optional[int] = None,
        smooth_samples: int = 20,
        noise_std: float = 0.1,
        device: str = "auto",
        *,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        if background_data is None:
            raise ValueError("GradientShap requires background_data.")
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)
        baselines = torch.as_tensor(background_data, dtype=torch.float32, device=device)
        if baselines.dim() == 2:
            baselines = baselines.unsqueeze(0)  # (M,C,T)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            gs = GradientShap(m)
            targets = self._select_targets(m, inputs, task, target_class)
            a = gs.attribute(
                inputs,
                baselines=baselines,
                target=targets,
                n_samples=int(smooth_samples),
                stdevs=float(noise_std),
            )
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


class OcclusionExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        mode: str = "time",
        time_window: int = 10,
        time_stride: Optional[int] = None,
        perturbations_per_eval: int = 32,
        baseline: Optional[NDArray] = None,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        """
        Occlusion over time or channels:
          - mode='time': slide window of length time_window along T
          - mode='channel': occlude one full channel across all time at once
        Uses on-manifold fill via 'baseline' where provided (expanded to batch).
        """
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)
        # baseline fill (optional -> zeros)
        if baseline is None:
            base = torch.zeros_like(inputs[0])
        else:
            base = torch.as_tensor(baseline, dtype=torch.float32, device=device)
        base = self._expand_baseline_to_batch(inputs, base)

        time_stride = int(time_stride or time_window)
        C, T = inputs.shape[1], inputs.shape[2]
        m_lower = (mode or "time").lower()

        if m_lower == "time":
            sliding_window_shapes = (1, int(time_window))  # (channel window=1, time window)
            strides = (1, time_stride)
        elif m_lower == "channel":
            sliding_window_shapes = (1, T)   # one full-time channel per occlusion
            strides = (1, T)                  # move channel-by-channel
        else:
            warnings.warn("OcclusionExplainer: unknown mode; defaulting to 'time'.")
            sliding_window_shapes = (1, int(time_window))
            strides = (1, time_stride)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            occ = Occlusion(m)
            targets = self._select_targets(m, inputs, task, target_class)
            a = occ.attribute(
                inputs,
                target=targets,
                sliding_window_shapes=sliding_window_shapes,  # (C_win, T_win) with C_win=1
                strides=strides,
                perturbations_per_eval=int(perturbations_per_eval),
                baselines=base,
            )
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


class FeatureAblationExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        baseline: Optional[NDArray] = None,
        group_mode: str = "none",        # "none" | "channel" | "time_window"
        time_window: int = 10,           # used when group_mode="time_window"
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        """
        Feature ablation with optional grouping to speed up:
          - group_mode="none": per-(C,T) cell ablation (slowest, most granular)
          - group_mode="channel": ablate one full channel across all time
          - group_mode="time_window": ablate time windows shared across channels
        Uses 'baseline' as fill for on-manifold perturbations.
        """
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)

        # baseline fill
        if baseline is None:
            base = torch.zeros_like(inputs[0])
        else:
            base = torch.as_tensor(baseline, dtype=torch.float32, device=device)
        base = self._expand_baseline_to_batch(inputs, base)

        # Grouped feature mask
        C, T = inputs.shape[1], inputs.shape[2]
        feature_mask = None
        gm = (group_mode or "none").lower()
        if gm == "channel":
            # one group per channel across all time
            feature_mask = torch.arange(C, device=device, dtype=torch.long).view(C, 1).expand(C, T)
        elif gm == "time_window":
            tw = max(1, int(time_window))
            time_bins = torch.arange(T, device=device, dtype=torch.long) // tw   # (T,)
            feature_mask = time_bins.view(1, T).expand(C, T)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            fabl = FeatureAblation(m)
            targets = self._select_targets(m, inputs, task, target_class)
            a = fabl.attribute(
                inputs,
                target=targets,
                baselines=base,
                feature_mask=feature_mask,
            )
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


class SmoothGradientExplainer(BaseExplainer):
    def explain(
        self,
        x_val: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        smooth_samples: int = 20,
        noise_std: float = 0.1,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        """
        SmoothGrad using Saliency with Gaussian noise on inputs.
        """
        device = self._resolve_device(self.model, device)
        x_clean = self._prepare_inputs(x_val, device)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            # lock targets on clean
            with torch.no_grad():
                targets = self._select_targets(m, x_clean, task, target_class)
            sal = Saliency(m)
            per_model = []
            for _ in range(int(smooth_samples)):
                noise = torch.randn_like(x_clean) * float(noise_std)
                xn = (x_clean + noise).clone().detach().requires_grad_(True)
                a = sal.attribute(xn, target=targets)
                per_model.append(a)
            a_mean = torch.stack(per_model, dim=0).mean(dim=0)
            attrs.append(self._to_numpy(a_mean))
        out = np.mean(attrs, axis=0)
        return np.abs(out) if return_abs else out


# ---------------- Extra explainers ----------------
class ShapleySamplingExplainer(BaseExplainer):
    def __init__(self, model):
        super().__init__(model)
        self._last = None  # for optional additivity check

    def explain(
        self,
        x_val: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        sample_size: int = 2048,
        baseline: Optional[NDArray] = None,
        return_abs: bool = True,
        **_,
    ) -> NDArray:
        """
        Captum's Shapley Value Sampling. Uses a zero baseline by default.
        """
        device = self._resolve_device(self.model, device)
        inputs = self._prepare_inputs(x_val, device)

        if baseline is None:
            base = torch.zeros_like(inputs[0])  # (C,T)
        else:
            base = torch.as_tensor(baseline, dtype=torch.float32, device=device)
        base = self._expand_baseline_to_batch(inputs, base)

        models = self.model if self.is_ensemble else [self.model]
        attrs = []
        for m in models:
            m.to(device).eval()
            targets = self._select_targets(m, inputs, task, target_class)
            svs = ShapleyValueSampling(m)
            a = svs.attribute(inputs, target=targets, baselines=base, n_samples=int(sample_size))
            attrs.append(self._to_numpy(a))
        out = np.mean(attrs, axis=0)
        self._last = (inputs.detach().cpu().numpy(), base.detach().cpu().numpy(), out.copy())
        return np.abs(out) if return_abs else out

    def additivity_check(self):
        if self._last is None:
            return None
        x, base, attr = self._last
        try:
            m = self.model[0] if self.is_ensemble else self.model
            m.eval()
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32, device=self._resolve_device(self.model, "auto"))
                bt = torch.tensor(base, dtype=torch.float32, device=self._resolve_device(self.model, "auto"))
                fx = m(xt)
                fbg = m(bt).mean()
                if isinstance(fx, (tuple, list)):
                    fx = fx[0]
                if fx.dim() == 2 and fx.size(1) > 1:
                    fx = fx.max(dim=1).values.mean()
                else:
                    fx = fx.mean()
                phi_sum = attr.sum(axis=(1, 2)).mean()
                return float(fx.item()), float(fbg.item()), float(phi_sum)
        except Exception:
            return None


class LimeTabularTimeseriesExplainer(BaseExplainer):
    """
    LIME Tabular wrapper for (B, C, T) tensors.
    Flattens to (B, C*T) and reshapes the explanation back to (C, T).
    """
    def explain(
        self,
        x_val: NDArray,
        background_data: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        num_features: Optional[int] = None,   # default: all features (C*T)
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
        # Work on CPU for LIME (uses numpy)
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

        # Prediction function wrapping the PyTorch model
        device_model = self._resolve_device(self.model, device)
        models = self.model if self.is_ensemble else [self.model]

        def _predict_fn(Xtab: NDArray) -> NDArray:
            # Xtab: (N, D)
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
                # out: (N, K) or (N, 1)
                if mode == "classification":
                    # softmax over last dim if logits
                    if out.size(-1) > 1:
                        prob = torch.softmax(out, dim=1)  # (N, K)
                    else:
                        p1 = torch.sigmoid(out)          # (N, 1)
                        prob = torch.cat([1 - p1, p1], dim=1)  # (N, 2)
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
                # choose class to explain
                if target_class is None:
                    probs = _predict_fn(xi.reshape(1, -1))
                    cls = int(probs.argmax(axis=1).item())
                else:
                    cls = int(target_class)
                exp = explainer.explain_instance(
                    xi, _predict_fn, labels=(cls,), num_features=num_features, num_samples=int(num_samples)
                )
                # local_exp is dict: label -> list[(feature_idx, weight)]
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


class ShapExplainer(BaseExplainer):
    """
    SHAP wrapper. Tries GradientExplainer for torch; falls back to KernelExplainer on flattened features.
    Returns (B,C,T) attributions.
    """
    def __init__(self, model):
        super().__init__(model)
        self._last = None  # for optional additivity check

    def explain(
        self,
        x_val: NDArray,
        background_data: NDArray,
        task: str = "classification",
        target_class: Optional[int] = None,
        device: str = "auto",
        *,
        return_abs: bool = True,
        background_k: int = 20,      # cap background for speed
        max_samples_kernel: int = 10, # cap query samples for KernelExplainer
        max_explain: Optional[int] = None,  # optional tighter cap (tiles back)
        **_,
    ) -> NDArray:
        if not _HAVE_SHAP:
            warnings.warn("shap is not installed; returning zeros.")
            return np.zeros_like(x_val, dtype=np.float32)

        device_t = self._resolve_device(self.model, device)
        x = self._prepare_inputs(x_val, device_t)  # (B,C,T)
        bg = torch.as_tensor(background_data, dtype=torch.float32, device=device_t)
        if bg.ndim == 2:
            bg = bg.unsqueeze(0)

        # Subsample background to at most background_k to keep Kernel SHAP tractable
        if bg.size(0) > int(background_k):
            perm = torch.randperm(bg.size(0), device=bg.device)[: int(background_k)]
            bg = bg[perm]

        models = self.model if self.is_ensemble else [self.model]
        m = models[0]  # SHAP doesn't natively average models; use first
        m.to(device_t).eval()

        B, C, T = x.shape
        # Try GradientExplainer (fast)
        try:
            explainer = _shap.GradientExplainer(m, bg)
            # For classification: shap returns list [array per output]
            with torch.no_grad():
                out = m(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            if (task or "classification").lower() == "classification" and out.dim() == 2 and out.size(1) > 1:
                # choose target per sample
                if target_class is None:
                    target_idx = out.argmax(dim=1).detach().cpu().numpy()
                else:
                    target_idx = np.full((B,), int(target_class), dtype=np.int64)
                shap_vals = []
                for i in range(B):
                    sv = explainer.shap_values(x[i:i+1], ranked_outputs=None)
                    # sv is list of arrays [K outputs]; pick target_idx[i]
                    sv_i = sv[target_idx[i]]  # (1,C,T)
                    shap_vals.append(sv_i)
                a = np.concatenate(shap_vals, axis=0)
            else:
                # regression or single output => returns one array
                sv = explainer.shap_values(x)
                a = sv if isinstance(sv, np.ndarray) else sv[0]
            self._last = (x.detach().cpu().numpy(), bg.detach().cpu().numpy(), a.copy())
            return np.abs(a) if return_abs else a
        except Exception as _eg_err:
            # Fall back to KernelExplainer on flattened input (CPU/numpy)
            warnings.warn(f"GradientExplainer failed ({_eg_err}); falling back to SHAP KernelExplainer (slow).")

            x_np = x.detach().cpu().numpy().astype(np.float32)
            bg_np = bg.detach().cpu().numpy().astype(np.float32)
            B, C, T = x_np.shape
            D = C * T
            X = x_np.reshape(B, D)
            BG = bg_np.reshape(bg_np.shape[0], D)

            mode = (task or "classification").lower()
            models = self.model if self.is_ensemble else [self.model]

            def _predict(Xtab: NDArray) -> NDArray:
                Xb = torch.from_numpy(Xtab.astype(np.float32)).to(device_t).view(-1, C, T)
                with torch.no_grad():
                    outs = []
                    for mm in models:
                        mm.to(device_t).eval()
                        out = mm(Xb)
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                        outs.append(out)
                    out = torch.stack([o if o.dim() > 1 else o.unsqueeze(1) for o in outs], dim=0).mean(dim=0)
                    if mode == "classification":
                        if out.size(-1) > 1:
                            prob = torch.softmax(out, dim=1)
                        else:
                            p1 = torch.sigmoid(out)
                            prob = torch.cat([1 - p1, p1], dim=1)  # (N,2)
                        return prob.detach().cpu().numpy()
                    else:
                        return out.detach().cpu().numpy()

            try:
                expl = _shap.KernelExplainer(_predict, BG)
                # choose class to explain (classification)
                if mode == "classification":
                    if target_class is None:
                        probs = _predict(X[:1])
                        cls = int(probs.argmax(axis=1).item())
                    else:
                        cls = int(target_class)
                # cap #query samples
                limit = int(max_explain) if max_explain is not None else int(max_samples_kernel)
                Xq = X if X.shape[0] <= limit else X[:limit]
                sv = expl.shap_values(Xq)
                if mode == "classification":
                    if isinstance(sv, list):
                        a_flat = sv[cls]  # (N,D)
                    else:
                        # binary often returns single ndarray (positive class)
                        a_flat = sv      # (N,D)
                else:
                    a_flat = sv if isinstance(sv, np.ndarray) else sv[0]
                # tile back if truncated
                if Xq.shape[0] != X.shape[0]:
                    reps = int(np.ceil(X.shape[0] / Xq.shape[0]))
                    a_flat = np.vstack([a_flat] * reps)[: X.shape[0]]
            except Exception:
                # Catastrophic fallback
                a_flat = np.zeros_like(X, dtype=np.float32)

            a = a_flat.reshape(B, C, T).astype(np.float32)
            self._last = (x_np, bg_np, a.copy())
            return np.abs(a) if return_abs else a

    def additivity_check(self):
        if self._last is None:
            return None
        x, bg, a = self._last
        try:
            m = self.model[0] if self.is_ensemble else self.model
            dev = self._resolve_device(self.model, "auto")
            m.to(dev).eval()
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32, device=dev)
                bt = torch.tensor(bg, dtype=torch.float32, device=dev)
                fx = m(xt)
                if isinstance(fx, (tuple, list)):
                    fx = fx[0]
                if fx.dim() == 2 and fx.size(1) > 1:
                    fx = fx.max(dim=1).values.mean()
                else:
                    fx = fx.mean()
                fbg = m(bt)
                if isinstance(fbg, (tuple, list)):
                    fbg = fbg[0]
                if fbg.dim() == 2 and fbg.size(1) > 1:
                    fbg = fbg.max(dim=1).values.mean()
                else:
                    fbg = fbg.mean()
                phi_sum = a.sum(axis=(1, 2)).mean()
                return float(fx.item()), float(fbg.item()), float(phi_sum)
        except Exception:
            return None


# ---------------- Factory ----------------
class ExplainerFactory:
    _REGISTRY = {
        "expected_gradients": ExpectedGradientsExplainer,
        "integrated_gradients": IntegratedGradientsExplainer,
        "deeplift": DeepLiftExplainer,
        "deepliftshap": DeepLiftShapExplainer,
        "saliency": SaliencyExplainer,
        "guided_back_prop": GuidedBackPropExplainer,
        "gradient_x_input": GradientXInputExplainer,
        "gradientshap": GradientShapExplainer,
        "occlusion": OcclusionExplainer,
        "feature_ablation": FeatureAblationExplainer,
        "smooth_gradient": SmoothGradientExplainer,
        "lime_tabular": LimeTabularTimeseriesExplainer,
        "shapley_sampling": ShapleySamplingExplainer,
        "shap": ShapExplainer,
    }

    @classmethod
    def get(cls, method_name: str, model):
        key = (method_name or "").lower().strip()
        if key not in cls._REGISTRY:
            raise ValueError(f"Unknown XAI method: {method_name}")
        return cls._REGISTRY[key](model)
