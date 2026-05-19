import numpy as np

__all__ = ["pearson_1d", "reg_distance_to_target", "clf_distance_to_target"]


def pearson_1d(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    a = a - a.mean()
    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def reg_distance_to_target(
    y: np.ndarray,
    *,
    target_value: float | None = None,
    target_range: tuple[float, float] | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if target_range is not None:
        lo, hi = float(target_range[0]), float(target_range[1])
        return np.asarray(np.maximum(0.0, np.abs(y - np.clip(y, lo, hi))), dtype=np.float64)
    if target_value is not None:
        return np.asarray(np.abs(y - float(target_value)), dtype=np.float64)
    return np.asarray(np.zeros_like(y), dtype=np.float64)


def clf_distance_to_target(
    scores: np.ndarray,
    *,
    target_class: int,
    score_margin: float = 0.0,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    _, _ = scores.shape

    margins = scores[:, target_class] - np.max(np.delete(scores, target_class, axis=1), axis=1)
    shortfall = np.maximum(0.0, score_margin - margins)
    return np.asarray(shortfall, dtype=np.float64)
