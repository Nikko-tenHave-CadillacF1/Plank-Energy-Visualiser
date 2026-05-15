"""Surface fitting for plank energy: plane or quadratic in (h, K)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Model = Literal["plane", "quadratic"]


@dataclass
class SurfaceFit:
    model: Model
    coeffs: np.ndarray   # raw least-squares coefficients (length 3 or 6)
    rmse: float
    r2: float

    def predict(self, h: np.ndarray, k: np.ndarray) -> np.ndarray:
        return _design(h, k, self.model) @ self.coeffs

    def equation_text(self) -> str:
        c = self.coeffs
        if self.model == "plane":
            a, b, c0 = c
            return f"E = {a:.2f}\u00b7h + {b:.3f}\u00b7K + {c0:.1f}"
        a_h, a_k, a_hh, a_kk, a_hk, c0 = c
        return (f"E = {a_h:.2f}\u00b7h + {a_k:.3f}\u00b7K + "
                f"{a_hh:.3f}\u00b7h\u00b2 + {a_kk:.2e}\u00b7K\u00b2 + "
                f"{a_hk:.2e}\u00b7h\u00b7K + {c0:.1f}")


def _design(h: np.ndarray, k: np.ndarray, model: Model) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    ones = np.ones_like(h)
    if model == "plane":
        return np.column_stack([h, k, ones])
    if model == "quadratic":
        return np.column_stack([h, k, h * h, k * k, h * k, ones])
    raise ValueError(f"Unknown model: {model!r}")


def fit_surface(h: np.ndarray, k: np.ndarray, e: np.ndarray, model: Model = "plane") -> SurfaceFit:
    A = _design(h, k, model)
    coeffs, _, _, _ = np.linalg.lstsq(A, e, rcond=None)
    pred = A @ coeffs
    resid = e - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((e - np.mean(e)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return SurfaceFit(model=model, coeffs=coeffs, rmse=rmse, r2=r2)
