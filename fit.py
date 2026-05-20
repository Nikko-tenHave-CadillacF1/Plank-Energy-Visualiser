"""Surface fitting: plane or quadratic in (h, K) -> E."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Model = Literal["plane", "quadratic"]


@dataclass
class SurfaceFit:
    model: Model
    coeffs: np.ndarray
    rmse: float
    r2: float

    def predict(self, h: np.ndarray, k: np.ndarray) -> np.ndarray:
        return _design(h, k, self.model) @ self.coeffs

    def equation_text(self) -> str:
        c = self.coeffs
        if self.model == "plane":
            return f"E = {c[0]:.2f}\u00b7h + {c[1]:.3f}\u00b7K + {c[2]:.1f}"
        return (f"E = {c[0]:.2f}\u00b7h + {c[1]:.3f}\u00b7K + "
                f"{c[2]:.3f}\u00b7h\u00b2 + {c[3]:.2e}\u00b7K\u00b2 + "
                f"{c[4]:.2e}\u00b7h\u00b7K + {c[5]:.1f}")


def _design(h: np.ndarray, k: np.ndarray, model: Model) -> np.ndarray:
    h, k = np.asarray(h, float), np.asarray(k, float)
    if model == "plane":
        return np.column_stack([h, k, np.ones_like(h)])
    if model == "quadratic":
        return np.column_stack([h, k, h*h, k*k, h*k, np.ones_like(h)])
    raise ValueError(f"Unknown model: {model!r}")


def fit_surface(h: np.ndarray, k: np.ndarray, e: np.ndarray, model: Model = "plane") -> SurfaceFit:
    A = _design(h, k, model)
    coeffs, *_ = np.linalg.lstsq(A, e, rcond=None)
    resid = e - A @ coeffs
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((e - np.mean(e))**2))
    return SurfaceFit(
        model=model, coeffs=coeffs,
        rmse=float(np.sqrt(ss_res / len(e))),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    )
