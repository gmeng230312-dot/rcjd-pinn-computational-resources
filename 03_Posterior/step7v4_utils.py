# -*- coding: utf-8 -*-
"""
Utility functions for Step7 V4 posterior reliability updating.

Design principle:
- Step6 V2B provides MAP-like inversion benchmark cases.
- Step3 provides a 2000-sample high-fidelity prior response library.
- Step7 V4 constructs a nonparametric posterior by likelihood reweighting of the
  prior library against a selected Step6 observation curve.

Date: 2026-06-27
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import norm
except Exception:  # pragma: no cover
    norm = None


EPS = 1e-300


def now_stamp() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: str | Path) -> Dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def setup_output_dir(config: Dict, prefix: str = "outputs_step7_v4_nonparametric_posterior") -> Path:
    output_root = Path(config["project"]["output_root"])
    out = output_root / f"{prefix}_{now_stamp()}"
    for sub in [
        "logs", "step7_ready_inputs", "tables", "figures", "summary",
        "convergence", "exceedance", "diagnostics", "large_mc_optional"
    ]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    save_json(config, out / "summary" / "step7_v4_config_used.json")
    return out


def setup_logger(out_dir: Path, name: str = "Step7V4") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "logs" / f"{name.lower()}_{now_stamp()}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def timer() -> float:
    return time.perf_counter()


def elapsed(t0: float) -> float:
    return time.perf_counter() - t0


def as_path(p: str | Path) -> Path:
    return Path(str(p).replace("\\", "/"))


def rmse(a: np.ndarray, b: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis))


def mae(a: np.ndarray, b: np.ndarray, axis=None) -> np.ndarray:
    return np.mean(np.abs(np.asarray(a) - np.asarray(b)), axis=axis)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def effective_sample_size(weights: np.ndarray) -> float:
    w = normalize_weights(weights)
    return float(1.0 / np.sum(w ** 2))


def normalize_weights(log_or_weights: np.ndarray, is_log: bool = False) -> np.ndarray:
    x = np.asarray(log_or_weights, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    if is_log:
        m = np.max(x)
        w = np.exp(x - m)
    else:
        w = np.maximum(x, 0.0)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def likelihood_weights_from_rmse(
    d: np.ndarray,
    sigma: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64).ravel()
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    logw = -0.5 * (d / sigma) ** 2
    if mask is not None:
        logw = np.where(mask, logw, -np.inf)
    return normalize_weights(logw, is_log=True)


def find_sigma_for_target_ess(
    d: np.ndarray,
    target_ess: float,
    n_grid: int = 200,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Find sigma_y so ESS is close to target_ess by log-grid search.

    ESS increases monotonically with sigma in normal cases. A robust log-grid
    search is used to avoid numerical issues when the best RMSE is almost zero.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    finite = np.isfinite(d)
    if mask is not None:
        finite &= mask.astype(bool)
    if not np.any(finite):
        raise ValueError("No finite candidate misfit values are available.")

    d_pos = d[finite]
    d_scale = np.median(d_pos[d_pos > 0]) if np.any(d_pos > 0) else np.max(d_pos) + 1e-12
    d_min = max(np.min(d_pos[d_pos > 0]) if np.any(d_pos > 0) else d_scale * 1e-6, 1e-12)
    d_max = max(np.max(d_pos), d_scale, d_min * 10)

    # Wide search range. Units follow d, typically meters.
    sigmas = np.logspace(np.log10(d_min * 0.05), np.log10(d_max * 20.0), n_grid)
    best = None
    for s in sigmas:
        w = likelihood_weights_from_rmse(d, s, mask=mask)
        ess = effective_sample_size(w)
        score = abs(ess - target_ess)
        if best is None or score < best[0]:
            best = (score, s, ess, w)
    assert best is not None
    return float(best[1]), float(best[2]), best[3]


def weighted_mean(x: np.ndarray, weights: Optional[np.ndarray] = None, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if weights is None:
        return np.mean(x, axis=axis)
    w = normalize_weights(weights)
    return np.sum(x * np.expand_dims(w, tuple(range(1, x.ndim))), axis=axis)


def weighted_var(x: np.ndarray, weights: Optional[np.ndarray] = None, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if weights is None:
        return np.var(x, axis=axis, ddof=1 if x.shape[axis] > 1 else 0)
    mu = weighted_mean(x, weights, axis=axis)
    w = normalize_weights(weights)
    return np.sum(((x - mu) ** 2) * np.expand_dims(w, tuple(range(1, x.ndim))), axis=axis)


def weighted_std(x: np.ndarray, weights: Optional[np.ndarray] = None, axis: int = 0) -> np.ndarray:
    return np.sqrt(np.maximum(weighted_var(x, weights, axis=axis), 0.0))


def weighted_quantile(values: np.ndarray, quantiles: Sequence[float], sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted quantiles for 1D values.

    Returns quantiles in the same units as values. For unweighted data, uses
    numpy.quantile.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    q = np.asarray(quantiles, dtype=np.float64)
    if len(v) == 0:
        return np.full_like(q, np.nan, dtype=np.float64)
    if sample_weight is None:
        return np.quantile(v, q)
    w = normalize_weights(np.asarray(sample_weight, dtype=np.float64).ravel())
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    cw[-1] = 1.0
    return np.interp(q, cw, v_sorted)


def weighted_quantile_axis0(x: np.ndarray, quantiles: Sequence[float], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted quantiles along axis 0.

    x: [N, M]. Returns [len(quantiles), M].
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("weighted_quantile_axis0 expects a 2D array [N, M].")
    out = np.zeros((len(quantiles), x.shape[1]), dtype=np.float64)
    for j in range(x.shape[1]):
        out[:, j] = weighted_quantile(x[:, j], quantiles, weights)
    return out


def reliability_index_from_pf(pf: float, n_eff: Optional[float] = None) -> Tuple[float, float, bool]:
    """Return beta, clipped_pf, clipped_flag.

    If pf is 0 or 1, it is clipped using an effective sample-size bound so that
    beta remains finite. The raw Pf should still be reported separately.
    """
    if norm is None:
        return float("nan"), float(pf), False
    pf = float(pf)
    clipped = False
    if n_eff is None or n_eff <= 0:
        lo, hi = 1e-12, 1 - 1e-12
    else:
        lo, hi = 0.5 / n_eff, 1.0 - 0.5 / n_eff
    pf_c = min(max(pf, lo), hi)
    clipped = abs(pf_c - pf) > 0
    beta = -float(norm.ppf(pf_c))
    return beta, pf_c, clipped


def write_csv(path: str | Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def response_features(curves: np.ndarray, x: np.ndarray, response_type: str, affected_fraction: float = 0.5) -> Dict[str, np.ndarray]:
    """Extract peak, RMS, affected length/count, and peak location.

    curves: [N, M]
    x: [M]
    response_type: 'w', 'opening', or 'faulting'
    """
    y = np.asarray(curves, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    if y.ndim != 2:
        raise ValueError(f"curves for {response_type} must be [N, M]")
    abs_y = np.abs(y)
    peak_idx = np.argmax(abs_y, axis=1)
    peak = abs_y[np.arange(y.shape[0]), peak_idx]
    rms = np.sqrt(np.mean(y ** 2, axis=1))
    peak_location = x[peak_idx]
    frac = float(affected_fraction)
    threshold_each = frac * peak[:, None]
    active = abs_y > threshold_each
    if response_type == "w":
        dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
        affected = np.sum(active, axis=1) * dx
        affected_name = "affected_length_m"
    else:
        affected = np.sum(active, axis=1).astype(float)
        affected_name = "active_joint_count"
    return {
        "peak_abs_m": peak,
        "rms_m": rms,
        affected_name: affected,
        "peak_location_m": peak_location,
    }


def summarize_feature(name: str, values: np.ndarray, weights: Optional[np.ndarray], q_list: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    q = weighted_quantile(values, q_list, weights)
    out = {
        f"{name}_mean": float(weighted_mean(values[:, None], weights)[0]),
        f"{name}_std": float(weighted_std(values[:, None], weights)[0]),
        f"{name}_min": float(np.min(values)),
        f"{name}_max": float(np.max(values)),
    }
    for qq, val in zip(q_list, q):
        out[f"{name}_q{int(round(qq * 100)):02d}"] = float(val)
    return out


def system_info() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }


def safe_np_load(path: str | Path) -> np.lib.npyio.NpzFile:
    return np.load(Path(path), allow_pickle=True)


def npz_scalar_to_int(x) -> int:
    arr = np.asarray(x)
    return int(arr.item()) if arr.shape == () else int(arr.ravel()[0])


def describe_array(a: np.ndarray) -> str:
    a = np.asarray(a)
    if a.size == 0:
        return f"shape={a.shape}, dtype={a.dtype}, empty"
    if np.issubdtype(a.dtype, np.number):
        return f"shape={a.shape}, dtype={a.dtype}, min={np.nanmin(a):.6g}, mean={np.nanmean(a):.6g}, max={np.nanmax(a):.6g}"
    return f"shape={a.shape}, dtype={a.dtype}"
