# -*- coding: utf-8 -*-
"""Common utilities for Ch4 Step6 curve-constrained inversion."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


# =============================================================================
# Basic IO, logging and reproducibility
# =============================================================================
def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(device_mode: str):
    if torch is None:
        raise RuntimeError(f"PyTorch could not be imported: {_TORCH_IMPORT_ERROR}")
    mode = str(device_mode).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(log_path: Path | str, name: str = "Step6") -> logging.Logger:
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_json(obj, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path | str, encoding: str = "utf-8-sig") -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding=encoding)


def snapshot_source_code(out_dir: Path, files: Optional[Iterable[str]] = None) -> None:
    src_dir = ensure_dir(out_dir / "source_code_snapshot")
    if files is None:
        files = [
            "step6v2_config.py",
            "step6v2_common.py",
            "step6v2a_prior_mode_diagnostics.py",
            "step6v2b_expanded_synthetic_benchmark.py",
            "step6v2c_sparse_noise_benchmark.py",
            "step6v2d_mode_count_selection.py",
            "step6v2e_prior_regularization_ablation.py",
            "step6v2f_oracle_projection_check.py",
            "step6v2g_ltsbsm_physical_backcheck.py",
            "step6v2h_final_summary_and_figures.py",
        ]
    here = Path(__file__).resolve().parent
    for fn in files:
        p = here / fn
        if p.exists():
            shutil.copy2(p, src_dir / fn)


def setup_plot_style(cfg=None):
    font_family = getattr(cfg, "FIG_FONT_FAMILY", "Times New Roman") if cfg is not None else "Times New Roman"
    plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7


def cm_to_inch(x: float) -> float:
    return float(x) / 2.54


def save_figure(fig, out_dir: Path, base_name: str, cfg, logger: Optional[logging.Logger] = None) -> None:
    if not bool(getattr(cfg, "SAVE_FIGURES", True)):
        return
    ensure_dir(out_dir)
    for ext in list(getattr(cfg, "SAVE_FIG_FORMATS", ["png", "svg"])):
        path = out_dir / f"{base_name}.{ext}"
        try:
            if ext.lower() == "png":
                fig.savefig(path, dpi=int(getattr(cfg, "FIG_DPI", 600)), bbox_inches="tight")
            else:
                fig.savefig(path, bbox_inches="tight")
            if logger is not None:
                logger.info(f"Saved figure: {path}")
        except Exception as exc:
            if logger is not None:
                logger.warning(f"Failed to save figure {path}: {exc}")


# =============================================================================
# Path locators
# =============================================================================
def locate_step2b_database(cfg) -> Path:
    if bool(getattr(cfg, "USE_LATEST_STEP2B", True)):
        root = Path(cfg.STEP2_DIR)
        cands = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("Step2B_RandomCoefficient_AsymmetricThrust_")]
        cands = [p for p in cands if (p / cfg.STEP2B_DATABASE_NAME).exists()]
        if not cands:
            raise FileNotFoundError(f"No Step2B database folder found under {root}")
        latest = max(cands, key=lambda p: p.stat().st_mtime)
        return latest / cfg.STEP2B_DATABASE_NAME
    return Path(cfg.MANUAL_STEP2B_DIR) / cfg.STEP2B_DATABASE_NAME


def locate_step4d_checkpoint(cfg) -> Path:
    if bool(getattr(cfg, "USE_LATEST_STEP4D_TRAIN_DIR", True)):
        root = Path(cfg.STEP4D_OUTPUT_ROOT)
        cands = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("Step4D_RCJD_PINN_Forward_Stage1Train_")]
        cands = [p for p in cands if (p / cfg.STEP4D_CHECKPOINT_NAME).exists()]
        if not cands:
            # fallback to best_model if naming changed
            cands = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("Step4D_RCJD_PINN_Forward_Stage1Train_") and (p / "best_model.pt").exists()]
            if cands:
                latest = max(cands, key=lambda p: p.stat().st_mtime)
                return latest / "best_model.pt"
            raise FileNotFoundError(f"No Step4D checkpoint found under {root}")
        latest = max(cands, key=lambda p: p.stat().st_mtime)
        return latest / cfg.STEP4D_CHECKPOINT_NAME
    p = Path(cfg.MANUAL_STEP4D_TRAIN_DIR) / cfg.STEP4D_CHECKPOINT_NAME
    if not p.exists() and (Path(cfg.MANUAL_STEP4D_TRAIN_DIR) / "best_model.pt").exists():
        p = Path(cfg.MANUAL_STEP4D_TRAIN_DIR) / "best_model.pt"
    return p


# =============================================================================
# NPZ and Step3 normalization utilities
# =============================================================================
def load_npz(path: Path | str) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {k: np.asarray(data[k]).copy() for k in data.files}


def safe_str_array(arr) -> List[str]:
    out = []
    for v in np.asarray(arr).reshape(-1):
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def load_step3_normalization_stats(cfg) -> Dict:
    path = Path(cfg.STEP3_NORMALIZATION_JSON)
    if not path.exists():
        raise FileNotFoundError(f"Step3 normalization_stats.json not found: {path}")
    return load_json(path)


def normalize_node_features_raw(features_raw: np.ndarray, feature_names: List[str], stats: Dict) -> np.ndarray:
    features_raw = np.asarray(features_raw, dtype=np.float64)
    out = np.empty_like(features_raw, dtype=np.float64)
    node_stats = stats["node_features"]
    for j, name in enumerate(feature_names):
        st = node_stats[name]
        mean = float(st["mean"])
        std = float(st["std"])
        if abs(std) < 1.0e-30:
            std = 1.0
        out[:, :, j] = (features_raw[:, :, j] - mean) / std
    return out


def load_step3_split(npz_path: Path | str) -> Dict[str, np.ndarray]:
    d = load_npz(npz_path)
    if "feature_names" in d:
        d["feature_names_list"] = safe_str_array(d["feature_names"])
    return d


# =============================================================================
# Latent prior PCA utilities
# =============================================================================
def transform_channel(values: np.ndarray, transform: str, eps: float = 1.0e-12) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if transform == "log":
        # The input should be positive. Clip only for numerical robustness.
        return np.log(np.maximum(v, eps))
    if transform == "identity":
        return v.copy()
    raise ValueError(f"Unsupported transform: {transform}")


def inverse_transform_channel(values: np.ndarray, transform: str) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if transform == "log":
        return np.exp(v)
    if transform == "identity":
        return v.copy()
    raise ValueError(f"Unsupported transform: {transform}")


def build_prior_matrix_from_step2b(step2b_db: Dict[str, np.ndarray], cfg) -> Tuple[np.ndarray, Dict]:
    """Return transformed prior matrix with shape (n_samples, n_channels*n_coeff)."""
    channel_names = list(cfg.PRIOR_CHANNEL_NAMES)
    key_map = dict(cfg.STEP2B_CHANNEL_KEY_MAP)
    transforms = dict(cfg.CHANNEL_TRANSFORMS)
    x_coeff = np.asarray(step2b_db["x_coeff_grid_m"], dtype=np.float64).reshape(-1)
    n_coeff = len(x_coeff)
    arrays = []
    raw_stats = {}
    for name in channel_names:
        key = key_map[name]
        if key not in step2b_db:
            raise KeyError(f"Key {key} for channel {name} not found in Step2B database.")
        arr = np.asarray(step2b_db[key], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Channel {name} must be 2D, got shape {arr.shape}")
        if arr.shape[1] != n_coeff:
            if arr.shape[0] == n_coeff:
                arr = arr.T
            else:
                raise ValueError(f"Channel {name} spatial dimension does not match x_coeff length.")
        t = transforms[name]
        arrays.append(transform_channel(arr, t))
        raw_stats[name] = {
            "transform": t,
            "raw_min": float(np.nanmin(arr)),
            "raw_max": float(np.nanmax(arr)),
            "raw_mean": float(np.nanmean(arr)),
            "raw_std": float(np.nanstd(arr)),
            "p01": float(np.nanpercentile(arr, 1)),
            "p99": float(np.nanpercentile(arr, 99)),
        }
    n_samples = arrays[0].shape[0]
    mat = np.concatenate(arrays, axis=1)
    meta = {
        "channel_names": channel_names,
        "transforms": [transforms[name] for name in channel_names],
        "n_samples": int(n_samples),
        "n_coeff": int(n_coeff),
        "x_coeff_grid_m": x_coeff,
        "raw_stats": raw_stats,
    }
    return mat, meta


def compute_pca_prior(prior_matrix: np.ndarray, max_modes: int, eps: float = 1.0e-12) -> Dict[str, np.ndarray]:
    X = np.asarray(prior_matrix, dtype=np.float64)
    mean = np.nanmean(X, axis=0)
    scale = np.nanstd(X, axis=0)
    scale = np.where(scale < eps, 1.0, scale)
    Z = (X - mean) / scale
    # Thin SVD. For 2000 x 968 this is acceptable on a typical workstation.
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    eigenvalues = (S ** 2) / max(Z.shape[0] - 1, 1)
    explained_ratio = eigenvalues / np.sum(eigenvalues)
    max_modes = min(int(max_modes), Vt.shape[0])
    components = Vt[:max_modes, :]
    eig = eigenvalues[:max_modes]
    # Standard-normal latent scores for diagnostics.
    scores_all = (Z @ components.T) / np.sqrt(eig.reshape(1, -1) + eps)
    return {
        "data_mean": mean,
        "data_scale": scale,
        "components": components,
        "eigenvalues": eig,
        "explained_ratio_all": explained_ratio,
        "scores_all": scores_all,
    }


def reconstruct_prior_matrix_from_z(z: np.ndarray, prior: Dict[str, np.ndarray], n_modes: Optional[int] = None) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(1, -1)
    if n_modes is None:
        n_modes = z.shape[1]
    components = prior["components"][:n_modes, :]
    eig = prior["eigenvalues"][:n_modes]
    Zstd = (z * np.sqrt(eig.reshape(1, -1))) @ components
    Xtrans = prior["data_mean"].reshape(1, -1) + Zstd * prior["data_scale"].reshape(1, -1)
    return Xtrans


def split_prior_matrix_to_channels(xtrans: np.ndarray, meta: Dict, cfg) -> Dict[str, np.ndarray]:
    xtrans = np.asarray(xtrans, dtype=np.float64).reshape(-1)
    channel_names = list(meta["channel_names"])
    transforms = list(meta["transforms"])
    n_coeff = int(meta["n_coeff"])
    out = {}
    for j, name in enumerate(channel_names):
        sl = slice(j * n_coeff, (j + 1) * n_coeff)
        out[name] = inverse_transform_channel(xtrans[sl], transforms[j]).reshape(n_coeff)
    return out


def reconstruct_channels_from_z(z: np.ndarray, prior: Dict[str, np.ndarray], meta: Dict, cfg, n_modes: Optional[int] = None) -> Dict[str, np.ndarray]:
    xtrans = reconstruct_prior_matrix_from_z(z, prior, n_modes=n_modes)
    return split_prior_matrix_to_channels(xtrans, meta, cfg)


def channels_to_node_feature_raw(channels_coeff: Dict[str, np.ndarray], x_coeff: np.ndarray, x_node: np.ndarray, cfg) -> np.ndarray:
    """Create raw Step3 node features with shape (1, n_node, 9)."""
    x_coeff = np.asarray(x_coeff, dtype=np.float64).reshape(-1)
    x_node = np.asarray(x_node, dtype=np.float64).reshape(-1)
    feature_names = list(cfg.NODE_FEATURE_NAMES)
    features = np.empty((1, len(x_node), len(feature_names)), dtype=np.float64)
    for j, name in enumerate(feature_names):
        if name == "x_m":
            features[0, :, j] = x_node
        else:
            values = np.asarray(channels_coeff[name], dtype=np.float64).reshape(-1)
            order = np.argsort(x_coeff)
            features[0, :, j] = np.interp(x_node, x_coeff[order], values[order], left=values[order][0], right=values[order][-1])
    return features


def load_latent_prior_npz(path: Path | str) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
    d = load_npz(path)
    prior = {
        "data_mean": d["data_mean"],
        "data_scale": d["data_scale"],
        "components": d["components"],
        "eigenvalues": d["eigenvalues"],
        "explained_ratio_all": d["explained_ratio_all"],
        "scores_all": d["scores_all"],
    }
    meta = {
        "channel_names": safe_str_array(d["channel_names"]),
        "transforms": safe_str_array(d["transforms"]),
        "n_samples": int(d["n_samples"]),
        "n_coeff": int(d["n_coeff"]),
        "x_coeff_grid_m": d["x_coeff_grid_m"],
    }
    quantiles = {}
    for key in d.keys():
        if key.startswith("q01_"):
            name = key[4:]
            quantiles.setdefault(name, {})["q01"] = d[key]
        if key.startswith("q99_"):
            name = key[4:]
            quantiles.setdefault(name, {})["q99"] = d[key]
    return prior, meta, quantiles


# =============================================================================
# Step4D surrogate model utilities
# =============================================================================
class MLPForward(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_hidden_layers: int = 5,
                 dropout: float = 0.0, activation: str = "gelu", use_layer_norm: bool = True):
        if nn is None:
            raise RuntimeError("PyTorch is required for MLPForward.")
        super().__init__()
        act_name = str(activation).lower()
        if act_name == "relu":
            act_factory = nn.ReLU
        elif act_name == "tanh":
            act_factory = nn.Tanh
        elif act_name == "silu":
            act_factory = nn.SiLU
        else:
            act_factory = nn.GELU
        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(num_hidden_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            if bool(use_layer_norm):
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(act_factory())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(int(hidden_dim), int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model_from_ckpt_config(config: Dict):
    return MLPForward(
        input_dim=int(config.get("INPUT_DIM", 4500)),
        output_dim=int(config.get("OUTPUT_DIM", 500)),
        hidden_dim=int(config.get("HIDDEN_DIM", 256)),
        num_hidden_layers=int(config.get("NUM_HIDDEN_LAYERS", 5)),
        dropout=float(config.get("DROPOUT", 0.0)),
        activation=str(config.get("ACTIVATION", "gelu")),
        use_layer_norm=bool(config.get("USE_LAYER_NORM", True)),
    )


@dataclass
class LoadedSurrogate:
    checkpoint_path: Path
    model: object
    device: object
    config: Dict
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


def load_step4d_surrogate(checkpoint_path: Path | str, device_mode: str = "cuda_if_available") -> LoadedSurrogate:
    if torch is None:
        raise RuntimeError(f"PyTorch could not be imported: {_TORCH_IMPORT_ERROR}")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Step4D checkpoint not found: {checkpoint_path}")
    device = get_device(device_mode)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model = build_model_from_ckpt_config(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return LoadedSurrogate(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        config=config,
        x_mean=np.asarray(ckpt["x_mean"], dtype=np.float32),
        x_std=np.asarray(ckpt["x_std"], dtype=np.float32),
        y_mean=np.asarray(ckpt["y_mean"], dtype=np.float32),
        y_std=np.asarray(ckpt["y_std"], dtype=np.float32),
    )


def predict_w_m_from_features_norm(features_norm: np.ndarray, surrogate: LoadedSurrogate) -> np.ndarray:
    """Predict w in meters from Step3-normalized node features."""
    X = np.asarray(features_norm, dtype=np.float32)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    X2 = (X - surrogate.x_mean.reshape(1, -1)) / (surrogate.x_std.reshape(1, -1) + 1.0e-12)
    with torch.no_grad():
        xb = torch.as_tensor(X2, dtype=torch.float32, device=surrogate.device)
        pred_norm = surrogate.model(xb).detach().cpu().numpy().astype(np.float64)
    pred_m = pred_norm * surrogate.y_std.astype(np.float64).reshape(1, -1) + surrogate.y_mean.astype(np.float64).reshape(1, -1)
    # Hard boundary projection inherited from Step4D.
    if pred_m.shape[1] >= 2:
        pred_m[:, 0] = 0.0
        pred_m[:, -1] = 0.0
    return pred_m


# =============================================================================
# Metrics and objective utilities
# =============================================================================
def regression_metrics_mm(y_true_m: np.ndarray, y_pred_m: np.ndarray, x_m: Optional[np.ndarray] = None,
                          peak_exclude_boundary_nodes: int = 0) -> Dict[str, float]:
    yt = np.asarray(y_true_m, dtype=np.float64).reshape(-1) * 1000.0
    yp = np.asarray(y_pred_m, dtype=np.float64).reshape(-1) * 1000.0
    err = yp - yt
    mse = np.mean(err ** 2)
    mae = np.mean(np.abs(err))
    sse = np.sum(err ** 2)
    sst = np.sum((yt - np.mean(yt)) ** 2)
    out = {
        "RMSE_mm": float(np.sqrt(mse)),
        "MAE_mm": float(mae),
        "R2": float(1.0 - sse / (sst + 1.0e-30)),
        "MaxAbsError_mm": float(np.max(np.abs(err))),
    }
    m = len(yt)
    lo = int(peak_exclude_boundary_nodes)
    hi = m - int(peak_exclude_boundary_nodes)
    if lo >= hi:
        lo, hi = 0, m
    true_idx = int(np.argmax(np.abs(yt[lo:hi])) + lo)
    pred_idx = int(np.argmax(np.abs(yp[lo:hi])) + lo)
    out.update({
        "true_peak_mm": float(yt[true_idx]),
        "pred_peak_mm": float(yp[pred_idx]),
        "peak_abs_error_mm": float(abs(yp[pred_idx] - yt[true_idx])),
        "true_peak_index": int(true_idx),
        "pred_peak_index": int(pred_idx),
    })
    if x_m is not None:
        x = np.asarray(x_m).reshape(-1)
        out["true_peak_x_m"] = float(x[true_idx])
        out["pred_peak_x_m"] = float(x[pred_idx])
        out["peak_position_error_m"] = float(abs(x[pred_idx] - x[true_idx]))
    return out


def field_error_metrics(true_features_raw: np.ndarray, inv_features_raw: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    out = {}
    true = np.asarray(true_features_raw, dtype=np.float64)
    inv = np.asarray(inv_features_raw, dtype=np.float64)
    if true.ndim == 3:
        true = true[0]
    if inv.ndim == 3:
        inv = inv[0]
    for j, name in enumerate(feature_names):
        if name == "x_m":
            continue
        err = inv[:, j] - true[:, j]
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"{name}_mae"] = float(np.mean(np.abs(err)))
        denom = np.std(true[:, j]) + 1.0e-30
        out[f"{name}_nrmse_by_std"] = float(np.sqrt(np.mean(err ** 2)) / denom)
    return out


def evenly_spaced_obs_indices(n_points: int, n_obs: int, exclude_boundary: int = 0) -> np.ndarray:
    lo = int(exclude_boundary)
    hi = int(n_points) - int(exclude_boundary) - 1
    if hi <= lo:
        return np.arange(n_points, dtype=int)
    n_obs = min(int(n_obs), hi - lo + 1)
    return np.unique(np.linspace(lo, hi, n_obs).round().astype(int))


def bound_penalty_channels(channels: Dict[str, np.ndarray], quantiles: Dict[str, Dict[str, np.ndarray]]) -> float:
    total = 0.0
    count = 0
    for name, qs in quantiles.items():
        if name not in channels or "q01" not in qs or "q99" not in qs:
            continue
        v = np.asarray(channels[name], dtype=np.float64)
        q01 = np.asarray(qs["q01"], dtype=np.float64)
        q99 = np.asarray(qs["q99"], dtype=np.float64)
        width = np.maximum(q99 - q01, 1.0e-12)
        low = np.maximum(q01 - v, 0.0) / width
        high = np.maximum(v - q99, 0.0) / width
        total += float(np.mean(low ** 2 + high ** 2))
        count += 1
    return total / max(count, 1)


@dataclass
class InversionContext:
    prior: Dict[str, np.ndarray]
    meta: Dict
    quantiles: Dict[str, Dict[str, np.ndarray]]
    cfg: object
    step3_stats: Dict
    surrogate: LoadedSurrogate
    x_node_m: np.ndarray
    y_obs_m: np.ndarray
    obs_indices: np.ndarray
    n_modes: int
    lambda_prior: float
    lambda_bound: float


def predict_from_z(z: np.ndarray, ctx: InversionContext) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    channels = reconstruct_channels_from_z(z, ctx.prior, ctx.meta, ctx.cfg, n_modes=ctx.n_modes)
    features_raw = channels_to_node_feature_raw(channels, ctx.meta["x_coeff_grid_m"], ctx.x_node_m, ctx.cfg)
    features_norm = normalize_node_features_raw(features_raw, list(ctx.cfg.NODE_FEATURE_NAMES), ctx.step3_stats)
    y_pred_m = predict_w_m_from_features_norm(features_norm, ctx.surrogate)[0]
    return y_pred_m, features_raw[0], channels


def inversion_objective(z: np.ndarray, ctx: InversionContext) -> float:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    try:
        y_pred_m, _, channels = predict_from_z(z, ctx)
        oi = np.asarray(ctx.obs_indices, dtype=int)
        err_mm = (y_pred_m[oi] - ctx.y_obs_m[oi]) * 1000.0
        obs_loss = float(np.mean(err_mm ** 2))
        prior_loss = float(np.mean(z ** 2))
        bpen = bound_penalty_channels(channels, ctx.quantiles)
        total = obs_loss + float(ctx.lambda_prior) * prior_loss + float(ctx.lambda_bound) * bpen
        if not np.isfinite(total):
            return 1.0e30
        return float(total)
    except Exception:
        return 1.0e30


def fallback_random_search(ctx: InversionContext, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    n_modes = int(ctx.n_modes)
    bound = float(getattr(ctx.cfg, "LATENT_BOUND", 3.0))
    n_eval = int(getattr(ctx.cfg, "MAXFEV", 2500))
    best_z = np.zeros(n_modes, dtype=np.float64)
    best_f = inversion_objective(best_z, ctx)
    scale = 1.0
    for it in range(n_eval):
        cand = best_z + rng.normal(0.0, scale, size=n_modes)
        cand = np.clip(cand, -bound, bound)
        f = inversion_objective(cand, ctx)
        if f < best_f:
            best_z, best_f = cand, f
            scale = max(scale * 0.98, 0.05)
        else:
            scale = max(scale * 0.999, 0.05)
    return best_z, {"success": True, "method": "fallback_random_search", "fun": best_f, "nfev": n_eval, "message": "SciPy not available; used fallback random search."}


def run_inversion(ctx: InversionContext, seed: int = 0, logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(seed)
    n_modes = int(ctx.n_modes)
    bound = float(getattr(ctx.cfg, "LATENT_BOUND", 3.0))
    starts = [np.zeros(n_modes, dtype=np.float64)]
    for _ in range(int(getattr(ctx.cfg, "N_RANDOM_STARTS", 4))):
        starts.append(np.clip(rng.normal(0.0, float(getattr(ctx.cfg, "RANDOM_START_SCALE", 0.5)), size=n_modes), -bound, bound))

    best_z = None
    best_info = None
    best_f = float("inf")
    if minimize is None:
        if logger is not None:
            logger.warning("scipy.optimize.minimize is not available. Using fallback random search.")
        return fallback_random_search(ctx, rng)

    bounds = [(-bound, bound)] * n_modes
    for i, z0 in enumerate(starts):
        try:
            res = minimize(
                lambda zz: inversion_objective(zz, ctx),
                z0,
                method=str(getattr(ctx.cfg, "OPT_METHOD", "Powell")),
                bounds=bounds,
                options={
                    "maxiter": int(getattr(ctx.cfg, "MAXITER", 250)),
                    "maxfev": int(getattr(ctx.cfg, "MAXFEV", 2500)),
                    "xtol": float(getattr(ctx.cfg, "XTOL", 1e-5)),
                    "ftol": float(getattr(ctx.cfg, "FTOL", 1e-8)),
                    "disp": False,
                },
            )
            f = float(res.fun)
            info = {
                "success": bool(res.success),
                "method": str(getattr(ctx.cfg, "OPT_METHOD", "Powell")),
                "fun": f,
                "nfev": int(getattr(res, "nfev", -1)),
                "nit": int(getattr(res, "nit", -1)),
                "message": str(res.message),
                "start_id": int(i),
            }
            if logger is not None:
                logger.info(f"  start {i}: f={f:.6e}, success={res.success}, nfev={info['nfev']}")
            if f < best_f:
                best_f = f
                best_z = np.asarray(res.x, dtype=np.float64)
                best_info = info
        except Exception as exc:
            if logger is not None:
                logger.warning(f"Optimization start {i} failed: {exc}")
    if best_z is None:
        return fallback_random_search(ctx, rng)
    return best_z, best_info


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_curve_comparison(x_m: np.ndarray, y_true_m: np.ndarray, y_init_m: np.ndarray, y_inv_m: np.ndarray,
                          out_dir: Path, base_name: str, cfg, title: str = "Curve inversion", obs_indices: Optional[np.ndarray] = None,
                          logger: Optional[logging.Logger] = None) -> None:
    setup_plot_style(cfg)
    fig, ax = plt.subplots(figsize=(cm_to_inch(float(getattr(cfg, "FIG_WIDTH_CM", 10.0))), cm_to_inch(float(getattr(cfg, "FIG_HEIGHT_CM", 7.0)))))
    ax.plot(x_m, np.asarray(y_true_m) * 1000.0, "-", label="Target")
    ax.plot(x_m, np.asarray(y_init_m) * 1000.0, "--", label="Initial mean")
    ax.plot(x_m, np.asarray(y_inv_m) * 1000.0, "-.", label="Inverted")
    if obs_indices is not None:
        oi = np.asarray(obs_indices, dtype=int)
        ax.plot(np.asarray(x_m)[oi], np.asarray(y_true_m)[oi] * 1000.0, "o", markersize=2.5, label="Observation points")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Vertical displacement w (mm)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, out_dir, base_name, cfg, logger)
    plt.close(fig)


def plot_feature_comparison(x_m: np.ndarray, true_features_raw: np.ndarray, inv_features_raw: np.ndarray,
                            feature_names: List[str], names_to_plot: List[str], out_dir: Path, base_name: str, cfg,
                            logger: Optional[logging.Logger] = None) -> None:
    setup_plot_style(cfg)
    true = np.asarray(true_features_raw)
    inv = np.asarray(inv_features_raw)
    if true.ndim == 3:
        true = true[0]
    if inv.ndim == 3:
        inv = inv[0]
    for name in names_to_plot:
        if name not in feature_names:
            continue
        j = feature_names.index(name)
        fig, ax = plt.subplots(figsize=(cm_to_inch(float(getattr(cfg, "FIG_WIDTH_CM", 10.0))), cm_to_inch(float(getattr(cfg, "FIG_HEIGHT_CM", 7.0)))))
        ax.plot(x_m, true[:, j], "-", label="Target field")
        ax.plot(x_m, inv[:, j], "--", label="Inverted field")
        ax.set_xlabel("x (m)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} comparison")
        ax.legend(frameon=False)
        fig.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        save_figure(fig, out_dir, f"{base_name}_{safe_name}", cfg, logger)
        plt.close(fig)


# =============================================================================
# Step6 TopJournal V2 helper utilities
# =============================================================================
def locate_latest_prior_npz(cfg) -> Path:
    """Locate the validated Step6A latent-prior file."""
    if bool(getattr(cfg, "USE_LATEST_STEP6A_PRIOR", True)):
        roots = [Path(getattr(cfg, "STEP6_V1_OUTPUT_ROOT", "")), Path(getattr(cfg, "OUTPUT_ROOT", ""))]
        cands = []
        for root in roots:
            if root.exists():
                cands.extend(root.glob("Step6A_LatentPriorModes_*/step6_latent_prior_modes.npz"))
                cands.extend(root.glob("Step6V2A_PriorModeDiagnostics_*/step6_latent_prior_modes.npz"))
        cands = [p for p in cands if p.exists()]
        if cands:
            return max(cands, key=lambda p: p.stat().st_mtime)
    p = Path(getattr(cfg, "MANUAL_PRIOR_NPZ"))
    if not p.exists():
        raise FileNotFoundError(f"No latent prior file found. Expected: {p}. Run Step6A V1 first.")
    return p


def latest_output_dir(root: Path | str, prefix: str) -> Optional[Path]:
    root = Path(root)
    if not root.exists():
        return None
    cands = [p for p in root.glob(prefix + "*") if p.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def locate_latest_v2b_dir(cfg) -> Path:
    p = latest_output_dir(Path(cfg.OUTPUT_ROOT), "Step6V2B_ExpandedBenchmark_")
    if p is None:
        raise FileNotFoundError("No Step6V2B expanded benchmark output found. Run step6v2b first.")
    return p


def locate_latest_v2c_dir(cfg) -> Optional[Path]:
    return latest_output_dir(Path(cfg.OUTPUT_ROOT), "Step6V2C_SparseNoiseBenchmark_")


def locate_latest_v2d_dir(cfg) -> Optional[Path]:
    return latest_output_dir(Path(cfg.OUTPUT_ROOT), "Step6V2D_ModeCountSelection_")


def locate_latest_v2e_dir(cfg) -> Optional[Path]:
    return latest_output_dir(Path(cfg.OUTPUT_ROOT), "Step6V2E_PriorAblation_")


def locate_latest_v2f_dir(cfg) -> Optional[Path]:
    return latest_output_dir(Path(cfg.OUTPUT_ROOT), "Step6V2F_OracleProjection_")


def locate_latest_v2g_dir(cfg) -> Optional[Path]:
    return latest_output_dir(Path(cfg.OUTPUT_ROOT), "Step6V2G_LTSBSMBackcheck_")


def compute_initial_mean_prediction(cfg, prior, meta, quantiles, step3_stats, surrogate, x_node_m, n_modes=None):
    if n_modes is None:
        n_modes = int(getattr(cfg, "MAIN_N_MODES", 8))
    ctx = InversionContext(
        prior=prior,
        meta=meta,
        quantiles=quantiles,
        cfg=cfg,
        step3_stats=step3_stats,
        surrogate=surrogate,
        x_node_m=x_node_m,
        y_obs_m=np.zeros_like(x_node_m, dtype=np.float64),
        obs_indices=np.arange(len(x_node_m), dtype=int),
        n_modes=int(n_modes),
        lambda_prior=float(getattr(cfg, "LAMBDA_PRIOR", 1e-3)),
        lambda_bound=float(getattr(cfg, "LAMBDA_BOUND", 1e-2)),
    )
    y0, f0, ch0 = predict_from_z(np.zeros(int(n_modes)), ctx)
    return y0, f0, ch0


def select_stratified_indices_by_metric(metric: np.ndarray, n_cases: int, mode: str = "quantile") -> np.ndarray:
    metric = np.asarray(metric, dtype=np.float64).reshape(-1)
    n = len(metric)
    n_cases = min(int(n_cases), n)
    if n_cases <= 0:
        return np.array([], dtype=int)
    if mode == "first":
        return np.arange(n_cases, dtype=int)
    qs = np.linspace(0.02, 0.98, n_cases)
    ids = []
    for q in qs:
        val = float(np.quantile(metric, q))
        order = np.argsort(np.abs(metric - val))
        for j in order:
            j = int(j)
            if j not in ids:
                ids.append(j)
                break
    # Fill possible duplicates with sorted high-low alternating cases.
    order_full = list(np.argsort(metric))
    fill_order = []
    lo, hi = 0, len(order_full) - 1
    while lo <= hi:
        fill_order.append(int(order_full[lo])); lo += 1
        if lo <= hi:
            fill_order.append(int(order_full[hi])); hi -= 1
    for j in fill_order:
        if len(ids) >= n_cases:
            break
        if j not in ids:
            ids.append(j)
    return np.array(ids[:n_cases], dtype=int)


def select_topjournal_cases(test_dict, y_init_m: np.ndarray, n_cases: int, strategy: str = "stratified_initial_rmse", exclude_boundary: int = 3):
    y_all = np.asarray(test_dict["w_m_raw"], dtype=np.float64)
    n = y_all.shape[0]
    if strategy == "first":
        return np.arange(min(n_cases, n), dtype=int)
    if strategy == "stratified_peak":
        metric = np.max(np.abs(y_all * 1000.0), axis=1)
    else:
        lo = int(exclude_boundary); hi = y_all.shape[1] - int(exclude_boundary)
        err = (y_init_m.reshape(1, -1) - y_all)[:, lo:hi] * 1000.0
        metric = np.sqrt(np.mean(err ** 2, axis=1))
    return select_stratified_indices_by_metric(metric, n_cases, mode="quantile")


def get_sample_identifiers(test_dict, local_idx: int) -> Tuple[int, str]:
    if "global_sample_indices" in test_dict:
        global_idx = int(np.asarray(test_dict["global_sample_indices"]).reshape(-1)[local_idx])
    else:
        global_idx = int(local_idx)
    if "sample_ids" in test_dict:
        sample_id = str(np.asarray(test_dict["sample_ids"]).reshape(-1)[local_idx])
    else:
        sample_id = str(global_idx + 1)
    return global_idx, sample_id


def make_context(cfg, prior, meta, quantiles, step3_stats, surrogate, x_node_m, y_obs_m, obs_indices, n_modes=None, lambda_prior=None, lambda_bound=None):
    return InversionContext(
        prior=prior,
        meta=meta,
        quantiles=quantiles,
        cfg=cfg,
        step3_stats=step3_stats,
        surrogate=surrogate,
        x_node_m=np.asarray(x_node_m, dtype=np.float64),
        y_obs_m=np.asarray(y_obs_m, dtype=np.float64),
        obs_indices=np.asarray(obs_indices, dtype=int),
        n_modes=int(n_modes if n_modes is not None else getattr(cfg, "MAIN_N_MODES", 8)),
        lambda_prior=float(lambda_prior if lambda_prior is not None else getattr(cfg, "LAMBDA_PRIOR", 1e-3)),
        lambda_bound=float(lambda_bound if lambda_bound is not None else getattr(cfg, "LAMBDA_BOUND", 1e-2)),
    )


def compute_improvement(initial_rmse: float, inv_rmse: float) -> float:
    if not np.isfinite(initial_rmse) or abs(initial_rmse) < 1e-30:
        return np.nan
    return float((initial_rmse - inv_rmse) / initial_rmse * 100.0)


def summarize_metric_df(df: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
    out = {}
    for col in ["initial_RMSE_mm", "inv_RMSE_mm", "oracle_RMSE_mm", "ltsbsm_backcheck_RMSE_mm", "inv_R2", "inv_peak_abs_error_mm", "improvement_percent"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[f"{prefix}mean_{col}"] = float(vals.mean())
            out[f"{prefix}median_{col}"] = float(vals.median())
            out[f"{prefix}p90_{col}"] = float(vals.quantile(0.90))
            out[f"{prefix}max_{col}"] = float(vals.max())
    if "optimizer_success" in df.columns:
        out[f"{prefix}success_rate"] = float(pd.Series(df["optimizer_success"]).astype(bool).mean())
    return out


def plot_initial_vs_inverted_rmse(df: pd.DataFrame, out_dir: Path, base_name: str, cfg, logger=None):
    setup_plot_style(cfg)
    fig, ax = plt.subplots(figsize=(cm_to_inch(getattr(cfg, "SUMMARY_FIG_WIDTH_CM", 10.0)), cm_to_inch(getattr(cfg, "SUMMARY_FIG_HEIGHT_CM", 6.5))))
    x = np.arange(len(df)) + 1
    ax.plot(x, df["initial_RMSE_mm"], "o-", label="Initial mean")
    ax.plot(x, df["inv_RMSE_mm"], "s-", label="Inverted")
    ax.set_xlabel("Test case")
    ax.set_ylabel("RMSE (mm)")
    ax.set_title("Expanded synthetic inversion benchmark")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, base_name, cfg, logger)
    plt.close(fig)


def plot_metric_box(df: pd.DataFrame, group_col: str, value_col: str, out_dir: Path, base_name: str, cfg, ylabel: str = None, logger=None):
    setup_plot_style(cfg)
    groups = list(dict.fromkeys(df[group_col].tolist()))
    data = [pd.to_numeric(df.loc[df[group_col] == g, value_col], errors="coerce").dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(cm_to_inch(getattr(cfg, "SUMMARY_FIG_WIDTH_CM", 10.0)), cm_to_inch(getattr(cfg, "SUMMARY_FIG_HEIGHT_CM", 6.5))))
    ax.boxplot(data, labels=[str(g) for g in groups], showmeans=True)
    ax.set_xlabel(group_col)
    ax.set_ylabel(ylabel or value_col)
    ax.set_title(value_col)
    fig.tight_layout()
    save_figure(fig, out_dir, base_name, cfg, logger)
    plt.close(fig)


def plot_heatmap_table(pivot: pd.DataFrame, out_dir: Path, base_name: str, cfg, title: str, cbar_label: str, logger=None):
    setup_plot_style(cfg)
    fig, ax = plt.subplots(figsize=(cm_to_inch(getattr(cfg, "SUMMARY_FIG_WIDTH_CM", 10.0)), cm_to_inch(getattr(cfg, "SUMMARY_FIG_HEIGHT_CM", 6.5))))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1])); ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0])); ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Noise standard deviation (mm)")
    ax.set_ylabel("Number of observation points")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.4f}", ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    save_figure(fig, out_dir, base_name, cfg, logger)
    plt.close(fig)


def copy_if_exists(src: Path | str, dst_dir: Path | str, new_name: Optional[str] = None) -> Optional[Path]:
    src = Path(src); dst_dir = ensure_dir(dst_dir)
    if not src.exists():
        return None
    dst = dst_dir / (new_name if new_name else src.name)
    shutil.copy2(src, dst)
    return dst
