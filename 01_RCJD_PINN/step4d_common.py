# -*- coding: utf-8 -*-
"""Common utilities for Step4D RCJD-PINN forward modeling."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Basic utilities
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(cfg) -> torch.device:
    mode = str(getattr(cfg, "DEVICE", "cuda_if_available")).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(obj, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def config_to_dict(cfg) -> Dict:
    out = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        v = getattr(cfg, k)
        if callable(v):
            continue
        if isinstance(v, (str, int, float, bool, list, tuple, dict, Path, type(None))):
            out[k] = to_jsonable(v)
    return out


def namespace_from_config_dict(cfg_module, cfg_dict: Optional[Dict] = None):
    base = config_to_dict(cfg_module)
    if cfg_dict:
        base.update(cfg_dict)
    # Convert likely path fields back to Path objects.
    for k in list(base.keys()):
        if k.endswith("_DIR") or k.endswith("_ROOT") or k.endswith("_NPZ") or k in ["TRAIN_NPZ", "VAL_NPZ", "TEST_NPZ"]:
            base[k] = Path(base[k])
    return SimpleNamespace(**base)


def snapshot_source_code(out_dir: Path, files: Optional[Iterable[str]] = None) -> None:
    src_dir = ensure_dir(out_dir / "source_code_snapshot")
    if files is None:
        files = [
            "step4d_rcjd_pinn_forward_config.py",
            "step4d_common.py",
            "step4d_check_dataset.py",
            "step4d_rcjd_pinn_forward_train.py",
            "step4d_rcjd_pinn_forward_postprocess.py",
        ]
    here = Path(__file__).resolve().parent
    for fn in files:
        p = here / fn
        if p.exists():
            shutil.copy2(p, src_dir / fn)


# =============================================================================
# Data loading
# =============================================================================

def npz_keys(path: Path | str) -> List[str]:
    with np.load(path, allow_pickle=True) as data:
        return list(data.files)


def load_npz_array(path: Path | str, key: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {data.files}")
        return np.asarray(data[key])


def load_npz_optional(path: Path | str, key: str):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        if key in data.files:
            return np.asarray(data[key])
    return None


def flatten_features(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported array shape for flattening: {arr.shape}")


def load_global_indices(npz_path: Path, n: int, fallback_start: int, cfg) -> np.ndarray:
    for key in getattr(cfg, "GLOBAL_INDEX_KEYS", ["global_sample_indices", "sample_ids"]):
        v = load_npz_optional(npz_path, key)
        if v is not None:
            v = np.asarray(v).reshape(-1).astype(np.int64)
            if len(v) == n:
                return v
    return np.arange(fallback_start, fallback_start + n, dtype=np.int64)


def load_coeffs(npz_path: Path, cfg) -> Dict[str, np.ndarray]:
    coeffs = {}
    for name, key_attr in [("B2", "B2_KEY"), ("B0", "B0_KEY"), ("C2", "C2_KEY"), ("C0", "C0_KEY")]:
        key = getattr(cfg, key_attr, None)
        if key:
            arr = load_npz_optional(npz_path, key)
            if arr is not None:
                coeffs[name] = flatten_features(arr).astype(np.float32)
    return coeffs


def load_split(npz_path: Path, split_name: str, cfg, fallback_start: int = 0) -> Dict[str, np.ndarray]:
    X_raw = load_npz_array(npz_path, cfg.X_KEY)
    Y_raw = load_npz_array(npz_path, cfg.Y_KEY)
    X = flatten_features(X_raw).astype(np.float32)
    Y_m = flatten_features(Y_raw).astype(np.float32)
    coeffs = load_coeffs(npz_path, cfg)
    idx = load_global_indices(npz_path, X.shape[0], fallback_start, cfg)
    d = {"split": split_name, "X": X, "Y_m": Y_m, "global_idx": idx}
    d.update(coeffs)
    return d


def load_all_splits(cfg) -> Dict[str, Dict[str, np.ndarray]]:
    n_train = int(getattr(cfg, "EXPECTED_N_TRAIN", 1400))
    n_val = int(getattr(cfg, "EXPECTED_N_VAL", 300))
    return {
        "train": load_split(Path(cfg.TRAIN_NPZ), "train", cfg, 0),
        "val": load_split(Path(cfg.VAL_NPZ), "val", cfg, n_train),
        "test": load_split(Path(cfg.TEST_NPZ), "test", cfg, n_train + n_val),
    }


def compute_standardizers(X_train: np.ndarray, Y_train_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = X_train.mean(axis=0).astype(np.float32)
    x_std = X_train.std(axis=0).astype(np.float32)
    x_std = np.where(x_std < 1.0e-12, 1.0, x_std).astype(np.float32)
    y_mean = Y_train_m.mean(axis=0).astype(np.float32)
    y_std = Y_train_m.std(axis=0).astype(np.float32)
    y_std = np.where(y_std < 1.0e-12, 1.0, y_std).astype(np.float32)
    return x_mean, x_std, y_mean, y_std


class Step4DDataset(Dataset):
    def __init__(self, split_dict: Dict[str, np.ndarray], x_mean, x_std, y_mean, y_std):
        self.X = ((split_dict["X"] - x_mean) / (x_std + 1.0e-12)).astype(np.float32)
        self.Y_m = split_dict["Y_m"].astype(np.float32)
        self.Y = ((self.Y_m - y_mean) / (y_std + 1.0e-12)).astype(np.float32)
        self.global_idx = split_dict["global_idx"].astype(np.int64)
        n = self.X.shape[0]
        self.B2 = split_dict.get("B2", np.zeros((n, self.Y.shape[1]), dtype=np.float32)).astype(np.float32)
        self.B0 = split_dict.get("B0", np.zeros((n, self.Y.shape[1]), dtype=np.float32)).astype(np.float32)
        self.C2 = split_dict.get("C2", np.zeros((n, self.Y.shape[1]), dtype=np.float32)).astype(np.float32)
        self.C0 = split_dict.get("C0", np.zeros((n, self.Y.shape[1]), dtype=np.float32)).astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return {
            "X": torch.from_numpy(self.X[idx]),
            "Y": torch.from_numpy(self.Y[idx]),
            "Y_m": torch.from_numpy(self.Y_m[idx]),
            "B2": torch.from_numpy(self.B2[idx]),
            "B0": torch.from_numpy(self.B0[idx]),
            "C2": torch.from_numpy(self.C2[idx]),
            "C0": torch.from_numpy(self.C0[idx]),
            "global_idx": torch.tensor(self.global_idx[idx], dtype=torch.long),
        }


def make_dataloaders(splits, x_mean, x_std, y_mean, y_std, cfg):
    train_ds = Step4DDataset(splits["train"], x_mean, x_std, y_mean, y_std)
    val_ds = Step4DDataset(splits["val"], x_mean, x_std, y_mean, y_std)
    test_ds = Step4DDataset(splits["test"], x_mean, x_std, y_mean, y_std)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.BATCH_SIZE), shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=int(cfg.BATCH_SIZE), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(cfg.BATCH_SIZE), shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# =============================================================================
# Model
# =============================================================================

class MLPForward(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_hidden_layers: int = 5,
                 dropout: float = 0.0, activation: str = "gelu", use_layer_norm: bool = True):
        super().__init__()
        act_name = activation.lower()
        if act_name == "relu":
            act_factory = nn.ReLU
        elif act_name == "tanh":
            act_factory = nn.Tanh
        elif act_name == "silu":
            act_factory = nn.SiLU
        else:
            act_factory = nn.GELU

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_factory())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(cfg) -> nn.Module:
    return MLPForward(
        input_dim=int(cfg.INPUT_DIM),
        output_dim=int(cfg.OUTPUT_DIM),
        hidden_dim=int(cfg.HIDDEN_DIM),
        num_hidden_layers=int(cfg.NUM_HIDDEN_LAYERS),
        dropout=float(cfg.DROPOUT),
        activation=str(cfg.ACTIVATION),
        use_layer_norm=bool(getattr(cfg, "USE_LAYER_NORM", True)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Boundary projection and residuals
# =============================================================================

def y_norm_to_m(y_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return y_norm * y_std + y_mean


def y_m_to_norm(y_m: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return (y_m - y_mean) / (y_std + 1.0e-12)


def apply_boundary_projection_m_torch(y_m: torch.Tensor, cfg) -> torch.Tensor:
    if not bool(getattr(cfg, "USE_HARD_BOUNDARY_PROJECTION", False)):
        return y_m
    y = y_m.clone()
    value = float(getattr(cfg, "BOUNDARY_VALUE_M", 0.0))
    for idx in list(getattr(cfg, "BOUNDARY_INDICES", [])):
        y[:, int(idx)] = value
    return y


def apply_boundary_projection_norm_torch(y_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor, cfg) -> torch.Tensor:
    y_m = y_norm_to_m(y_norm, y_mean, y_std)
    y_m = apply_boundary_projection_m_torch(y_m, cfg)
    return y_m_to_norm(y_m, y_mean, y_std)


def apply_boundary_projection_np(y_m: np.ndarray, cfg) -> np.ndarray:
    y = np.array(y_m, copy=True)
    if bool(getattr(cfg, "USE_HARD_BOUNDARY_PROJECTION", False)):
        value = float(getattr(cfg, "BOUNDARY_VALUE_M", 0.0))
        for idx in list(getattr(cfg, "BOUNDARY_INDICES", [])):
            y[:, int(idx)] = value
    return y


def raw_boundary_loss_norm(y_pred_norm_raw: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor, cfg) -> torch.Tensor:
    idxs = list(getattr(cfg, "BOUNDARY_INDICES", []))
    if not idxs:
        return torch.zeros((), dtype=y_pred_norm_raw.dtype, device=y_pred_norm_raw.device)
    target_vals = []
    value_m = float(getattr(cfg, "BOUNDARY_VALUE_M", 0.0))
    for idx in idxs:
        idx = int(idx)
        target_vals.append((value_m - y_mean[idx]) / (y_std[idx] + 1.0e-12))
    target = torch.stack(target_vals).view(1, -1).to(y_pred_norm_raw.device)
    pred = y_pred_norm_raw[:, [int(i) for i in idxs]]
    return torch.mean((pred - target) ** 2)


def ring_interior_indices(cfg) -> torch.Tensor:
    n_rings = int(getattr(cfg, "N_RINGS", 50))
    n_per = int(getattr(cfg, "N_POINTS_PER_RING", 10))
    margin = int(getattr(cfg, "RING_INTERIOR_MARGIN", 2))
    indices = []
    for r in range(n_rings):
        start = r * n_per
        end = start + n_per - 1
        for i in range(start + margin, end - margin + 1):
            if i - 2 >= start and i + 2 <= end:
                indices.append(i)
    return torch.tensor(indices, dtype=torch.long)


def compute_ring_residual(y_m: torch.Tensor, B2: torch.Tensor, B0: torch.Tensor, cfg,
                          return_pointwise: bool = False):
    """Ring-continuity residual using within-ring central finite differences.

    Residual: d4w + B2 d2w + B0 w = 0.
    It is evaluated only at interior nodes of each ring, so the stencil does not
    cross ring joints.
    """
    idx = ring_interior_indices(cfg).to(y_m.device)
    dl = float(getattr(cfg, "DL", 0.13333333333333333))
    w = y_m
    i = idx
    d2 = (w[:, i + 1] - 2.0 * w[:, i] + w[:, i - 1]) / (dl ** 2)
    d4 = (w[:, i - 2] - 4.0 * w[:, i - 1] + 6.0 * w[:, i] - 4.0 * w[:, i + 1] + w[:, i + 2]) / (dl ** 4)
    res = d4 + B2[:, i] * d2 + B0[:, i] * w[:, i]
    scaled = res * float(getattr(cfg, "RING_RESIDUAL_SCALE", 1.0e-4))
    if bool(getattr(cfg, "USE_ADAPTIVE_RING_RESIDUAL_WEIGHTS", True)):
        with torch.no_grad():
            abs_r = torch.abs(scaled)
            mean_abs = torch.mean(abs_r, dim=1, keepdim=True) + 1.0e-12
            weights = 1.0 + float(getattr(cfg, "RING_RESIDUAL_WEIGHT_GAMMA", 1.0)) * abs_r / mean_abs
            weights = torch.clamp(weights, float(getattr(cfg, "RING_RESIDUAL_WEIGHT_MIN", 1.0)), float(getattr(cfg, "RING_RESIDUAL_WEIGHT_MAX", 5.0)))
        loss = torch.mean(weights * scaled ** 2)
    else:
        loss = torch.mean(scaled ** 2)
    if return_pointwise:
        return loss, scaled, idx
    return loss


def joint_indices(cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    n_rings = int(getattr(cfg, "N_RINGS", 50))
    n_per = int(getattr(cfg, "N_POINTS_PER_RING", 10))
    left = []
    right = []
    for r in range(n_rings - 1):
        left.append(r * n_per + n_per - 1)
        right.append((r + 1) * n_per)
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)


def compute_joint_features(y_m: torch.Tensor, cfg):
    """Return joint displacement jumps and rotation jumps at ring joints."""
    left, right = joint_indices(cfg)
    left = left.to(y_m.device)
    right = right.to(y_m.device)
    dl = float(getattr(cfg, "DL", 0.13333333333333333))
    # displacement jump across adjacent ring-end nodes
    disp_jump = y_m[:, right] - y_m[:, left]
    # slope immediately left and right of the joint. Use one-sided derivatives.
    theta_left = (y_m[:, left] - y_m[:, left - 1]) / dl
    theta_right = (y_m[:, right + 1] - y_m[:, right]) / dl
    rot_jump = theta_right - theta_left
    return disp_jump, rot_jump, left, right


def compute_joint_loss(y_pred_m: torch.Tensor, y_true_m: torch.Tensor, cfg, return_parts: bool = False):
    pred_dw, pred_dt, _, _ = compute_joint_features(y_pred_m, cfg)
    true_dw, true_dt, _, _ = compute_joint_features(y_true_m, cfg)
    disp_scale = float(getattr(cfg, "JOINT_DISP_SCALE_TO_MM", 1000.0))
    rot_scale = float(getattr(cfg, "JOINT_ROT_SCALE", 1.0))
    loss_disp = torch.mean(((pred_dw - true_dw) * disp_scale) ** 2)
    loss_rot = torch.mean(((pred_dt - true_dt) * rot_scale) ** 2)
    loss = float(getattr(cfg, "JOINT_DISP_LOSS_WEIGHT", 1.0)) * loss_disp + float(getattr(cfg, "JOINT_ROT_LOSS_WEIGHT", 1.0)) * loss_rot
    if return_parts:
        return loss, loss_disp, loss_rot
    return loss


# =============================================================================
# Metrics
# =============================================================================

def regression_metrics(y_true_m: np.ndarray, y_pred_m: np.ndarray, scale_to_mm: float = 1000.0) -> Dict[str, float]:
    yt = np.asarray(y_true_m, dtype=np.float64) * scale_to_mm
    yp = np.asarray(y_pred_m, dtype=np.float64) * scale_to_mm
    err = yp - yt
    mse = np.mean(err ** 2)
    mae = np.mean(np.abs(err))
    sse = np.sum(err ** 2)
    sst = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1.0 - sse / (sst + 1.0e-30)
    return {"RMSE": float(np.sqrt(mse)), "MAE": float(mae), "R2": float(r2)}


def per_sample_metrics(y_true_m: np.ndarray, y_pred_m: np.ndarray, x_coords: np.ndarray,
                       scale_to_mm: float = 1000.0, peak_exclude_boundary_nodes: int = 0,
                       global_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
    yt = np.asarray(y_true_m, dtype=np.float64) * scale_to_mm
    yp = np.asarray(y_pred_m, dtype=np.float64) * scale_to_mm
    x = np.asarray(x_coords, dtype=np.float64).reshape(-1)
    n, m = yt.shape
    rows = []
    lo = int(peak_exclude_boundary_nodes)
    hi = m - int(peak_exclude_boundary_nodes)
    if lo >= hi:
        lo, hi = 0, m
    for i in range(n):
        err = yp[i] - yt[i]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        sl = slice(lo, hi)
        true_local = np.argmax(np.abs(yt[i, sl])) + lo
        pred_local = np.argmax(np.abs(yp[i, sl])) + lo
        true_peak = float(yt[i, true_local])
        pred_peak = float(yp[i, pred_local])
        rows.append({
            "local_sample_index": i,
            "global_sample_index": int(global_indices[i]) if global_indices is not None else i,
            "rmse_mm": rmse,
            "mae_mm": mae,
            "true_peak_mm": true_peak,
            "pred_peak_mm": pred_peak,
            "peak_abs_error_mm": float(abs(pred_peak - true_peak)),
            "true_peak_index": int(true_local),
            "pred_peak_index": int(pred_local),
            "true_peak_x_m": float(x[true_local]),
            "pred_peak_x_m": float(x[pred_local]),
            "peak_position_error_m": float(abs(x[pred_local] - x[true_local])),
        })
    return pd.DataFrame(rows)


def error_envelope(y_true_m: np.ndarray, y_pred_m: np.ndarray, x_coords: np.ndarray, scale_to_mm: float = 1000.0) -> pd.DataFrame:
    err = (np.asarray(y_pred_m) - np.asarray(y_true_m)) * scale_to_mm
    abs_err = np.abs(err)
    return pd.DataFrame({
        "x_m": np.asarray(x_coords).reshape(-1),
        "mae_mm": np.mean(abs_err, axis=0),
        "rmse_mm": np.sqrt(np.mean(err ** 2, axis=0)),
        "p95_abs_error_mm": np.percentile(abs_err, 95, axis=0),
        "max_abs_error_mm": np.max(abs_err, axis=0),
    })


def get_x_coords(npz_path: Path, cfg, n_points: int) -> np.ndarray:
    arr = load_npz_optional(npz_path, getattr(cfg, "X_COORD_KEY", "x_node_m"))
    if arr is not None:
        arr = np.asarray(arr)
        if arr.ndim == 1 and arr.shape[0] == n_points:
            return arr.astype(np.float64)
        if arr.ndim == 2 and arr.shape[1] == n_points:
            return arr[0].astype(np.float64)
    return np.linspace(-30.0, 30.0, n_points)


def joint_error_metrics(y_true_m: np.ndarray, y_pred_m: np.ndarray, cfg) -> Dict[str, float]:
    yt = torch.as_tensor(y_true_m, dtype=torch.float64)
    yp = torch.as_tensor(y_pred_m, dtype=torch.float64)
    dw_t, dt_t, _, _ = compute_joint_features(yt, cfg)
    dw_p, dt_p, _, _ = compute_joint_features(yp, cfg)
    disp_err_mm = (dw_p - dw_t).numpy() * float(getattr(cfg, "JOINT_DISP_SCALE_TO_MM", 1000.0))
    rot_err = (dt_p - dt_t).numpy()
    return {
        "joint_disp_jump_rmse_mm": float(np.sqrt(np.mean(disp_err_mm ** 2))),
        "joint_disp_jump_mae_mm": float(np.mean(np.abs(disp_err_mm))),
        "joint_disp_jump_max_abs_mm": float(np.max(np.abs(disp_err_mm))),
        "joint_rot_jump_rmse": float(np.sqrt(np.mean(rot_err ** 2))),
        "joint_rot_jump_mae": float(np.mean(np.abs(rot_err))),
        "joint_rot_jump_max_abs": float(np.max(np.abs(rot_err))),
    }


def joint_error_by_joint(y_true_m: np.ndarray, y_pred_m: np.ndarray, cfg) -> pd.DataFrame:
    yt = torch.as_tensor(y_true_m, dtype=torch.float64)
    yp = torch.as_tensor(y_pred_m, dtype=torch.float64)
    dw_t, dt_t, left, right = compute_joint_features(yt, cfg)
    dw_p, dt_p, _, _ = compute_joint_features(yp, cfg)
    disp_err_mm = (dw_p - dw_t).numpy() * float(getattr(cfg, "JOINT_DISP_SCALE_TO_MM", 1000.0))
    rot_err = (dt_p - dt_t).numpy()
    rows = []
    for j in range(disp_err_mm.shape[1]):
        rows.append({
            "joint_id": j + 1,
            "left_node_index": int(left[j]),
            "right_node_index": int(right[j]),
            "disp_jump_mae_mm": float(np.mean(np.abs(disp_err_mm[:, j]))),
            "disp_jump_rmse_mm": float(np.sqrt(np.mean(disp_err_mm[:, j] ** 2))),
            "disp_jump_max_abs_mm": float(np.max(np.abs(disp_err_mm[:, j]))),
            "rot_jump_mae": float(np.mean(np.abs(rot_err[:, j]))),
            "rot_jump_rmse": float(np.sqrt(np.mean(rot_err[:, j] ** 2))),
            "rot_jump_max_abs": float(np.max(np.abs(rot_err[:, j]))),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Checkpoint and prediction
# =============================================================================

def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, cfg, x_mean, x_std, y_mean, y_std,
                    lambda_ring: float, lambda_joint: float, best_val: float, extra: Optional[Dict] = None):
    ensure_dir(path.parent)
    obj = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config_to_dict(cfg),
        "x_mean": np.asarray(x_mean, dtype=np.float32),
        "x_std": np.asarray(x_std, dtype=np.float32),
        "y_mean": np.asarray(y_mean, dtype=np.float32),
        "y_std": np.asarray(y_std, dtype=np.float32),
        "lambda_ring": float(lambda_ring),
        "lambda_joint": float(lambda_joint),
        "best_val": float(best_val),
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_checkpoint(path: Path | str, map_location="cpu") -> Dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def find_latest_stage1_train_dir(cfg, specified: Optional[str] = None) -> Path:
    if specified:
        p = Path(specified)
        if not p.exists():
            raise FileNotFoundError(f"Specified train_dir not found: {p}")
        return p
    root = Path(cfg.OUTPUT_ROOT)
    cands = sorted(root.glob("Step4D_RCJD_PINN_Forward_Stage1Train_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No Stage1Train directory found under {root}")
    return cands[0]


def predict_split(model: nn.Module, split_dict: Dict[str, np.ndarray], x_mean, x_std, y_mean, y_std, cfg,
                  batch_size: int = 256, device: torch.device | str = "cpu") -> Dict[str, np.ndarray]:
    device = torch.device(device)
    X = ((split_dict["X"] - x_mean) / (x_std + 1.0e-12)).astype(np.float32)
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    pred_norm_raw = np.vstack(preds).astype(np.float64)
    pred_m_raw = pred_norm_raw * y_std.astype(np.float64) + y_mean.astype(np.float64)
    pred_m = apply_boundary_projection_np(pred_m_raw, cfg)
    true_m = apply_boundary_projection_np(split_dict["Y_m"].astype(np.float64), cfg)
    return {
        "y_true_m": true_m,
        "y_pred_m": pred_m,
        "y_pred_m_raw": pred_m_raw,
        "y_pred_norm_raw": pred_norm_raw,
        "global_idx": split_dict["global_idx"],
    }
