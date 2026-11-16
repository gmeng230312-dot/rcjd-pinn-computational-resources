# -*- coding: utf-8 -*-
"""Stage2 postprocess for Step4D RCJD-PINN forward model."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import step4d_rcjd_pinn_forward_config as cfg_module
from step4d_common import (
    apply_boundary_projection_np,
    build_model,
    compute_ring_residual,
    config_to_dict,
    count_parameters,
    ensure_dir,
    error_envelope,
    find_latest_stage1_train_dir,
    get_x_coords,
    joint_error_by_joint,
    joint_error_metrics,
    load_all_splits,
    load_checkpoint,
    namespace_from_config_dict,
    now_stamp,
    per_sample_metrics,
    predict_split,
    regression_metrics,
    save_json,
    snapshot_source_code,
    write_text,
)


def cm_to_inch(x):
    return x / 2.54


def setup_matplotlib(cfg):
    plt.rcParams["font.family"] = str(getattr(cfg, "FIG_FONT_FAMILY", "Times New Roman"))
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9


def save_fig(fig, out_dir: Path, name: str, cfg):
    for fmt in list(getattr(cfg, "SAVE_FIG_FORMATS", ["png"])):
        fig.savefig(out_dir / f"{name}.{fmt}", dpi=int(getattr(cfg, "FIG_DPI", 600)), bbox_inches="tight")
    plt.close(fig)


def safe_plot(func, name, *args, **kwargs):
    try:
        func(*args, **kwargs)
        print(f"[Step4D-Stage2] Saved {name}")
    except Exception as e:
        print(f"[Step4D-Stage2-Warning] Failed to plot {name}: {repr(e)}")


def plot_loss_curves(train_dir: Path, out_dir: Path, cfg):
    df = pd.read_csv(train_dir / "train_log.csv")
    fig, ax = plt.subplots(figsize=(cm_to_inch(10), cm_to_inch(7)))
    ax.semilogy(df["epoch"], df["train_data"], label="Train data")
    ax.semilogy(df["epoch"], df["val_data"], label="Val data")
    ax.semilogy(df["epoch"], df["train_ring"].clip(lower=1e-30), label="Train ring")
    ax.semilogy(df["epoch"], df["train_joint"].clip(lower=1e-30), label="Train joint")
    ax.semilogy(df["epoch"], df["val_total"].clip(lower=1e-30), label="Val total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "Fig_D1_loss_curves", cfg)


def plot_lambda_history(train_dir: Path, out_dir: Path, cfg):
    df = pd.read_csv(train_dir / "lambda_history.csv")
    fig, ax = plt.subplots(figsize=(cm_to_inch(10), cm_to_inch(7)))
    ax.semilogy(df["epoch"], df["lambda_ring"], label="lambda_ring")
    ax.semilogy(df["epoch"], df["lambda_joint"], label="lambda_joint")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Adaptive physics weight")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "Fig_D2_lambda_rcjd_evolution", cfg)


def plot_selected_curves(y_true_m, y_pred_m, sample_df, x_coords, out_dir, cfg):
    order = sample_df.sort_values("rmse_mm", ascending=False).head(3)["local_sample_index"].tolist()
    order += [0, len(sample_df) // 2]
    seen = []
    for i in order:
        if i not in seen and 0 <= i < y_true_m.shape[0]:
            seen.append(i)
    fig, ax = plt.subplots(figsize=(cm_to_inch(10), cm_to_inch(7)))
    scale = float(getattr(cfg, "Y_UNIT_SCALE_TO_MM", 1000.0))
    for i in seen[:5]:
        ax.plot(x_coords, y_true_m[i] * scale, lw=1.2, label=f"True {i}")
        ax.plot(x_coords, y_pred_m[i] * scale, lw=1.0, ls="--", label=f"Pred {i}")
    ax.set_xlabel("Longitudinal coordinate, x (m)")
    ax.set_ylabel("Response, w (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_fig(fig, out_dir, "Fig_D3_selected_test_curves", cfg)


def plot_rmse_hist(sample_df, out_dir, cfg):
    fig, ax = plt.subplots(figsize=(cm_to_inch(10), cm_to_inch(7)))
    ax.hist(sample_df["rmse_mm"].to_numpy(), bins=30)
    ax.set_xlabel("Per-sample RMSE (mm)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_dir, "Fig_D4_per_sample_rmse_histogram", cfg)


def plot_error_envelope(env_df, out_dir, cfg):
    fig, ax = plt.subplots(figsize=(cm_to_inch(10), cm_to_inch(7)))
    ax.plot(env_df["x_m"], env_df["mae_mm"], label="MAE")
    ax.plot(env_df["x_m"], env_df["rmse_mm"], label="RMSE")
    ax.plot(env_df["x_m"], env_df["p95_abs_error_mm"], label="95% abs. error")
    ax.set_xlabel("Longitudinal coordinate, x (m)")
    ax.set_ylabel("Error (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "Fig_D5_error_envelope_along_tunnel", cfg)


def plot_ring_residual_envelope(residual_abs_stats, out_dir, cfg):
    x = residual_abs_stats["x_m"]
    fig, ax = plt.subplots(figsize=(cm_to_inch(10), cm_to_inch(7)))
    ax.semilogy(x, residual_abs_stats["mean"], label="Mean")
    ax.semilogy(x, residual_abs_stats["p95"], label="95%")
    ax.semilogy(x, residual_abs_stats["max"], label="Max")
    ax.set_xlabel("Longitudinal coordinate, x (m)")
    ax.set_ylabel("Scaled ring residual magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "Fig_D6_ring_residual_envelope", cfg)


def plot_joint_error(joint_df, out_dir, cfg):
    fig, ax = plt.subplots(figsize=(cm_to_inch(10), cm_to_inch(7)))
    ax.semilogy(joint_df["joint_id"], joint_df["disp_jump_rmse_mm"].clip(lower=1e-12), label="Disp. jump RMSE")
    ax.semilogy(joint_df["joint_id"], joint_df["disp_jump_max_abs_mm"].clip(lower=1e-12), label="Disp. jump max")
    ax.set_xlabel("Joint index")
    ax.set_ylabel("Displacement jump error (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "Fig_D7_joint_discontinuity_error", cfg)


def plot_model_comparison(test_metrics, out_dir, cfg):
    refs = dict(getattr(cfg, "REFERENCE_MODELS", {}))
    refs["Step4D"] = test_metrics
    rows = []
    for name, vals in refs.items():
        rows.append({"model": name, "RMSE": vals.get("RMSE", np.nan), "MAE": vals.get("MAE", np.nan), "R2": vals.get("R2", np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "model_comparison_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(cm_to_inch(10), cm_to_inch(7)))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["RMSE"], width=0.36, label="RMSE")
    ax.bar(x + 0.18, df["MAE"], width=0.36, label="MAE")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylabel("Error metric (mm, log scale)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=False)
    save_fig(fig, out_dir, "Fig_D8_error_metric_comparison", cfg)

    fig, ax = plt.subplots(figsize=(cm_to_inch(10), cm_to_inch(7)))
    ax.bar(x, df["R2"])
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylabel("R2")
    ax.set_ylim(max(0.0, df["R2"].min() - 0.01), 1.0005)
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(fig, out_dir, "Fig_D9_r2_comparison", cfg)


def compute_residual_stats(model, split_dict, ckpt, cfg, x_coords, batch_size=256):
    # For residual stats, use CPU in Stage2 for robustness.
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = torch.as_tensor(np.asarray(ckpt["y_mean"], dtype=np.float32))
    y_std = torch.as_tensor(np.asarray(ckpt["y_std"], dtype=np.float32))
    X = ((split_dict["X"] - x_mean) / (x_std + 1e-12)).astype(np.float32)
    model.cpu().eval()
    all_scaled = []
    idx_ref = None
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32)
            B2 = torch.as_tensor(split_dict.get("B2")[start:start + batch_size], dtype=torch.float32)
            B0 = torch.as_tensor(split_dict.get("B0")[start:start + batch_size], dtype=torch.float32)
            pred_raw = model(xb)
            from step4d_common import apply_boundary_projection_norm_torch, y_norm_to_m
            pred_proj = apply_boundary_projection_norm_torch(pred_raw, y_mean, y_std, cfg)
            pred_m = y_norm_to_m(pred_proj, y_mean, y_std)
            _, scaled, idx = compute_ring_residual(pred_m, B2, B0, cfg, return_pointwise=True)
            all_scaled.append(scaled.cpu().numpy())
            idx_ref = idx.cpu().numpy()
    arr = np.vstack(all_scaled)
    abs_arr = np.abs(arr)
    return {
        "x_m": x_coords[idx_ref],
        "mean": np.mean(abs_arr, axis=0),
        "p95": np.percentile(abs_arr, 95, axis=0),
        "max": np.max(abs_arr, axis=0),
        "abs_mean_global": float(np.mean(abs_arr)),
        "abs_median_global": float(np.median(abs_arr)),
        "abs_p95_global": float(np.percentile(abs_arr, 95)),
        "abs_max_global": float(np.max(abs_arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Step4D RCJD-PINN Stage2 postprocess")
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    train_dir = find_latest_stage1_train_dir(cfg_module, args.train_dir)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        default_name = str(getattr(cfg_module, "STAGE2_DEFAULT_CHECKPOINT", "best_rcjd_model.pt"))
        checkpoint_path = train_dir / default_name
        if not checkpoint_path.exists():
            fallback_names = ["best_rcjd_model.pt", "best_model.pt", "last_model.pt"]
            for name in fallback_names:
                cand = train_dir / name
                if cand.exists():
                    checkpoint_path = cand
                    print(f"[Step4D-Stage2-Warning] Default checkpoint was not found. Using fallback: {name}")
                    break
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = ensure_dir(Path(cfg_module.OUTPUT_ROOT) / f"Step4D_RCJD_PINN_Forward_Stage2Postprocess_{now_stamp()}")
    print(f"[Step4D-Stage2] Train directory: {train_dir}")
    print(f"[Step4D-Stage2] Checkpoint: {checkpoint_path}")
    print(f"[Step4D-Stage2] Output directory: {out_dir}")
    t0 = time.time()

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = namespace_from_config_dict(cfg_module, ckpt.get("config", None))
    # Release adaptation: keep archived model hyperparameters, but resolve all
    # data/output locations from the distribution's current configuration.
    for key in ("CH4_ROOT", "STEP3_DATASET_DIR", "STEP4D_DIR", "OUTPUT_ROOT",
                "TRAIN_NPZ", "VAL_NPZ", "TEST_NPZ"):
        setattr(cfg, key, getattr(cfg_module, key))
    setup_matplotlib(cfg)
    save_json({"train_dir": str(train_dir), "checkpoint": str(checkpoint_path), "config": config_to_dict(cfg)}, out_dir / "stage2_config_and_source.json")
    snapshot_source_code(out_dir)

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = count_parameters(model)
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)

    splits = load_all_splits(cfg)
    pred_train = predict_split(model, splits["train"], x_mean, x_std, y_mean, y_std, cfg, device="cpu")
    pred_val = predict_split(model, splits["val"], x_mean, x_std, y_mean, y_std, cfg, device="cpu")
    pred_test = predict_split(model, splits["test"], x_mean, x_std, y_mean, y_std, cfg, device="cpu")

    scale = float(getattr(cfg, "Y_UNIT_SCALE_TO_MM", 1000.0))
    train_metrics = regression_metrics(pred_train["y_true_m"], pred_train["y_pred_m"], scale)
    val_metrics = regression_metrics(pred_val["y_true_m"], pred_val["y_pred_m"], scale)
    test_metrics = regression_metrics(pred_test["y_true_m"], pred_test["y_pred_m"], scale)

    x_coords = get_x_coords(Path(cfg.TEST_NPZ), cfg, pred_test["y_true_m"].shape[1])
    sample_df = per_sample_metrics(
        pred_test["y_true_m"], pred_test["y_pred_m"], x_coords, scale,
        int(getattr(cfg, "PEAK_SEARCH_EXCLUDE_BOUNDARY_NODES", 0)), pred_test["global_idx"]
    )
    env_df = error_envelope(pred_test["y_true_m"], pred_test["y_pred_m"], x_coords, scale)
    joint_metrics = joint_error_metrics(pred_test["y_true_m"], pred_test["y_pred_m"], cfg)
    joint_df = joint_error_by_joint(pred_test["y_true_m"], pred_test["y_pred_m"], cfg)
    residual_stats = compute_residual_stats(model, splits["test"], ckpt, cfg, x_coords)

    sample_summary = {
        "per_sample_RMSE_mean_mm": float(sample_df["rmse_mm"].mean()),
        "per_sample_RMSE_median_mm": float(sample_df["rmse_mm"].median()),
        "per_sample_RMSE_max_mm": float(sample_df["rmse_mm"].max()),
        "per_sample_MAE_mean_mm": float(sample_df["mae_mm"].mean()),
        "peak_abs_error_mean_mm": float(sample_df["peak_abs_error_mm"].mean()),
        "peak_abs_error_median_mm": float(sample_df["peak_abs_error_mm"].median()),
        "peak_abs_error_max_mm": float(sample_df["peak_abs_error_mm"].max()),
        "peak_position_error_mean_m": float(sample_df["peak_position_error_m"].mean()),
        "peak_position_error_median_m": float(sample_df["peak_position_error_m"].median()),
        "peak_position_error_max_m": float(sample_df["peak_position_error_m"].max()),
    }

    sample_df.to_csv(out_dir / "test_sample_metrics.csv", index=False)
    env_df.to_csv(out_dir / "test_error_envelope_along_tunnel.csv", index=False)
    joint_df.to_csv(out_dir / "test_joint_error_by_joint.csv", index=False)
    np.save(out_dir / "y_true_test.npy", pred_test["y_true_m"])
    np.save(out_dir / "y_pred_test.npy", pred_test["y_pred_m"])
    np.savez_compressed(out_dir / "test_prediction_curves.npz", y_true_m=pred_test["y_true_m"], y_pred_m=pred_test["y_pred_m"], x_coords=x_coords, global_sample_indices=pred_test["global_idx"])
    np.savez_compressed(out_dir / "test_ring_residual_envelope.npz", **residual_stats)

    report = {
        "model_name": cfg.MODEL_NAME,
        "version_tag": str(getattr(cfg, "VERSION_TAG", "V1.2_RCJDCheckpoint")),
        "stage": "Stage2Postprocess",
        "train_dir": str(train_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_type": str(ckpt.get("checkpoint_type", "unknown")),
        "checkpoint_phase": str(ckpt.get("phase", "unknown")),
        "selection_metric": str(ckpt.get("selection_metric", "unknown")),
        "dataset_mode": str(getattr(cfg, "DATASET_MODE", "split_npz")),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "n_params": n_params,
        "ring_residual_scale": float(getattr(cfg, "RING_RESIDUAL_SCALE", np.nan)),
        "lambda_ring": float(ckpt.get("lambda_ring", np.nan)),
        "lambda_joint": float(ckpt.get("lambda_joint", np.nan)),
        "is_formal_rcjd_checkpoint": bool(
            str(ckpt.get("checkpoint_type", "")) == "best_rcjd_model"
            and float(ckpt.get("lambda_ring", 0.0)) > float(getattr(cfg, "RCJD_ARCHIVE_MIN_LAMBDA_RING", 1.0e-8))
            and float(ckpt.get("lambda_joint", 0.0)) > float(getattr(cfg, "RCJD_ARCHIVE_MIN_LAMBDA_JOINT", 1.0e-8))
        ),
        "use_hard_boundary_projection": bool(getattr(cfg, "USE_HARD_BOUNDARY_PROJECTION", False)),
        "boundary_indices": list(getattr(cfg, "BOUNDARY_INDICES", [])),
        "peak_search_exclude_boundary_nodes": int(getattr(cfg, "PEAK_SEARCH_EXCLUDE_BOUNDARY_NODES", 0)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_sample_summary": sample_summary,
        "test_joint_discontinuity_summary": joint_metrics,
        "test_ring_residual_summary": {
            "scaled_residual_abs_mean": residual_stats["abs_mean_global"],
            "scaled_residual_abs_median": residual_stats["abs_median_global"],
            "scaled_residual_abs_p95": residual_stats["abs_p95_global"],
            "scaled_residual_abs_max": residual_stats["abs_max_global"],
        },
        "postprocess_time_sec": float(time.time() - t0),
    }
    save_json(report, out_dir / "test_global_metrics.json")

    if not report["is_formal_rcjd_checkpoint"]:
        print("[Step4D-Stage2-Warning] The evaluated checkpoint is not a formal best_rcjd_model with nonzero RCJD weights. Do not archive as RCJD-PINN.")

    lines = [
        "Step4D RCJD-PINN Forward - Stage2 Summary",
        "=" * 72,
        f"train_dir: {train_dir}",
        f"checkpoint: {checkpoint_path}",
        f"checkpoint_file: {report['checkpoint_file']}",
        f"checkpoint_type: {report['checkpoint_type']}",
        f"checkpoint_phase: {report['checkpoint_phase']}",
        f"selection_metric: {report['selection_metric']}",
        f"is_formal_rcjd_checkpoint: {report['is_formal_rcjd_checkpoint']}",
        f"best_epoch: {report['best_epoch']}",
        f"n_params: {n_params}",
        f"lambda_ring: {report['lambda_ring']:.8e}",
        f"lambda_joint: {report['lambda_joint']:.8e}",
        "",
        "Test metrics:",
        f"  RMSE: {test_metrics['RMSE']:.8e}",
        f"  MAE: {test_metrics['MAE']:.8e}",
        f"  R2: {test_metrics['R2']:.8e}",
        "",
        "Per-sample summary:",
    ]
    for k, v in sample_summary.items():
        lines.append(f"  {k}: {v:.8e}")
    lines.extend(["", "Joint discontinuity summary:"])
    for k, v in joint_metrics.items():
        lines.append(f"  {k}: {v:.8e}")
    lines.extend(["", "Ring residual summary:"])
    for k in ["scaled_residual_abs_mean", "scaled_residual_abs_median", "scaled_residual_abs_p95", "scaled_residual_abs_max"]:
        lines.append(f"  {k}: {report['test_ring_residual_summary'][k]:.8e}")
    write_text(out_dir / "run_step4d_stage2_summary.txt", "\n".join(lines))

    if bool(getattr(cfg, "SAVE_FIGURES", True)):
        safe_plot(plot_loss_curves, "Fig_D1_loss_curves", train_dir, out_dir, cfg)
        safe_plot(plot_lambda_history, "Fig_D2_lambda_rcjd_evolution", train_dir, out_dir, cfg)
        safe_plot(plot_selected_curves, "Fig_D3_selected_test_curves", pred_test["y_true_m"], pred_test["y_pred_m"], sample_df, x_coords, out_dir, cfg)
        safe_plot(plot_rmse_hist, "Fig_D4_per_sample_rmse_histogram", sample_df, out_dir, cfg)
        safe_plot(plot_error_envelope, "Fig_D5_error_envelope_along_tunnel", env_df, out_dir, cfg)
        safe_plot(plot_ring_residual_envelope, "Fig_D6_ring_residual_envelope", residual_stats, out_dir, cfg)
        safe_plot(plot_joint_error, "Fig_D7_joint_discontinuity_error", joint_df, out_dir, cfg)
        safe_plot(plot_model_comparison, "Fig_D8_D9_model_comparison", test_metrics, out_dir, cfg)

    print("[Step4D-Stage2] Done.")
    print(f"[Step4D-Stage2] Test RMSE = {test_metrics['RMSE']:.8e} mm")
    print(f"[Step4D-Stage2] Test MAE  = {test_metrics['MAE']:.8e} mm")
    print(f"[Step4D-Stage2] Test R2   = {test_metrics['R2']:.8e}")
    print(f"[Step4D-Stage2] Joint disp jump RMSE = {joint_metrics['joint_disp_jump_rmse_mm']:.8e} mm")
    print(f"[Step4D-Stage2] Output directory: {out_dir}")


if __name__ == "__main__":
    main()
