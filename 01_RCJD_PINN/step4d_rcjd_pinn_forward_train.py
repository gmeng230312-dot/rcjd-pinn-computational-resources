# -*- coding: utf-8 -*-
"""Stage1 training for Step4D RCJD-PINN forward model V1.2.

V1.2 keeps the stable Step4C V1.2 pipeline and the first Step4D RCJD losses,
but fixes the key checkpoint-selection issue:

- best_data_model.pt is selected during data-only pretraining.
- best_rcjd_model.pt is selected only after the RCJD-constrained phase starts.
- best_model.pt is an alias of best_rcjd_model.pt for compatibility.

This prevents the final Stage2 evaluation from accidentally reporting a
pretraining checkpoint with lambda_ring=lambda_joint=0.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import step4d_rcjd_pinn_forward_config as cfg_module
from step4d_common import (
    build_model,
    compute_joint_loss,
    compute_ring_residual,
    config_to_dict,
    count_parameters,
    ensure_dir,
    get_device,
    load_all_splits,
    make_dataloaders,
    namespace_from_config_dict,
    now_stamp,
    raw_boundary_loss_norm,
    save_checkpoint,
    save_json,
    set_seed,
    snapshot_source_code,
    write_text,
    y_norm_to_m,
    apply_boundary_projection_norm_torch,
    compute_standardizers,
)


def rcjd_best_start_epoch(cfg) -> int:
    return int(getattr(
        cfg,
        "RCJD_BEST_START_EPOCH",
        int(cfg.DATA_PRETRAIN_EPOCHS) + int(getattr(cfg, "ADAPTIVE_LAMBDA_WARMUP_EPOCHS", 0)) + 1,
    ))


def current_phase(epoch: int, cfg) -> str:
    data_end = int(cfg.DATA_PRETRAIN_EPOCHS)
    warm_end = data_end + int(getattr(cfg, "ADAPTIVE_LAMBDA_WARMUP_EPOCHS", 0))
    if epoch <= data_end:
        return "data_pretrain"
    if epoch <= warm_end:
        return "rcjd_warmup"
    if epoch > int(cfg.MAX_EPOCHS) - int(cfg.FINETUNE_EPOCHS):
        return "finetune"
    return "adaptive_rcjd"


def update_lambda(old_lambda: float, data_loss: float, phys_loss: float, lmin: float, lmax: float, cfg) -> float:
    if phys_loss <= 0 or not np.isfinite(phys_loss):
        return float(old_lambda)
    target = (data_loss / (phys_loss + float(cfg.ADAPT_EPS))) ** float(cfg.ADAPT_ALPHA)
    target = float(np.clip(target, lmin, lmax))
    new = float(cfg.ADAPT_EMA) * float(old_lambda) + (1.0 - float(cfg.ADAPT_EMA)) * target
    return float(np.clip(new, lmin, lmax))


def train_or_eval_one_epoch(model, loader, optimizer, device, cfg, y_mean_t, y_std_t,
                            lambda_ring: float, lambda_joint: float, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    totals = {"total": 0.0, "data": 0.0, "ring": 0.0, "joint": 0.0, "joint_disp": 0.0, "joint_rot": 0.0, "bc": 0.0}
    n_total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            xb = batch["X"].to(device, non_blocking=True)
            yb = batch["Y"].to(device, non_blocking=True)
            yb_m = batch["Y_m"].to(device, non_blocking=True)
            B2 = batch["B2"].to(device, non_blocking=True)
            B0 = batch["B0"].to(device, non_blocking=True)

            pred_raw_norm = model(xb)
            pred_proj_norm = apply_boundary_projection_norm_torch(pred_raw_norm, y_mean_t, y_std_t, cfg)
            pred_proj_m = y_norm_to_m(pred_proj_norm, y_mean_t, y_std_t)

            data_loss = F.mse_loss(pred_proj_norm, yb)
            bc_loss = raw_boundary_loss_norm(pred_raw_norm, y_mean_t, y_std_t, cfg)

            ring_loss = torch.zeros((), dtype=torch.float32, device=device)
            if bool(getattr(cfg, "USE_RING_RESIDUAL", True)):
                ring_loss = compute_ring_residual(pred_proj_m, B2, B0, cfg)

            joint_loss = torch.zeros((), dtype=torch.float32, device=device)
            joint_disp_loss = torch.zeros((), dtype=torch.float32, device=device)
            joint_rot_loss = torch.zeros((), dtype=torch.float32, device=device)
            if bool(getattr(cfg, "USE_JOINT_DISCONTINUITY_LOSS", True)):
                joint_loss, joint_disp_loss, joint_rot_loss = compute_joint_loss(pred_proj_m, yb_m, cfg, return_parts=True)

            total_loss = data_loss + float(cfg.RAW_BOUNDARY_LOSS_WEIGHT) * bc_loss + float(lambda_ring) * ring_loss + float(lambda_joint) * joint_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if float(getattr(cfg, "GRAD_CLIP_NORM", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.GRAD_CLIP_NORM))
                optimizer.step()

            bs = xb.shape[0]
            n_total += bs
            totals["total"] += float(total_loss.detach().cpu()) * bs
            totals["data"] += float(data_loss.detach().cpu()) * bs
            totals["ring"] += float(ring_loss.detach().cpu()) * bs
            totals["joint"] += float(joint_loss.detach().cpu()) * bs
            totals["joint_disp"] += float(joint_disp_loss.detach().cpu()) * bs
            totals["joint_rot"] += float(joint_rot_loss.detach().cpu()) * bs
            totals["bc"] += float(bc_loss.detach().cpu()) * bs

    return {k: v / max(n_total, 1) for k, v in totals.items()}


def main():
    cfg = namespace_from_config_dict(cfg_module, None)
    set_seed(int(cfg.SEED))
    device = get_device(cfg)

    out_dir = ensure_dir(Path(cfg.OUTPUT_ROOT) / f"Step4D_RCJD_PINN_Forward_Stage1Train_{now_stamp()}")
    print(f"[Step4D-Stage1] Output directory: {out_dir}")
    print(f"[Step4D-Stage1] Device: {device}")
    save_json(config_to_dict(cfg), out_dir / "config.json")
    snapshot_source_code(out_dir)

    t0 = time.time()
    splits = load_all_splits(cfg)
    x_mean, x_std, y_mean, y_std = compute_standardizers(splits["train"]["X"], splits["train"]["Y_m"])
    save_json({
        "X_train_shape": list(splits["train"]["X"].shape),
        "Y_train_shape": list(splits["train"]["Y_m"].shape),
        "X_val_shape": list(splits["val"]["X"].shape),
        "Y_val_shape": list(splits["val"]["Y_m"].shape),
        "X_test_shape": list(splits["test"]["X"].shape),
        "Y_test_shape": list(splits["test"]["Y_m"].shape),
        "x_key": cfg.X_KEY,
        "y_key": cfg.Y_KEY,
        "rcjd_best_start_epoch": rcjd_best_start_epoch(cfg),
    }, out_dir / "data_info.json")

    train_loader, val_loader, _ = make_dataloaders(splits, x_mean, x_std, y_mean, y_std, cfg)

    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    print(f"[Step4D-Stage1] Model parameters: {n_params}")
    print(f"[Step4D-Stage1] ring_residual_scale = {float(cfg.RING_RESIDUAL_SCALE):.8e}")
    print(f"[Step4D-Stage1] RCJD best selection starts at epoch {rcjd_best_start_epoch(cfg)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.LEARNING_RATE), weight_decay=float(cfg.WEIGHT_DECAY))
    scheduler = None
    if bool(getattr(cfg, "USE_LR_SCHEDULER", True)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(cfg.LR_SCHEDULER_FACTOR), patience=int(cfg.LR_SCHEDULER_PATIENCE), min_lr=float(cfg.LR_SCHEDULER_MIN_LR)
        )

    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    lambda_ring = 0.0
    lambda_joint = 0.0
    best_data_val = float("inf")
    best_data_epoch = -1
    best_rcjd_val = float("inf")
    best_rcjd_epoch = -1
    best_rcjd_lambda_ring = float("nan")
    best_rcjd_lambda_joint = float("nan")
    epochs_no_improve = 0
    train_rows = []
    lambda_rows = []
    rcjd_start = rcjd_best_start_epoch(cfg)

    max_epochs = int(cfg.MAX_EPOCHS)
    for epoch in range(1, max_epochs + 1):
        phase = current_phase(epoch, cfg)
        if phase == "data_pretrain":
            lambda_ring = 0.0
            lambda_joint = 0.0
        elif epoch == int(cfg.DATA_PRETRAIN_EPOCHS) + 1:
            lambda_ring = float(cfg.LAMBDA_RING_INIT)
            lambda_joint = float(cfg.LAMBDA_JOINT_INIT)

        train_loss = train_or_eval_one_epoch(model, train_loader, optimizer, device, cfg, y_mean_t, y_std_t, lambda_ring, lambda_joint, train=True)
        val_loss = train_or_eval_one_epoch(model, val_loader, None, device, cfg, y_mean_t, y_std_t, lambda_ring, lambda_joint, train=False)

        # Update adaptive lambdas after validation to use stable full-epoch statistics.
        if phase == "adaptive_rcjd" and (epoch % int(cfg.ADAPT_UPDATE_INTERVAL) == 0):
            lambda_ring = update_lambda(lambda_ring, train_loss["data"], train_loss["ring"], float(cfg.LAMBDA_RING_MIN), float(cfg.LAMBDA_RING_MAX), cfg)
            lambda_joint = update_lambda(lambda_joint, train_loss["data"], train_loss["joint"], float(cfg.LAMBDA_JOINT_MIN), float(cfg.LAMBDA_JOINT_MAX), cfg)

        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss["total"])

        row = {
            "epoch": epoch,
            "phase": phase,
            "lr": lr,
            "lambda_ring": lambda_ring,
            "lambda_joint": lambda_joint,
            "eligible_for_rcjd_best": epoch >= rcjd_start,
        }
        for prefix, d in [("train", train_loss), ("val", val_loss)]:
            for k, v in d.items():
                row[f"{prefix}_{k}"] = v
        train_rows.append(row)
        lambda_rows.append({"epoch": epoch, "phase": phase, "lambda_ring": lambda_ring, "lambda_joint": lambda_joint, "eligible_for_rcjd_best": epoch >= rcjd_start})

        # Save the best data-only checkpoint for diagnostics only.
        if bool(getattr(cfg, "SAVE_BEST_DATA_MODEL", True)) and phase == "data_pretrain":
            if val_loss["data"] < best_data_val - float(cfg.MIN_DELTA):
                best_data_val = val_loss["data"]
                best_data_epoch = epoch
                save_checkpoint(
                    out_dir / "best_data_model.pt", model, optimizer, epoch, cfg, x_mean, x_std, y_mean, y_std,
                    lambda_ring, lambda_joint, best_data_val,
                    extra={"n_params": n_params, "phase": phase, "checkpoint_type": "best_data_model", "selection_metric": "val_data"},
                )

        # Save the final reported RCJD checkpoint only after the RCJD-constrained phase starts.
        rcjd_candidate = epoch >= rcjd_start
        improved_rcjd = False
        if bool(getattr(cfg, "SAVE_BEST_RCJD_MODEL", True)) and rcjd_candidate:
            if val_loss["total"] < best_rcjd_val - float(cfg.MIN_DELTA):
                best_rcjd_val = val_loss["total"]
                best_rcjd_epoch = epoch
                best_rcjd_lambda_ring = float(lambda_ring)
                best_rcjd_lambda_joint = float(lambda_joint)
                improved_rcjd = True
                save_checkpoint(
                    out_dir / "best_rcjd_model.pt", model, optimizer, epoch, cfg, x_mean, x_std, y_mean, y_std,
                    lambda_ring, lambda_joint, best_rcjd_val,
                    extra={
                        "n_params": n_params,
                        "phase": phase,
                        "checkpoint_type": "best_rcjd_model",
                        "selection_metric": "val_total_after_rcjd_start",
                        "rcjd_best_start_epoch": rcjd_start,
                    },
                )
                shutil.copy2(out_dir / "best_rcjd_model.pt", out_dir / "best_model.pt")

        if rcjd_candidate:
            if improved_rcjd:
                epochs_no_improve = 0
            elif epoch >= int(cfg.EARLY_STOP_START_EPOCH):
                epochs_no_improve += 1

        if epoch % int(cfg.SAVE_LAST_EVERY) == 0:
            save_checkpoint(
                out_dir / "last_model.pt", model, optimizer, epoch, cfg, x_mean, x_std, y_mean, y_std,
                lambda_ring, lambda_joint, best_rcjd_val if np.isfinite(best_rcjd_val) else val_loss["total"],
                extra={"n_params": n_params, "phase": phase, "checkpoint_type": "last_model"},
            )

        if epoch == 1 or epoch % int(cfg.PRINT_INTERVAL) == 0:
            print(
                f"Epoch {epoch:04d}/{max_epochs} | {phase:13s} | "
                f"train data={train_loss['data']:.4e}, ring={train_loss['ring']:.4e}, joint={train_loss['joint']:.4e}, "
                f"val total={val_loss['total']:.4e}, val data={val_loss['data']:.4e}, "
                f"lambda_ring={lambda_ring:.4e}, lambda_joint={lambda_joint:.4e}, "
                f"rcjd_best_epoch={best_rcjd_epoch}, lr={lr:.2e}"
            )

        pd.DataFrame(train_rows).to_csv(out_dir / "train_log.csv", index=False)
        pd.DataFrame(lambda_rows).to_csv(out_dir / "lambda_history.csv", index=False)

        if epoch >= int(cfg.EARLY_STOP_START_EPOCH) and epochs_no_improve >= int(cfg.PATIENCE):
            print(f"[Step4D-Stage1] Early stopping at epoch {epoch}. Best RCJD epoch = {best_rcjd_epoch}.")
            break

    save_checkpoint(
        out_dir / "last_model.pt", model, optimizer, epoch, cfg, x_mean, x_std, y_mean, y_std,
        lambda_ring, lambda_joint, best_rcjd_val if np.isfinite(best_rcjd_val) else val_loss["total"],
        extra={"n_params": n_params, "phase": phase, "checkpoint_type": "last_model"},
    )

    if not (out_dir / "best_rcjd_model.pt").exists():
        print("[Step4D-Stage1-Warning] No best_rcjd_model.pt was saved. Stage2 will not have a formal RCJD checkpoint.")

    elapsed = time.time() - t0
    summary = {
        "model_name": cfg.MODEL_NAME,
        "version_tag": str(getattr(cfg, "VERSION_TAG", "V1.2_RCJDCheckpoint")),
        "stage": "Stage1Train",
        "output_dir": str(out_dir),
        "device": str(device),
        "n_params": n_params,
        "best_data_epoch": int(best_data_epoch),
        "best_data_val_data_loss": float(best_data_val),
        "best_rcjd_epoch": int(best_rcjd_epoch),
        "best_rcjd_val_total_loss": float(best_rcjd_val),
        "rcjd_best_start_epoch": int(rcjd_start),
        "lambda_ring_at_best_rcjd": float(best_rcjd_lambda_ring),
        "lambda_joint_at_best_rcjd": float(best_rcjd_lambda_joint),
        "lambda_ring_final": float(lambda_ring),
        "lambda_joint_final": float(lambda_joint),
        "ring_residual_scale": float(cfg.RING_RESIDUAL_SCALE),
        "checkpoint_for_stage2": "best_rcjd_model.pt" if (out_dir / "best_rcjd_model.pt").exists() else None,
        "training_time_sec": float(elapsed),
    }
    save_json(summary, out_dir / "stage1_train_summary.json")
    write_text(out_dir / "run_step4d_stage1_summary.txt", "\n".join([f"{k}: {v}" for k, v in summary.items()]))
    print("[Step4D-Stage1] Done.")
    print(f"[Step4D-Stage1] Best data-only epoch: {best_data_epoch}")
    print(f"[Step4D-Stage1] Best RCJD epoch: {best_rcjd_epoch}")
    print(f"[Step4D-Stage1] Best RCJD val total loss: {best_rcjd_val:.8e}")
    print(f"[Step4D-Stage1] Output directory: {out_dir}")


if __name__ == "__main__":
    main()
