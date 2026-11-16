# -*- coding: utf-8 -*-
"""Step6V2B: expanded synthetic full-curve inversion benchmark.

This script increases the synthetic benchmark to stratified test cases and uses
r=8 as the main retained-mode setting for the journal-level Step6 validation.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import step6v2_config as cfg
from step6v2_common import (
    InversionContext, bound_penalty_channels, compute_improvement, compute_initial_mean_prediction,
    ensure_dir, evenly_spaced_obs_indices, field_error_metrics, get_sample_identifiers,
    load_latent_prior_npz, load_step3_normalization_stats, load_step3_split, load_step4d_surrogate,
    locate_latest_prior_npz, locate_step4d_checkpoint, make_context, now_stamp, plot_curve_comparison,
    plot_feature_comparison, plot_initial_vs_inverted_rmse, predict_from_z, regression_metrics_mm,
    run_inversion, save_csv, save_json, select_topjournal_cases, set_seed, setup_logger,
    snapshot_source_code, summarize_metric_df, write_text,
)


def main():
    parser = argparse.ArgumentParser(description="Official V2B inversion, with optional single case smoke test")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit cases for interface testing; omit for the 30-case benchmark")
    args = parser.parse_args()
    if args.max_cases is not None and not 1 <= args.max_cases <= cfg.N_EXPANDED_CASES:
        parser.error("--max-cases must be between 1 and N_EXPANDED_CASES")
    set_seed(cfg.SEED)
    out_dir = ensure_dir(Path(cfg.OUTPUT_ROOT) / f"Step6V2B_ExpandedBenchmark_{now_stamp()}")
    fig_dir = ensure_dir(out_dir / "figures")
    case_root = ensure_dir(out_dir / "case_results")
    logger = setup_logger(out_dir / "logs" / "run_step6v2b_expanded_benchmark_log.txt", name="Step6V2B")
    snapshot_source_code(out_dir)

    logger.info("=" * 90)
    logger.info("Step6V2B started: expanded synthetic full-curve inversion benchmark")
    logger.info("=" * 90)

    prior_path = locate_latest_prior_npz(cfg)
    prior, meta, quantiles = load_latent_prior_npz(prior_path)
    ckpt_path = locate_step4d_checkpoint(cfg)
    surrogate = load_step4d_surrogate(ckpt_path, cfg.DEVICE)
    step3_stats = load_step3_normalization_stats(cfg)
    test = load_step3_split(cfg.STEP3_TEST_NPZ)
    feature_names = test["feature_names_list"]
    x_node_m = np.asarray(test["x_node_m"], dtype=np.float64).reshape(-1)

    logger.info(f"Latent prior: {prior_path}")
    logger.info(f"Step4D checkpoint: {ckpt_path}")
    logger.info(f"Device: {surrogate.device}")

    y_init_mean_m, init_features_raw, _ = compute_initial_mean_prediction(cfg, prior, meta, quantiles, step3_stats, surrogate, x_node_m, cfg.MAIN_N_MODES)
    target_indices = select_topjournal_cases(
        test, y_init_mean_m, cfg.N_EXPANDED_CASES, cfg.EXPANDED_SELECTION, cfg.FULL_CURVE_EXCLUDE_BOUNDARY_NODES
    )
    if args.max_cases is not None:
        target_indices = target_indices[:args.max_cases]
        logger.info("SMOKE TEST ONLY: running %d of %d cases; not the full paper benchmark.",
                    args.max_cases, cfg.N_EXPANDED_CASES)
    obs_indices = evenly_spaced_obs_indices(
        len(x_node_m), len(x_node_m) - 2 * int(cfg.FULL_CURVE_EXCLUDE_BOUNDARY_NODES), int(cfg.FULL_CURVE_EXCLUDE_BOUNDARY_NODES)
    )

    rows, z_rows = [], []
    n_modes = int(cfg.MAIN_N_MODES)
    logger.info(f"Selected {len(target_indices)} stratified target cases; n_modes={n_modes}; obs_count={len(obs_indices)}")

    for case_no, local_idx in enumerate(target_indices, start=1):
        global_idx, sample_id = get_sample_identifiers(test, int(local_idx))
        logger.info("-" * 90)
        logger.info(f"Case {case_no}/{len(target_indices)} | local_idx={local_idx}, global_idx={global_idx}, sample_id={sample_id}")

        y_obs_m = np.asarray(test["w_m_raw"], dtype=np.float64)[local_idx, :]
        true_features_raw = np.asarray(test["features_raw"], dtype=np.float64)[local_idx, :, :]
        ctx = make_context(cfg, prior, meta, quantiles, step3_stats, surrogate, x_node_m, y_obs_m, obs_indices, n_modes=n_modes)

        init_pred_m = y_init_mean_m.copy()
        z_inv, opt_info = run_inversion(ctx, seed=int(cfg.SEED + 1000 + case_no), logger=logger)
        inv_pred_m, inv_features_raw, inv_channels = predict_from_z(z_inv, ctx)

        init_metrics = regression_metrics_mm(y_obs_m, init_pred_m, x_node_m, cfg.FULL_CURVE_EXCLUDE_BOUNDARY_NODES)
        inv_metrics = regression_metrics_mm(y_obs_m, inv_pred_m, x_node_m, cfg.FULL_CURVE_EXCLUDE_BOUNDARY_NODES)
        field_metrics = field_error_metrics(true_features_raw, inv_features_raw, feature_names)
        bpen = bound_penalty_channels(inv_channels, quantiles)

        row = {
            "case_no": case_no,
            "local_sample_index": int(local_idx),
            "global_sample_index": global_idx,
            "sample_id": sample_id,
            "n_modes": n_modes,
            "obs_count": int(len(obs_indices)),
            "initial_RMSE_mm": init_metrics["RMSE_mm"],
            "initial_MAE_mm": init_metrics["MAE_mm"],
            "initial_R2": init_metrics["R2"],
            "initial_peak_abs_error_mm": init_metrics["peak_abs_error_mm"],
            "inv_RMSE_mm": inv_metrics["RMSE_mm"],
            "inv_MAE_mm": inv_metrics["MAE_mm"],
            "inv_R2": inv_metrics["R2"],
            "inv_peak_abs_error_mm": inv_metrics["peak_abs_error_mm"],
            "inv_peak_position_error_m": inv_metrics.get("peak_position_error_m", np.nan),
            "improvement_percent": compute_improvement(init_metrics["RMSE_mm"], inv_metrics["RMSE_mm"]),
            "latent_norm": float(np.linalg.norm(z_inv)),
            "latent_sum_square": float(np.sum(z_inv ** 2)),
            "latent_mean_square": float(np.mean(z_inv ** 2)),
            "bound_penalty": float(bpen),
            "objective_final": float(opt_info.get("fun", np.nan)),
            "optimizer_success": bool(opt_info.get("success", False)),
            "optimizer_message": str(opt_info.get("message", "")),
            "optimizer_nfev": int(opt_info.get("nfev", -1)),
        }
        row.update({f"field_{k}": v for k, v in field_metrics.items()})
        rows.append(row)

        for j, zj in enumerate(z_inv, start=1):
            z_rows.append({"case_no": case_no, "global_sample_index": global_idx, "mode_id": j, "z_inv": float(zj)})

        case_dir = ensure_dir(case_root / f"case_{case_no:02d}_global_{global_idx:04d}")
        np.savez_compressed(
            case_dir / "step6v2b_case_result.npz",
            x_node_m=x_node_m,
            obs_indices=obs_indices,
            y_obs_m=y_obs_m,
            y_init_m=init_pred_m,
            y_inv_m=inv_pred_m,
            z_inv=z_inv,
            true_features_raw=true_features_raw,
            init_features_raw=init_features_raw,
            inv_features_raw=inv_features_raw,
            feature_names=np.array(feature_names, dtype=str),
            local_sample_index=np.array(local_idx),
            global_sample_index=np.array(global_idx),
        )
        save_json({"metrics": row, "optimizer": opt_info}, case_dir / "case_summary.json")

        if case_no <= int(cfg.SAVE_EXPANDED_CASE_FIGURES_MAX):
            plot_curve_comparison(x_node_m, y_obs_m, init_pred_m, inv_pred_m, case_dir,
                                  f"Fig_step6v2b_case_{case_no:02d}_curve", cfg,
                                  title=f"Expanded inversion case {case_no}", obs_indices=None, logger=logger)
        if case_no <= int(cfg.SAVE_EXPANDED_FIELD_FIGURES_MAX):
            plot_feature_comparison(x_node_m, true_features_raw, inv_features_raw, feature_names,
                                    ["E_eff_MPa", "k", "B2", "B0"], case_dir,
                                    f"Fig_step6v2b_case_{case_no:02d}_field", cfg, logger=logger)
        logger.info(f"Case {case_no}: initial RMSE={init_metrics['RMSE_mm']:.6g} mm, inverted RMSE={inv_metrics['RMSE_mm']:.6g} mm, improvement={row['improvement_percent']:.2f}%")

    metrics_df = pd.DataFrame(rows)
    z_df = pd.DataFrame(z_rows)
    save_csv(metrics_df, out_dir / "step6v2b_expanded_inversion_metrics.csv", cfg.CSV_ENCODING)
    save_csv(z_df, out_dir / "step6v2b_inverted_latent_variables.csv", cfg.CSV_ENCODING)

    plot_initial_vs_inverted_rmse(metrics_df, fig_dir, "Fig_step6v2b_initial_vs_inverted_rmse", cfg, logger)

    summary = {
        "stage": "Step6V2B_ExpandedBenchmark",
        "output_dir": str(out_dir),
        "prior_path": str(prior_path),
        "step4d_checkpoint": str(ckpt_path),
        "n_cases": int(len(target_indices)),
        "n_modes": n_modes,
        "case_selection": cfg.EXPANDED_SELECTION,
        "obs_count": int(len(obs_indices)),
    }
    summary.update(summarize_metric_df(metrics_df))
    save_json(summary, out_dir / "step6v2b_expanded_inversion_summary.json")
    write_text(out_dir / "run_step6v2b_final_summary.txt", "\n".join([f"{k}: {v}" for k, v in summary.items()]))
    logger.info("Step6V2B completed successfully.")
    logger.info(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
