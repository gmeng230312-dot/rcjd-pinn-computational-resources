# -*- coding: utf-8 -*-
"""
Step7 V4B: compute compact posterior reliability and curve-feature updating.

V4 keeps the prior-quantile threshold strategy and adds two practical fixes:
1) Pf display columns use the empirical ESS resolution instead of reporting
   misleading ultra-small probabilities such as 1e-50.
2) The convergence representative case is selected automatically from the
   informative opening/faulting/w threshold candidates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from step7v4_utils import (
    effective_sample_size,
    elapsed,
    load_json,
    normalize_weights,
    reliability_index_from_pf,
    response_features,
    safe_np_load,
    setup_logger,
    summarize_feature,
    system_info,
    timer,
    weighted_mean,
    weighted_quantile,
    weighted_std,
    write_csv,
)

RESPONSES = ["w", "opening", "faulting"]


def locate_ready_npz(out_dir: Path) -> Path:
    # Release layout: the frozen input dataset is bundled separately from results.
    public_data = Path(__file__).resolve().parent / "data" / "step7_ready_nonparametric_posterior_case_median.npz"
    if public_data.exists():
        return public_data
    candidates = sorted((out_dir / "step7_ready_inputs").glob("step7_ready_nonparametric_posterior*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No Step7-ready npz found under {out_dir / 'step7_ready_inputs'}")
    return candidates[-1]


def get_response_x(data: Dict, response: str) -> np.ndarray:
    return np.asarray(data["x_node_m" if response == "w" else "x_joint_m"], dtype=float)


def get_reference_peak(data: Dict, features: Dict, mode: str, response: str) -> float:
    if mode == "posterior_map_candidate":
        key = f"{response}_map_candidate"
        if key in data:
            y = np.asarray(data[key], dtype=float)[None, :]
            x = get_response_x(data, response)
            return float(response_features(y, x, response)["peak_abs_m"][0])
    if mode == "prior_median":
        return float(np.median(features[response]["prior"]["peak_abs_m"]))
    if mode == "prior_mean":
        return float(np.mean(features[response]["prior"]["peak_abs_m"]))
    if mode == "posterior_weighted_median":
        return float(weighted_quantile(features[response]["prior"]["peak_abs_m"], [0.5], data["posterior_weights"])[0])
    raise ValueError(f"Unsupported reference peak mode: {mode}")


def prepare_features(data: Dict, config: Dict) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    frac = float(config["reliability"].get("affected_fraction_of_peak", 0.5))
    x_node = np.asarray(data["x_node_m"], dtype=float)
    x_joint = np.asarray(data["x_joint_m"], dtype=float)
    return {
        "w": {
            "prior": response_features(np.asarray(data["w_prior"], dtype=float), x_node, "w", frac),
            "post_resampled": response_features(np.asarray(data["w_post_resampled"], dtype=float), x_node, "w", frac),
        },
        "opening": {
            "prior": response_features(np.asarray(data["opening_prior"], dtype=float), x_joint, "opening", frac),
            "post_resampled": response_features(np.asarray(data["opening_post_resampled"], dtype=float), x_joint, "opening", frac),
        },
        "faulting": {
            "prior": response_features(np.asarray(data["faulting_prior"], dtype=float), x_joint, "faulting", frac),
            "post_resampled": response_features(np.asarray(data["faulting_post_resampled"], dtype=float), x_joint, "faulting", frac),
        },
    }



def format_pf_for_display(pf: float, n_eff: float) -> Tuple[str, int, float]:
    """Return a display string, below-resolution flag, and empirical resolution.

    Raw weighted probabilities can be extremely small when only a few weighted
    samples exceed a threshold. For an empirical posterior ensemble, values below
    1/ESS should not be interpreted as rare-event probabilities. They are kept
    numerically for reproducibility, but displayed as below the empirical
    resolution.
    """
    pf = float(pf)
    if n_eff is None or n_eff <= 0 or not np.isfinite(n_eff):
        resolution = float("nan")
    else:
        resolution = 1.0 / float(n_eff)
    if np.isfinite(resolution) and pf < resolution:
        return f"<1/ESS ({resolution:.3e})", 1, resolution
    return f"{pf:.6g}", 0, resolution


def pf_beta_row(response: str, threshold_type: str, threshold_m: float, peak: np.ndarray, post_w: np.ndarray, n_prior: int, ess_post: float, extra: Dict) -> Dict:
    pf_prior = float(np.mean(peak > threshold_m))
    pf_post = float(np.sum(post_w * (peak > threshold_m)))
    beta_prior, pf_prior_clip, clip_prior = reliability_index_from_pf(pf_prior, n_eff=n_prior)
    beta_post, pf_post_clip, clip_post = reliability_index_from_pf(pf_post, n_eff=ess_post)
    pf_prior_display, prior_below_res, prior_resolution = format_pf_for_display(pf_prior, n_prior)
    pf_post_display, post_below_res, post_resolution = format_pf_for_display(pf_post, ess_post)
    return {
        "response": response,
        "threshold_type": threshold_type,
        "threshold_m": float(threshold_m),
        "threshold_mm": float(threshold_m * 1000.0),
        "pf_prior": pf_prior,
        "pf_prior_display": pf_prior_display,
        "pf_prior_below_empirical_resolution": prior_below_res,
        "pf_prior_empirical_resolution": prior_resolution,
        "beta_prior": beta_prior,
        "pf_prior_clipped_for_beta": pf_prior_clip,
        "beta_prior_pf_clipped": int(clip_prior),
        "pf_posterior": pf_post,
        "pf_posterior_display": pf_post_display,
        "pf_posterior_below_empirical_resolution": post_below_res,
        "pf_posterior_empirical_resolution": post_resolution,
        "beta_posterior": beta_post,
        "pf_posterior_clipped_for_beta": pf_post_clip,
        "beta_posterior_pf_clipped": int(clip_post),
        "posterior_ess": ess_post,
        "posterior_ess_fraction": ess_post / n_prior if n_prior > 0 else float("nan"),
        "posterior_minus_prior_pf": pf_post - pf_prior,
        "pf_updating_ratio_reduction": (pf_prior - pf_post) / pf_prior if pf_prior > 0 else float("nan"),
        **extra,
    }


def compute_reliability_tables(data: Dict, features: Dict, config: Dict, out_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    rel_cfg = config["reliability"]
    post_w = normalize_weights(np.asarray(data["posterior_weights"], dtype=float))
    n_prior = len(post_w)
    ess_post = effective_sample_size(post_w)
    rows: List[Dict] = []
    thresholds_rows: List[Dict] = []

    # Primary thresholds: prior quantiles, compact and informative.
    q_list = list(rel_cfg.get("primary_prior_quantiles", [0.80, 0.90, 0.95]))
    for resp in RESPONSES:
        peak = np.asarray(features[resp]["prior"]["peak_abs_m"], dtype=float)
        for q in q_list:
            thr = float(np.quantile(peak, q))
            extra = {
                "threshold_family": "primary",
                "prior_quantile": float(q),
                "expected_prior_pf_by_definition": float(1.0 - q),
                "gamma": float("nan"),
                "reference_peak_m": float("nan"),
                "threshold_reference_mode": "prior_quantile",
            }
            rows.append(pf_beta_row(resp, "prior_quantile", thr, peak, post_w, n_prior, ess_post, extra))
            thresholds_rows.append({"response": resp, "threshold_type": "prior_quantile", "prior_quantile": float(q), "threshold_m": thr, "threshold_mm": thr * 1000.0})

    # Optional supplementary relative-to-MAP thresholds. Disabled by default for compactness.
    if bool(rel_cfg.get("include_relative_to_map_supplement", False)):
        gammas = list(rel_cfg.get("relative_threshold_gammas", [1.10, 1.25, 1.50, 1.75, 2.00]))
        ref_mode = rel_cfg.get("threshold_reference_mode", "posterior_map_candidate")
        for resp in RESPONSES:
            peak = np.asarray(features[resp]["prior"]["peak_abs_m"], dtype=float)
            ref = get_reference_peak(data, features, ref_mode, resp)
            for gamma in gammas:
                thr = float(gamma * ref)
                extra = {
                    "threshold_family": "supplementary",
                    "prior_quantile": float("nan"),
                    "expected_prior_pf_by_definition": float("nan"),
                    "gamma": float(gamma),
                    "reference_peak_m": ref,
                    "threshold_reference_mode": ref_mode,
                }
                rows.append(pf_beta_row(resp, "relative_to_reference_peak", thr, peak, post_w, n_prior, ess_post, extra))
                thresholds_rows.append({"response": resp, "threshold_type": "relative_to_reference_peak", "gamma": float(gamma), "threshold_m": thr, "threshold_mm": thr * 1000.0})

    # Optional absolute thresholds for safety reserve checks only.
    for resp, values in rel_cfg.get("safety_reserve_thresholds_m", {}).items():
        if resp not in features:
            continue
        peak = np.asarray(features[resp]["prior"]["peak_abs_m"], dtype=float)
        for thr in values:
            thr = float(thr)
            extra = {
                "threshold_family": "safety_reserve_check",
                "prior_quantile": float("nan"),
                "expected_prior_pf_by_definition": float("nan"),
                "gamma": float("nan"),
                "reference_peak_m": float("nan"),
                "threshold_reference_mode": "absolute_safety_reserve",
            }
            rows.append(pf_beta_row(resp, "absolute_safety_reserve", thr, peak, post_w, n_prior, ess_post, extra))
            thresholds_rows.append({"response": resp, "threshold_type": "absolute_safety_reserve", "threshold_m": thr, "threshold_mm": thr * 1000.0})

    # Back-compatible filename plus V4-specific filename.
    write_csv(out_dir / "tables" / "step7v4_peak_reliability_table.csv", rows)
    write_csv(out_dir / "tables" / "step7v4_thresholds_used.csv", thresholds_rows)
    return rows, thresholds_rows


def compute_curve_feature_table(data: Dict, features: Dict, config: Dict, out_dir: Path) -> List[Dict]:
    q_list = list(config["reliability"].get("summary_quantiles", [0.05, 0.5, 0.95]))
    post_w = normalize_weights(np.asarray(data["posterior_weights"], dtype=float))
    rows = []
    for resp in RESPONSES:
        for fname, vals in features[resp]["prior"].items():
            prior_summary = summarize_feature("prior", vals, None, q_list)
            post_summary = summarize_feature("posterior", vals, post_w, q_list)
            prior_std = prior_summary["prior_std"]
            post_std = post_summary["posterior_std"]
            rows.append({
                "response": resp,
                "feature": fname,
                "prior_mean": prior_summary["prior_mean"],
                "prior_std": prior_std,
                "prior_q05": prior_summary.get("prior_q05", float("nan")),
                "prior_q50": prior_summary.get("prior_q50", float("nan")),
                "prior_q95": prior_summary.get("prior_q95", float("nan")),
                "posterior_mean": post_summary["posterior_mean"],
                "posterior_std": post_std,
                "posterior_q05": post_summary.get("posterior_q05", float("nan")),
                "posterior_q50": post_summary.get("posterior_q50", float("nan")),
                "posterior_q95": post_summary.get("posterior_q95", float("nan")),
                "mean_change_ratio": (post_summary["posterior_mean"] - prior_summary["prior_mean"]) / prior_summary["prior_mean"] if abs(prior_summary["prior_mean"]) > 0 else float("nan"),
                "std_reduction_ratio": 1.0 - post_std / prior_std if prior_std > 0 else float("nan"),
            })
    write_csv(out_dir / "tables" / "step7v4_curve_feature_update_table.csv", rows)
    return rows


def compute_exceedance_curves(data: Dict, features: Dict, config: Dict, out_dir: Path) -> None:
    rel_cfg = config["reliability"]
    n_pts = int(rel_cfg.get("exceedance_curve_points", 300))
    qlo, qhi = rel_cfg.get("exceedance_threshold_quantile_range", [0.001, 0.999])
    post_w = normalize_weights(np.asarray(data["posterior_weights"], dtype=float))
    q_marks = list(rel_cfg.get("primary_prior_quantiles", [0.80, 0.90, 0.95]))
    for resp in RESPONSES:
        peak = np.asarray(features[resp]["prior"]["peak_abs_m"], dtype=float)
        lo_prior = float(np.quantile(peak, qlo))
        hi_prior = float(np.quantile(peak, qhi))
        lo_post = float(weighted_quantile(peak, [qlo], post_w)[0])
        hi_post = float(weighted_quantile(peak, [qhi], post_w)[0])
        lo = min(lo_prior, lo_post)
        hi = max(hi_prior, hi_post)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(peak)), float(np.max(peak))
        pad = 0.03 * max(hi - lo, 1e-12)
        thresholds = np.linspace(max(0.0, lo - pad), hi + pad, n_pts)
        quantile_thresholds = {f"prior_q{int(q*100):02d}_threshold_m": float(np.quantile(peak, q)) for q in q_marks}
        rows = []
        for t in thresholds:
            row = {
                "response": resp,
                "threshold_m": float(t),
                "threshold_mm": float(t * 1000.0),
                "poe_prior": float(np.mean(peak > t)),
                "poe_posterior": float(np.sum(post_w * (peak > t))),
            }
            row.update(quantile_thresholds)
            rows.append(row)
        write_csv(out_dir / "exceedance" / f"step7v4_exceedance_curve_{resp}.csv", rows)


def select_convergence_case(features: Dict, post_w: np.ndarray, config: Dict) -> Tuple[str, float, float, float]:
    """Select an informative representative convergence case.

    The selection avoids cases where posterior Pf is exactly zero, which makes
    the convergence curve visually uninformative. It prefers posterior Pf near
    the configured target, usually around 0.05.
    """
    rel_cfg = config["reliability"]
    q_list = list(rel_cfg.get("convergence_candidate_prior_quantiles", [0.90, 0.95, 0.80]))
    resp_pref = list(rel_cfg.get("convergence_response_preference", ["faulting", "opening", "w"]))
    target_pf = float(rel_cfg.get("convergence_target_posterior_pf", 0.05))
    pf_min, pf_max = rel_cfg.get("convergence_posterior_pf_range", [0.01, 0.20])
    candidates = []
    for pref_rank, resp in enumerate(resp_pref):
        if resp not in features:
            continue
        peak = np.asarray(features[resp]["prior"]["peak_abs_m"], dtype=float)
        for q in q_list:
            thr = float(np.quantile(peak, q))
            pf_post = float(np.sum(post_w * (peak > thr)))
            pf_prior = float(np.mean(peak > thr))
            in_range = float(pf_min <= pf_post <= pf_max)
            score = (0 if in_range else 1, pref_rank, abs(pf_post - target_pf), abs(float(q) - 0.90))
            candidates.append((score, resp, float(q), thr, pf_prior, pf_post))
    if not candidates:
        resp = str(rel_cfg.get("convergence_response", "w"))
        q = float(rel_cfg.get("representative_prior_quantile_for_convergence", 0.90))
        peak = np.asarray(features[resp]["prior"]["peak_abs_m"], dtype=float)
        thr = float(np.quantile(peak, q))
        return resp, q, thr, float(np.sum(post_w * (peak > thr)))
    candidates.sort(key=lambda x: x[0])
    _, resp, q, thr, _pf_prior, pf_post = candidates[0]
    return resp, q, thr, pf_post


def compute_convergence(data: Dict, features: Dict, config: Dict, out_dir: Path) -> List[Dict]:
    rel_cfg = config["reliability"]
    rng = np.random.default_rng(int(config["project"].get("random_seed", 20260627)) + 77)
    post_w = normalize_weights(np.asarray(data["posterior_weights"], dtype=float))

    if bool(rel_cfg.get("auto_select_convergence_case", True)):
        resp, q_rep, thr, pf_post_ref = select_convergence_case(features, post_w, config)
    else:
        resp = str(rel_cfg.get("convergence_response", "w"))
        peak_tmp = np.asarray(features[resp]["prior"]["peak_abs_m"], dtype=float)
        q_rep = float(rel_cfg.get("representative_prior_quantile_for_convergence", 0.90))
        thr = float(np.quantile(peak_tmp, q_rep))
        pf_post_ref = float(np.sum(post_w * (peak_tmp > thr)))

    peak = np.asarray(features[resp]["prior"]["peak_abs_m"], dtype=float)
    n_draws = int(rel_cfg.get("convergence_draws", 20000))
    checkpoints = [int(c) for c in rel_cfg.get("convergence_checkpoints", [100, 200, 500, 1000, 2000])]
    checkpoints = [c for c in checkpoints if c <= n_draws]
    if not checkpoints:
        checkpoints = [n_draws]

    idx_prior = rng.choice(np.arange(len(peak)), size=n_draws, replace=True)
    idx_post = rng.choice(np.arange(len(peak)), size=n_draws, replace=True, p=post_w)
    prior_draws = peak[idx_prior]
    post_draws = peak[idx_post]

    rows = []
    for c in checkpoints:
        pr = prior_draws[:c]
        po = post_draws[:c]
        rows.append({
            "n_draws": c,
            "response": f"{resp}_peak_abs",
            "threshold_type": "prior_quantile",
            "prior_quantile": q_rep,
            "threshold_m": thr,
            "threshold_mm": thr * 1000.0,
            "selected_by": "auto_informative_posterior_pf" if bool(rel_cfg.get("auto_select_convergence_case", True)) else "config",
            "posterior_pf_reference": pf_post_ref,
            "prior_running_mean_m": float(np.mean(pr)),
            "posterior_running_mean_m": float(np.mean(po)),
            "prior_running_q95_m": float(np.quantile(pr, 0.95)),
            "posterior_running_q95_m": float(np.quantile(po, 0.95)),
            "prior_running_pf": float(np.mean(pr > thr)),
            "posterior_running_pf": float(np.mean(po > thr)),
            "note": "Convergence draws are resampled from the empirical 2000-sample library. They check estimator stability, not new physical samples."
        })
    write_csv(out_dir / "convergence" / "step7v4_convergence_table.csv", rows)
    np.savez_compressed(
        out_dir / "convergence" / "step7v4_convergence_draws.npz",
        prior_draws=prior_draws,
        posterior_draws=post_draws,
        threshold_m=np.asarray(thr, dtype=float),
        threshold_type=np.asarray("prior_quantile"),
        prior_quantile=np.asarray(q_rep, dtype=float),
        checkpoints=np.asarray(checkpoints, dtype=np.int32),
        response=np.asarray(resp),
        posterior_pf_reference=np.asarray(pf_post_ref, dtype=float),
    )
    return rows


def compute_efficiency_summary(data: Dict, config: Dict, out_dir: Path, timings: Dict[str, float]) -> List[Dict]:
    n_lib = int(len(data["posterior_weights"]))
    n_post = int(len(data["posterior_resample_indices"])) if "posterior_resample_indices" in data else n_lib
    ess = float(np.asarray(data["posterior_ess"]).item())
    rows = [
        {"quantity": "prior_library_size", "value": n_lib, "unit": "samples", "note": "Frozen LTSBSM response library"},
        {"quantity": "posterior_resample_size", "value": n_post, "unit": "samples", "note": "Weighted posterior resampling"},
        {"quantity": "posterior_ess", "value": ess, "unit": "effective samples", "note": "1/sum(w_i^2)"},
        {"quantity": "posterior_ess_fraction", "value": ess / n_lib, "unit": "-", "note": "ESS divided by prior size"},
        {"quantity": "step7v4_compute_time", "value": timings.get("compute_s", float("nan")),
         "unit": "s", "note": "Timing of this statistical postprocessing run"},
    ]
    write_csv(out_dir / "tables" / "step7v4_efficiency_summary.csv", rows)
    return rows


def write_summary(out_dir: Path, data: Dict, config: Dict, reliability_rows: List[Dict]) -> None:
    p = out_dir / "summary" / "step7v4_final_summary.txt"
    with p.open("w", encoding="utf-8") as f:
        f.write("Step7 V4 Nonparametric Posterior Reliability Summary\n")
        f.write("=====================================================\n\n")
        f.write("Core logic:\n")
        f.write("  Step6V2B selected observation curve + Step3 2000-sample high-fidelity prior response library\n")
        f.write("  -> nonparametric posterior reweighting -> compact peak reliability + curve-feature updating.\n\n")
        f.write("V4 threshold strategy:\n")
        f.write("  Main reliability uses prior P80/P90/P95 thresholds, not overly large code-level limits.\n")
        f.write("  This follows the performance-based / monitoring-level threshold logic used in previous papers.\n")
        f.write("  Conventional absolute limits, if provided, are only safety-reserve checks.\n\n")
        f.write(f"Selected Step6 case: {np.asarray(data['selected_case_dir']).item()}\n")
        f.write(f"Selected global sample index: {int(np.asarray(data['selected_case_global_sample_index']).item())}\n")
        f.write(f"Posterior MAP global sample index: {int(np.asarray(data['posterior_map_global_sample_index']).item())}\n")
        f.write(f"sigma_y = {float(np.asarray(data['sigma_y_m']).item()):.8e} m\n")
        f.write(f"ESS = {float(np.asarray(data['posterior_ess']).item()):.3f}\n")
        f.write(f"ESS fraction = {float(np.asarray(data['posterior_ess_fraction']).item()):.6f}\n\n")
        f.write("Important caution:\n")
        f.write("  The 2000 samples are a high-fidelity empirical prior library. They support posterior updating and moderate-probability exceedance estimates.\n")
        f.write("  They should not be described as a 1e6-sample rare-event Monte Carlo analysis. Optional large-MC propagation must use the trained Step4D RCJD-PINN only as a forward operator.\n\n")
        f.write("Generated tables and figures:\n")
        f.write("  tables/step7v4_peak_reliability_table.csv\n")
        f.write("  tables/step7v4_curve_feature_update_table.csv\n")
        f.write("  tables/step7v4_efficiency_summary.csv\n")
        f.write("  convergence/step7v4_convergence_table.csv\n")
        f.write("  exceedance/step7v4_exceedance_curve_*.csv\n")
        f.write("  figures/Fig7_1 to Fig7_4 PNG/PDF\n\n")
        f.write("System info:\n")
        for k, v in system_info().items():
            f.write(f"  {k}: {v}\n")


def run_compute(config: Dict, out_dir: Path, logger) -> None:
    t0 = timer()
    ready_npz = locate_ready_npz(out_dir)
    logger.info("Loading Step7-ready inputs: %s", ready_npz)
    z = safe_np_load(ready_npz)
    data = {k: z[k] for k in z.keys()}

    logger.info("Prior library size: %d", len(data["posterior_weights"]))
    logger.info("Posterior ESS: %.3f", float(np.asarray(data["posterior_ess"]).item()))
    features = prepare_features(data, config)
    reliability_rows, _ = compute_reliability_tables(data, features, config, out_dir)
    compute_curve_feature_table(data, features, config, out_dir)
    compute_exceedance_curves(data, features, config, out_dir)
    compute_convergence(data, features, config, out_dir)
    timings = {"compute_s": elapsed(t0)}
    compute_efficiency_summary(data, config, out_dir, timings)

    save_dict = {}
    for resp in features:
        for group in features[resp]:
            for fname, val in features[resp][group].items():
                save_dict[f"{resp}_{group}_{fname}"] = val
    np.savez_compressed(out_dir / "diagnostics" / "step7v4_response_features.npz", **save_dict)
    write_summary(out_dir, data, config, reliability_rows)
    logger.info("Step7V4B compute finished in %.3f s", elapsed(t0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step7_v4_config.json")
    parser.add_argument("--output-dir", required=True, help="Existing Step7 V4 output directory containing step7_ready_inputs")
    args = parser.parse_args()
    config = load_json(args.config)
    out_dir = Path(args.output_dir)
    for child in ("logs", "tables", "summary", "convergence", "exceedance", "diagnostics", "figures"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir, "Step7V4B")
    run_compute(config, out_dir, logger)


if __name__ == "__main__":
    main()
