# -*- coding: utf-8 -*-
"""Configuration for Ch4 Step6 TopJournal V2 curve-constrained inversion.

V2 strengthens Step6 for journal-level review by adding:
1. expanded synthetic benchmark with stratified test cases;
2. sparse/noisy observation robustness benchmark;
3. mode-count sensitivity and final mode selection;
4. prior-regularization ablation;
5. oracle latent projection check;
6. optional LTSBSM physical back-check;
7. final summary tables and paper-oriented figures.

Scope:
    The existing Step2B/Step3/Step4D chain fixed P0, M0 and alpha_s. Therefore
    V2 still does NOT invert construction eccentricity e or joint shear factor
    alpha_s. It identifies low-dimensional longitudinal equivalent random
    coefficient modes compatible with the trained Step4D RCJD-PINN surrogate.
"""
from __future__ import annotations

from pathlib import Path

# =============================================================================
# Project paths
# =============================================================================
CH4_ROOT = Path(__file__).resolve().parents[1]
STEP6_DIR = CH4_ROOT / "02_Inversion"
THIS_PACKAGE_DIR = STEP6_DIR
OUTPUT_ROOT = CH4_ROOT / "outputs" / "inversion"
STEP6_V1_OUTPUT_ROOT = STEP6_DIR / "data"
USE_LATEST_STEP6A_PRIOR = False
MANUAL_PRIOR_NPZ = STEP6_DIR / "data" / "step6_latent_prior_modes.npz"
# The original LTSBSM solver is intentionally not distributed.
STEP2_DIR = None
USE_LATEST_STEP2B = False
MANUAL_STEP2B_DIR = None
STEP2B_DATABASE_NAME = "step2b_random_coefficient_forward_database.npz"
STEP2B_SCRIPT_NAME = "step2b_ltsbsm_random_coefficient_forward.py"
STEP3_DATASET_DIR = CH4_ROOT / "06_Dataset"
STEP3_NORMALIZATION_JSON = STEP3_DATASET_DIR / "normalization_stats.json"
STEP3_TRAIN_NPZ = STEP3_DATASET_DIR / "pinn_node_dataset_train.npz"
STEP3_VAL_NPZ = STEP3_DATASET_DIR / "pinn_node_dataset_val.npz"
STEP3_TEST_NPZ = STEP3_DATASET_DIR / "pinn_node_dataset_test.npz"
STEP4D_DIR = CH4_ROOT / "01_RCJD_PINN"
STEP4D_OUTPUT_ROOT = CH4_ROOT / "05_RCJD_Artifacts"
USE_LATEST_STEP4D_TRAIN_DIR = False
MANUAL_STEP4D_TRAIN_DIR = CH4_ROOT / "05_RCJD_Artifacts"
STEP4D_CHECKPOINT_NAME = "best_rcjd_model.pt"

# =============================================================================
# Reproducibility and device
# =============================================================================
SEED = 20260626
DEVICE = "cuda_if_available"  # choices: cuda_if_available, cuda, cpu

# =============================================================================
# Feature definitions
# =============================================================================
NODE_FEATURE_NAMES = [
    "x_m",
    "E_eff_MPa",
    "nu_eff",
    "k",
    "Gs",
    "B2",
    "B0",
    "C2",
    "C0",
]

PRIOR_CHANNEL_NAMES = [
    "E_eff_MPa",
    "nu_eff",
    "k",
    "Gs",
    "B2",
    "B0",
    "C2",
    "C0",
]

STEP2B_CHANNEL_KEY_MAP = {
    "E_eff_MPa": "E_eff_all_MPa",
    "nu_eff": "nu_eff_all",
    "k": "k_all",
    "Gs": "Gs_all",
    "B2": "B2_all",
    "B0": "B0_all",
    "C2": "C2_all",
    "C0": "C0_all",
}

CHANNEL_TRANSFORMS = {
    "E_eff_MPa": "log",
    "nu_eff": "identity",
    "k": "log",
    "Gs": "log",
    "B2": "identity",
    "B0": "log",
    "C2": "identity",
    "C0": "log",
}

# =============================================================================
# Latent-prior and optimization settings
# =============================================================================
MODE_COUNTS = [3, 5, 8]
MAIN_N_MODES = 8
MAX_N_MODES_TO_USE = 12
LATENT_BOUND = 3.0

# Recommended regularized objective.
LAMBDA_PRIOR = 1.0e-3
LAMBDA_BOUND = 1.0e-2

# Ablation settings.
ABLATION_METHODS = [
    "prior_mean",
    "latent_no_prior",
    "proposed_prior_regularized",
]
ABLATION_LAMBDA_PRIOR = {
    "latent_no_prior": 0.0,
    "proposed_prior_regularized": LAMBDA_PRIOR,
}
ABLATION_LAMBDA_BOUND = {
    "latent_no_prior": 0.0,
    "proposed_prior_regularized": LAMBDA_BOUND,
}

# Derivative-free optimization for small-dimensional latent variables.
OPT_METHOD = "Powell"
MAXITER = 300
MAXFEV = 3000
XTOL = 1.0e-5
FTOL = 1.0e-8
N_RANDOM_STARTS = 4
RANDOM_START_SCALE = 0.50

# =============================================================================
# TopJournal V2 benchmark sizes
# =============================================================================
# Expanded synthetic benchmark. Increase to 50 if time is sufficient.
N_EXPANDED_CASES = 30
EXPANDED_SELECTION = "stratified_initial_rmse"  # choices: stratified_initial_rmse, stratified_peak, first
FULL_CURVE_EXCLUDE_BOUNDARY_NODES = 3
SAVE_EXPANDED_CASE_FIGURES_MAX = 8
SAVE_EXPANDED_FIELD_FIGURES_MAX = 4

# Sparse/noisy robustness. These are chosen to be informative but not too long.
N_SPARSE_TARGET_CASES = 10
SPARSE_OBS_COUNTS = [20, 50, 100]
NOISE_STD_MM_LIST = [0.0, 0.005, 0.010]
N_NOISE_REPEATS = 2
SAVE_SPARSE_CASE_FIGURES_MAX = 6

# Mode-count sensitivity.
N_MODE_SELECTION_CASES = 15
SAVE_MODE_CASE_FIGURES_MAX = 4

# Prior ablation.
N_PRIOR_ABLATION_CASES = 15
SAVE_PRIOR_ABLATION_CASE_FIGURES_MAX = 4

# Oracle projection and LTSBSM back-check.
N_ORACLE_CASES = 30  # usually follows V2B cases
N_LTSBSM_BACKCHECK_CASES = 8
SAVE_LTSBSM_BACKCHECK_CASE_FIGURES_MAX = 6

# =============================================================================
# Output and figure settings
# =============================================================================
SAVE_FIGURES = True
SAVE_FIG_FORMATS = ["png", "svg"]
FIG_DPI = 600
FIG_FONT_FAMILY = "Times New Roman"
FIG_WIDTH_CM = 10.0
FIG_HEIGHT_CM = 7.0

# Smaller journal-style summary figures.
SUMMARY_FIG_WIDTH_CM = 10.0
SUMMARY_FIG_HEIGHT_CM = 6.5

CSV_ENCODING = "utf-8-sig"
