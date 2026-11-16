# -*- coding: utf-8 -*-
"""Configuration for Step4D RCJD-PINN forward modeling V1.2.

V1.2 fixes the checkpoint-selection issue in the first Step4D version:
the final reported model is selected only from the RCJD-constrained phase,
not from the data-only pretraining phase.

Step4D inherits the stable Step4C V1.2 pipeline and adds explicit
Ring-Continuity and Joint-Discontinuity (RCJD) losses.

Default dataset split:
    train = 1400, val = 300, test = 300
Data source:
    ../06_Dataset
"""
from __future__ import annotations

from pathlib import Path

# =============================================================================
# Project paths
# =============================================================================
CH4_ROOT = Path(__file__).resolve().parents[1]
STEP3_DATASET_DIR = CH4_ROOT / "06_Dataset"
STEP4D_DIR = CH4_ROOT / "01_RCJD_PINN"
OUTPUT_ROOT = CH4_ROOT / "outputs" / "rcjd"

TRAIN_NPZ = STEP3_DATASET_DIR / "pinn_node_dataset_train.npz"
VAL_NPZ = STEP3_DATASET_DIR / "pinn_node_dataset_val.npz"
TEST_NPZ = STEP3_DATASET_DIR / "pinn_node_dataset_test.npz"

# =============================================================================
# Dataset keys and dimensions
# =============================================================================
DATASET_MODE = "split_npz"
X_KEY = "features_norm"
Y_KEY = "w_m_raw"
X_COORD_KEY = "x_node_m"
GLOBAL_INDEX_KEYS = ["global_sample_indices", "sample_ids"]

# Raw coefficient keys used by RC residual.
B2_KEY = "B2_raw"
B0_KEY = "B0_raw"
C2_KEY = "C2_raw"
C0_KEY = "C0_raw"

EXPECTED_N_TRAIN = 1400
EXPECTED_N_VAL = 300
EXPECTED_N_TEST = 300
INPUT_DIM = 4500
OUTPUT_DIM = 500
N_POINTS = 500
N_FEATURE_CHANNELS = 9

Y_UNIT_NAME = "m"
Y_UNIT_SCALE_TO_MM = 1000.0

# =============================================================================
# Reproducibility and device
# =============================================================================
SEED = 20260624
DEVICE = "cuda_if_available"  # choices: cuda_if_available, cuda, cpu

# =============================================================================
# Model architecture. This matches Step4B/Step4C parameter count when
# HIDDEN_DIM=256 and NUM_HIDDEN_LAYERS=5, because the implementation uses
# 5 hidden representations, each followed by LayerNorm.
# =============================================================================
MODEL_NAME = "Step4D_RCJD_PINN_Forward_V1_2_RCJDCheckpoint"
VERSION_TAG = "V1.2_RCJDCheckpoint"
HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 5
DROPOUT = 0.0
ACTIVATION = "gelu"
USE_LAYER_NORM = True

# =============================================================================
# Training schedule
# =============================================================================
MAX_EPOCHS = 3000
DATA_PRETRAIN_EPOCHS = 800
ADAPTIVE_LAMBDA_WARMUP_EPOCHS = 100
FINETUNE_EPOCHS = 200
EARLY_STOP_START_EPOCH = 1000

BATCH_SIZE = 64
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
GRAD_CLIP_NORM = 1.0
PATIENCE = 800
MIN_DELTA = 1.0e-10
PRINT_INTERVAL = 20
SAVE_LAST_EVERY = 200

# V1.2 checkpoint selection. Data-only pretraining is used only for initialization.
# The final reported RCJD model is selected from epochs at or after
# DATA_PRETRAIN_EPOCHS + ADAPTIVE_LAMBDA_WARMUP_EPOCHS + 1.
RCJD_BEST_START_EPOCH = DATA_PRETRAIN_EPOCHS + ADAPTIVE_LAMBDA_WARMUP_EPOCHS + 1
STAGE2_DEFAULT_CHECKPOINT = "best_rcjd_model.pt"
SAVE_BEST_DATA_MODEL = True
SAVE_BEST_RCJD_MODEL = True
STRICT_STAGE2_RCJD_CHECKPOINT = False
RCJD_ARCHIVE_MIN_LAMBDA_RING = 1.0e-8
RCJD_ARCHIVE_MIN_LAMBDA_JOINT = 1.0e-8


# ReduceLROnPlateau scheduler.
USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 200
LR_SCHEDULER_MIN_LR = 1.0e-6

# =============================================================================
# Boundary handling, inherited from Step4C V1.2
# =============================================================================
USE_HARD_BOUNDARY_PROJECTION = True
BOUNDARY_INDICES = [0, 499]
BOUNDARY_VALUE_M = 0.0
RAW_BOUNDARY_LOSS_WEIGHT = 1.0e-2
PEAK_SEARCH_EXCLUDE_BOUNDARY_NODES = 3

# =============================================================================
# RCJD physics loss settings
# =============================================================================
N_RINGS = 50
N_POINTS_PER_RING = 10
LS_RING = 1.2
DL = LS_RING / (N_POINTS_PER_RING - 1)  # 0.133333333333...

# Ring-continuity residual: d4w + B2 d2w + B0 w = 0, evaluated only inside
# each ring, so the high-order stencil does not cross joints.
USE_RING_RESIDUAL = True
RING_INTERIOR_MARGIN = 2
RING_RESIDUAL_SCALE = 1.0e-4
USE_ADAPTIVE_RING_RESIDUAL_WEIGHTS = True
RING_RESIDUAL_WEIGHT_GAMMA = 1.0
RING_RESIDUAL_WEIGHT_MIN = 1.0
RING_RESIDUAL_WEIGHT_MAX = 5.0

# Joint-discontinuity residual. The first version uses data-derived joint
# targets for displacement jump and rotation jump, computed from the true
# response curves at the 49 ring joints. This keeps the first RCJD version
# stable while making the loss explicitly joint-aware.
USE_JOINT_DISCONTINUITY_LOSS = True
JOINT_DISP_SCALE_TO_MM = 1000.0
JOINT_ROT_SCALE = 1.0
JOINT_DISP_LOSS_WEIGHT = 1.0
JOINT_ROT_LOSS_WEIGHT = 1.0

# Adaptive physics weights. During data-only pretraining, both lambdas are zero.
# They are then updated using EMA after the adaptive phase begins.
LAMBDA_RING_INIT = 1.0
LAMBDA_JOINT_INIT = 1.0
LAMBDA_RING_MIN = 1.0e-4
LAMBDA_RING_MAX = 1.0e4
LAMBDA_JOINT_MIN = 1.0e-4
LAMBDA_JOINT_MAX = 1.0e4
ADAPT_UPDATE_INTERVAL = 10
ADAPT_EMA = 0.90
ADAPT_ALPHA = 0.50
ADAPT_EPS = 1.0e-16

# =============================================================================
# Output and figure settings
# =============================================================================
SAVE_FIGURES = True
SAVE_FIG_FORMATS = ["png", "svg"]
FIG_DPI = 600
FIG_FONT_FAMILY = "Times New Roman"
FIG_WIDTH_CM = 10.0
FIG_HEIGHT_CM = 7.0

# Optional reference metrics for Stage2 comparison plots.
REFERENCE_MODELS = {
    "Step4B": {"RMSE": 0.018005, "MAE": 0.007740, "R2": 0.973740},
    "Step4C V1.2": {"RMSE": 2.46349e-4, "MAE": 8.44748e-5, "R2": 0.999995084},
}
