# RCJD-PINN: Numerical Implementation and Reference Data

This repository accompanies the manuscript on stochastic longitudinal responses of segmental shield tunnels under asymmetric thrust.

## Contents

- `01_RCJD_PINN`: formal RCJD-PINN network, training, and inference scripts.
- `02_Inversion`: low dimensional equivalent random coefficient inversion.
- `03_Posterior`: empirical posterior updating and exceedance probability calculations.
- `verification_reference`: archived reference table used by the numerical verification script.

The numerical datasets and trained checkpoint are distributed separately as the GitHub Release asset **`R1-9_Data_and_Checkpoint_v1.0.0.zip`** (Release tag `v1.0.0`). After downloading the asset, **extract it into the repository root** so that its `02_Inversion`, `03_Posterior`, `05_RCJD_Artifacts`, and `06_Dataset` folders merge with the corresponding code folders.

## Requirements

Python 3.9 was used for the author's Windows verification, with PyTorch 2.7.1+cu128. Install PyTorch for your hardware using its official installation instructions and install NumPy, SciPy, pandas, and Matplotlib. `requirements_minimal.txt` lists the Python modules but does not promise identical training weights across software and GPU versions. All distributed model and data files should be treated as trusted research files.

## Run the Numerical Checks (Windows CMD)

From the repository root with the environment activated:

```bat
python 00_VERIFY_RELEASE_DATA.py
00_RUN_CHECKS_VISIBLE.bat
01_RUN_POSTERIOR_VISIBLE.bat
03_RUN_INVERSION_FULL_VISIBLE.bat
04_VERIFY_ALL_VISIBLE.bat
```

`02_RUN_INVERSION_SMOKE_VISIBLE.bat` is an optional one-case interface test; it **does not** reproduce the full 30-case result. The full inversion runs use eight retained modes and 494 internal observation nodes.

The final gate checks the 300-sample checkpoint replay, nine main posterior table rows, and the complete 30-case inversion summary against saved numerical references. The checks establish the reported **replay and numerical result consistency**, not bitwise retraining reproducibility or reconstruction of the original full numerical reference database.

## Scope and scientific limitations

- The general LTSBSM reference solver is **not included**. Frozen numerical response and coefficient datasets are supplied to enable the downstream numerical analyses; this repository does not regenerate the underlying LTSBSM solutions from first principles.
- The complete R1-8 direct audit and R1-11 efficiency benchmarking environments are **not included**. Their internal standalone scripts require additional private reference solver components. No claim is made that the repository independently reproduces those complete audits.
- The Step7 posterior calculation begins from a frozen dataset containing 2000 reference samples. It is an empirical reweighting calculation rather than a new high fidelity Monte Carlo simulation.
- Joint location metadata in the distributed Step7 dataset use the 49 true internal joint positions from 1.2 to 58.8 m. The metadata adjustment does not constitute a recomputation of the stored reference responses. Joint peak location numbers may differ from legacy figure metadata.
- The 30 synthetic inversion cases lie within the ensemble used to construct the low dimensional coefficient prior. Response agreement must not be interpreted as unique recovery of the generating coefficient fields.

## Provenance and checksums

`DATA_MANIFEST_SHA256.csv` lists every binary dataset and checkpoint, its expected size, and SHA-256 digest. If a dataset checksum differs after download, reacquire the release asset before running the numerical verification.

## License and citation

## Usage

Copyright © 2026. Permission is granted to download and run these materials for noncommercial research purposes. All other rights reserved.
