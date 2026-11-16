# Data and code availability

Code for RCJD-PINN training and evaluation, the equivalent coefficient inversion, and the empirical posterior update is included in this repository. The matching numerical datasets and selected trained checkpoint are provided as the `v1.0.0` GitHub Release asset `R1-9_Data_and_Checkpoint_v1.0.0.zip`.

The distributed data include training, validation, and test datasets, the retained low dimensional coefficient prior, reference model prediction curves, the selected model checkpoint, and the frozen empirical response library used for the reported posterior calculation.

The general LTSBSM solver is not part of this release. These materials enable evaluation and replay from the reference datasets but do not include a complete source implementation for creating the reference responses from first principles. The original direct audit and solver efficiency scripts are also outside the released subset. The repository is limited to the documented computational scope.

The authors should verify the public repository URL, release tag, attached asset, and licenses before inserting a final availability statement into the manuscript or review response.
