# Official Test Set Locked Policy

## ISIC 2019 Official Test Evaluation Policy

### 1. Strict Isolation Protocol
* The official ISIC 2019 test set contains **8,238 unlabelled/locked images** located under `dataset/raw/test/images/`.
* The test set is **never** used during:
  - Model training or backpropagation
  - Hyperparameter tuning (learning rate, weight decay, batch size)
  - Data augmentation strategy decisions
  - Calibration of temperature scaling parameter $T^*$
  - Selection of Energy / Mahalanobis Out-Of-Distribution (OOD) rejection thresholds
  - Model architecture selection or early stopping

### 2. Final Evaluation Execution
* The test set manifest is stored with `locked: true` at `ml/reports/test_manifest.csv`.
* Model checkpoints may only be evaluated on the locked test set **once**, strictly at the conclusion of all model iterations, to produce final unbiased benchmark metrics for the research paper.
* Any training script that attempts to ingest `test_manifest.csv` as training data must immediately raise a `PermissionError` and terminate.
