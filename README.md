# Automated Pulsar Candidate Selection on HTRU2

Leakage-free pulsar candidate classifier: RobustScaler -> SMOTE (training data only) -> XGBoost / RBF-SVM,
compared with logistic regression, decision tree and random forest baselines on the public HTRU2 benchmark
(17,898 candidates, 1,639 pulsars).

## Files
| File | Purpose |
|---|---|
| `pulsar_pipeline.py` | Main pipeline: scaling, SMOTE, XGBoost, SVM, ensemble, metrics and plots |
| `experiments.py` | Baselines, out-of-fold threshold selection, 12 functional tests, report figures |
| `data/HTRU_2.csv` | HTRU2 dataset (R. J. Lyon, 2015); also on the UCI repository |
| `results/` | Generated figures and metrics |

## Run
```bash
pip install -r requirements.txt
python pulsar_pipeline.py
python experiments.py
```
Random seed is fixed to 42, so results are reproducible.

## Headline results (3,580-candidate hold-out set)
XGBoost + SMOTE: recall 0.909, precision 0.754, ROC-AUC 0.971, about 3 s per million candidates (single CPU core).

## Citation
Dataset: R. J. Lyon, "HTRU2", figshare, 2015, doi:10.6084/m9.figshare.3080389.v1
