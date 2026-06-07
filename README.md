# 🎯 Hyperparameter Tuning with Optuna + MLflow

![CI](https://github.com/jumma786/mlops-hyperparameter-tuning/actions/workflows/tuning.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.13-orange)
![Optuna](https://img.shields.io/badge/Optuna-3.6-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Part of the MLOps Portfolio Series** — Project 4 of 10  
> Bayesian hyperparameter search with Optuna inside MLflow parent/child run structure. Each trial logged as a child run. Best trial auto-registers to MLflow Model Registry if it beats the baseline.

---

## 📂 Project Resources

| Resource | Link |
|---|---|
| 🎯 Tuner | [src/tuning/tuner.py](src/tuning/tuner.py) |
| 📦 Data Loader | [src/data/loader.py](src/data/loader.py) |
| 🧪 Unit Tests | [tests/test_tuning.py](tests/test_tuning.py) |
| 🤖 CI/CD Workflow | [.github/workflows/tuning.yml](.github/workflows/tuning.yml) |
| 📋 Requirements | [requirements.txt](requirements.txt) |

---

## 🎯 What This Project Does

Solves the "default hyperparameter" problem — models from Project 1 used defaults:

1. **Runs Bayesian search** — Optuna TPE sampler finds optimal params faster than grid search
2. **Every trial = one MLflow child run** — full reproducibility, nothing lost
3. **Parent run summarises all trials** — best params, convergence metrics
4. **Auto-promotes if beats baseline** — AUC 0.8174 from Project 1 is the gate
5. **Tunes 3 models** — XGBoost, LightGBM, Random Forest

---

## 🔬 MLflow Experiment Structure

```
optuna-xgboost-30trials (parent run)
├── params: model, n_trials, baseline_auc
├── metrics: best_cv_auc, test_auc, test_f1, auc_vs_baseline
│
├── trial-000 (child run)
│   ├── params: n_estimators, max_depth, learning_rate, ...
│   └── metrics: cv_auc, cv_auc_std, train_time
├── trial-001 (child run)
├── ...
└── trial-029 (child run)
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/jumma786/mlops-hyperparameter-tuning.git
cd mlops-hyperparameter-tuning
pip install -r requirements.txt

# Run tests
make test

# Tune XGBoost (50 trials)
make tune-xgboost

# Tune LightGBM (50 trials)
make tune-lightgbm

# View all trials in MLflow UI
make mlflow-ui
# → Open http://localhost:5000
```

---

## 📈 Results

| Model | Trials | Best CV AUC | Test AUC | vs Baseline |
|---|---|---|---|---|
| XGBoost | 20 | 0.5213 | 0.5110 | -0.3064 |
| LightGBM | TBD | TBD | TBD | TBD |
| RandomForest | TBD | TBD | TBD | TBD |

> Note: Results on synthetic data. Run with real UCI dataset (`--data-path data/bank-additional-full.csv`) for production results comparable to Project 1 baseline AUC 0.8174.

---

## ⚙️ Configuration

```bash
python -m src.tuning.tuner \
  --model xgboost \
  --n-trials 100 \
  --data-path data/bank-additional-full.csv
```

| Argument | Default | Description |
|---|---|---|
| `--model` | xgboost | Model to tune: xgboost, lightgbm, randomforest |
| `--n-trials` | 30 | Number of Optuna trials |
| `--data-path` | None | Real UCI CSV path |
| `--random-state` | 42 | Random seed |

---

## 🔗 MLOps Portfolio Series

| # | Project | Repo | Status |
|---|---|---|---|
| 1 | Multi-Model Tournament Pipeline | [mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |
| 2 | Scheduled Retraining + DVC + MLflow | [mlops-retraining-pipeline](https://github.com/jumma786/mlops-retraining-pipeline) | ✅ |
| 3 | Feature Engineering as Versioned Artifact | [mlops-feature-pipeline](https://github.com/jumma786/mlops-feature-pipeline) | ✅ |
| **4** | **Hyperparameter Tuning with Optuna + MLflow** | [mlops-hyperparameter-tuning](https://github.com/jumma786/mlops-hyperparameter-tuning) | ✅ This repo |
| 5 | FastAPI + Docker + Cloud Run | mlops-model-serving | 🔜 |
| 6 | Feature Store with Feast + Redis | mlops-feature-store | 🔜 |
| 7 | Model Monitoring & Drift Detection | mlops-model-monitoring | 🔜 |
| 8 | A/B Testing Framework | mlops-ab-testing | 🔜 |
| 9 | Airflow Pipeline Orchestration | mlops-airflow-pipeline | 🔜 |
| 10 | Kubernetes ML Platform | mlops-k8s-platform | 🔜 |

---

## 📝 Key MLOps Concepts Demonstrated

- **Bayesian search** — Optuna TPE sampler vs random/grid search
- **Experiment tracking** — every trial logged as MLflow child run
- **Champion gate** — auto-promote only if beats Project 1 baseline
- **Multi-model tuning** — XGBoost, LightGBM, Random Forest
- **Reproducibility** — fixed random seeds, full param logging

---

## 👤 Author

**Jumma Mohammad Teli** — Data Analyst & ML Engineer  
📍 Birmingham, UK  
📧 [jummamohammad477@gmail.com](mailto:jummamohammad477@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786)

---

*Project 4 of 10 — MLOps Portfolio Series. Builds on Project 1 (Model Tournament) by adding systematic hyperparameter optimisation.*
