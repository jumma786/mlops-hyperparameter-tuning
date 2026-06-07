"""
Hyperparameter Tuning with Optuna + MLflow
===========================================
Bayesian search inside MLflow parent/child run structure.
Each Optuna trial = one MLflow child run.
Best trial auto-registers to MLflow Model Registry.

Models tuned:
  - XGBoost (primary)
  - LightGBM (comparison)
  - Random Forest (baseline from Project 1)

Usage:
    python src/tuning/tuner.py --n-trials 50 --model xgboost
"""

import os
import sys
import time
import logging
import argparse

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import optuna
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loader import load_data

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "mlops-hyperparameter-tuning"
REGISTRY_NAME   = "TunedChampion"
BASELINE_AUC    = 0.8174   # Project 1 RandomForest baseline to beat


def get_xgboost_params(trial):
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight":  trial.suggest_int("scale_pos_weight", 5, 15),
        "eval_metric":       "logloss",
        "verbosity":         0,
        "random_state":      42,
    }


def get_lightgbm_params(trial):
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "class_weight":      "balanced",
        "verbose":           -1,
        "random_state":      42,
    }


def get_rf_params(trial):
    return {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
        "max_depth":       trial.suggest_int("max_depth", 5, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight":    "balanced",
        "random_state":    42,
        "n_jobs":          -1,
    }


def build_model(model_name: str, params: dict):
    if model_name == "xgboost":
        return xgb.XGBClassifier(**params)
    elif model_name == "lightgbm":
        return lgb.LGBMClassifier(**params)
    elif model_name == "randomforest":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def run_tuning(
    model_name: str = "xgboost",
    n_trials: int = 30,
    data_path: str = None,
    random_state: int = 42,
):
    """Run Optuna tuning with MLflow parent/child run structure."""

    # Load data
    X_train, X_test, y_train, y_test, features = load_data(
        data_path=data_path, random_state=random_state
    )

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_auc   = 0.0
    best_params = {}
    trial_results = []

    with mlflow.start_run(run_name=f"optuna-{model_name}-{n_trials}trials") as parent_run:
        mlflow.log_param("model",        model_name)
        mlflow.log_param("n_trials",     n_trials)
        mlflow.log_param("n_train",      len(X_train))
        mlflow.log_param("baseline_auc", BASELINE_AUC)
        mlflow.set_tag("tuning_method", "optuna-bayesian")

        def objective(trial):
            nonlocal best_auc, best_params

            # Get params for this trial
            if model_name == "xgboost":
                params = get_xgboost_params(trial)
            elif model_name == "lightgbm":
                params = get_lightgbm_params(trial)
            else:
                params = get_rf_params(trial)

            # Train and evaluate
            model = build_model(model_name, params)
            t0 = time.time()

            # Cross-validation AUC
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=3, scoring="roc_auc", n_jobs=-1
            )
            cv_auc = cv_scores.mean()
            train_time = round(time.time() - t0, 2)

            # Log each trial as child run
            with mlflow.start_run(
                run_name=f"trial-{trial.number:03d}",
                nested=True
            ) as child_run:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metric("cv_auc",        round(cv_auc, 4))
                mlflow.log_metric("cv_auc_std",    round(cv_scores.std(), 4))
                mlflow.log_metric("train_time",    train_time)
                mlflow.log_metric("trial_number",  trial.number)
                mlflow.set_tag("model", model_name)

            trial_results.append({
                "trial": trial.number,
                "cv_auc": cv_auc,
                "params": params
            })

            if cv_auc > best_auc:
                best_auc    = cv_auc
                best_params = params
                logger.info(f"Trial {trial.number:3d}: NEW BEST AUC={cv_auc:.4f}")
            else:
                logger.info(f"Trial {trial.number:3d}: AUC={cv_auc:.4f} (best={best_auc:.4f})")

            return cv_auc

        # Run Optuna study
        logger.info(f"Starting Optuna search: {n_trials} trials for {model_name}...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Evaluate best model on held-out test set
        best_model = build_model(model_name, study.best_params)
        best_model.fit(X_train, y_train)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        test_auc = round(roc_auc_score(y_test, y_prob), 4)
        test_f1  = round(f1_score(y_test, y_pred, zero_division=0), 4)

        # Log best results to parent run
        mlflow.log_metric("best_cv_auc",  round(study.best_value, 4))
        mlflow.log_metric("test_auc",     test_auc)
        mlflow.log_metric("test_f1",      test_f1)
        mlflow.log_metric("auc_vs_baseline", round(test_auc - BASELINE_AUC, 4))
        mlflow.log_params({f"best_{k}": str(v) for k, v in study.best_params.items()})
        mlflow.set_tag("best_trial", study.best_trial.number)

        # Log best model
        if model_name == "xgboost":
            mlflow.xgboost.log_model(best_model, "best_model")
        elif model_name == "lightgbm":
            mlflow.lightgbm.log_model(best_model, "best_model")
        else:
            mlflow.sklearn.log_model(best_model, "best_model")

        # Register if beats baseline
        beat_baseline = test_auc > BASELINE_AUC
        mlflow.set_tag("beat_baseline", str(beat_baseline))

        if beat_baseline:
            try:
                model_uri = f"runs:/{parent_run.info.run_id}/best_model"
                reg = mlflow.register_model(model_uri, REGISTRY_NAME)
                logger.info(f"Registered: {REGISTRY_NAME} v{reg.version}")
            except Exception as e:
                logger.warning(f"Registry: {e}")

        # Summary
        print("\n" + "="*60)
        print(f"OPTUNA TUNING COMPLETE — {model_name.upper()}")
        print("="*60)
        print(f"  Trials:           {n_trials}")
        print(f"  Best CV AUC:      {study.best_value:.4f}")
        print(f"  Test AUC:         {test_auc:.4f}")
        print(f"  Test F1:          {test_f1:.4f}")
        print(f"  Baseline AUC:     {BASELINE_AUC:.4f} (Project 1)")
        print(f"  vs Baseline:      {test_auc - BASELINE_AUC:+.4f}")
        print(f"  Beat baseline:    {'YES' if beat_baseline else 'NO'}")
        print(f"  Best trial:       #{study.best_trial.number}")
        print("="*60)
        print("\nBest params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study, test_auc, test_f1


def parse_args():
    p = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    p.add_argument("--model",        type=str,   default="xgboost",
                   choices=["xgboost","lightgbm","randomforest"])
    p.add_argument("--n-trials",     type=int,   default=30)
    p.add_argument("--data-path",    type=str,   default=None)
    p.add_argument("--random-state", type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tuning(
        model_name=args.model,
        n_trials=args.n_trials,
        data_path=args.data_path,
        random_state=args.random_state,
    )
