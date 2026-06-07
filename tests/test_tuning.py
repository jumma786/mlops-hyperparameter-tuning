"""
Test suite for mlops-hyperparameter-tuning.
Run: pytest tests/ -v --cov=src
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_data
from src.tuning.tuner import get_xgboost_params, get_lightgbm_params, get_rf_params, build_model
import optuna


# ── Data Tests ────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_returns_correct_shapes(self):
        X_train, X_test, y_train, y_test, features = load_data(n_samples=500)
        assert len(X_train) + len(X_test) == 500
        assert len(features) > 0

    def test_no_duration_feature(self):
        X_train, X_test, y_train, y_test, features = load_data(n_samples=300)
        assert "duration" not in features

    def test_binary_target(self):
        X_train, X_test, y_train, y_test, _ = load_data(n_samples=300)
        assert set(y_train.unique()).issubset({0, 1})

    def test_no_object_columns(self):
        X_train, _, _, _, _ = load_data(n_samples=300)
        assert X_train.select_dtypes(include="object").shape[1] == 0


# ── Param Tests ───────────────────────────────────────────────────────────────

class TestParams:
    def test_xgboost_params_keys(self):
        trial = optuna.trial.create_trial(
            params={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
                    "reg_alpha": 0.01, "reg_lambda": 0.01, "scale_pos_weight": 8},
            distributions={
                "n_estimators": optuna.distributions.IntDistribution(100, 500),
                "max_depth": optuna.distributions.IntDistribution(3, 10),
                "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                "subsample": optuna.distributions.FloatDistribution(0.6, 1.0),
                "colsample_bytree": optuna.distributions.FloatDistribution(0.6, 1.0),
                "min_child_weight": optuna.distributions.IntDistribution(1, 10),
                "reg_alpha": optuna.distributions.FloatDistribution(1e-8, 1.0, log=True),
                "reg_lambda": optuna.distributions.FloatDistribution(1e-8, 1.0, log=True),
                "scale_pos_weight": optuna.distributions.IntDistribution(5, 15),
            },
            value=0.75,
        )
        params = get_xgboost_params(trial)
        for key in ["n_estimators", "max_depth", "learning_rate", "subsample"]:
            assert key in params

    def test_build_xgboost_model(self):
        import xgboost as xgb
        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
                  "reg_alpha": 0.01, "reg_lambda": 0.01, "scale_pos_weight": 8,
                  "eval_metric": "logloss", "verbosity": 0, "random_state": 42}
        model = build_model("xgboost", params)
        assert isinstance(model, xgb.XGBClassifier)

    def test_build_lightgbm_model(self):
        import lightgbm as lgb
        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 5,
                  "reg_alpha": 0.01, "reg_lambda": 0.01, "class_weight": "balanced",
                  "verbose": -1, "random_state": 42}
        model = build_model("lightgbm", params)
        assert isinstance(model, lgb.LGBMClassifier)

    def test_build_rf_model(self):
        from sklearn.ensemble import RandomForestClassifier
        params = {"n_estimators": 50, "max_depth": 5, "min_samples_leaf": 1,
                  "max_features": "sqrt", "class_weight": "balanced",
                  "random_state": 42, "n_jobs": -1}
        model = build_model("randomforest", params)
        assert isinstance(model, RandomForestClassifier)

    def test_model_fits_and_predicts(self):
        X_train, X_test, y_train, y_test, _ = load_data(n_samples=300)
        params = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
                  "reg_alpha": 0.01, "reg_lambda": 0.01, "scale_pos_weight": 8,
                  "eval_metric": "logloss", "verbosity": 0, "random_state": 42}
        model = build_model("xgboost", params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_model_predict_proba(self):
        X_train, X_test, y_train, y_test, _ = load_data(n_samples=300)
        params = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
                  "reg_alpha": 0.01, "reg_lambda": 0.01, "scale_pos_weight": 8,
                  "eval_metric": "logloss", "verbosity": 0, "random_state": 42}
        model = build_model("xgboost", params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
