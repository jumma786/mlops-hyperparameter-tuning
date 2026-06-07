.PHONY: install test tune-xgboost tune-lightgbm tune-rf mlflow-ui clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

tune-xgboost:
	python -m src.tuning.tuner --n-trials 50 --model xgboost

tune-lightgbm:
	python -m src.tuning.tuner --n-trials 50 --model lightgbm

tune-rf:
	python -m src.tuning.tuner --n-trials 50 --model randomforest

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

clean:
	rm -rf mlruns/ __pycache__
	find . -name "*.pyc" -delete

help:
	@echo "Commands:"
	@echo "  make install        Install dependencies"
	@echo "  make test           Run 10 unit tests"
	@echo "  make tune-xgboost   Run 50-trial XGBoost search"
	@echo "  make tune-lightgbm  Run 50-trial LightGBM search"
	@echo "  make tune-rf        Run 50-trial Random Forest search"
	@echo "  make mlflow-ui      Launch MLflow UI"
