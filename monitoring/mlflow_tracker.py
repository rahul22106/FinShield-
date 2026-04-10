import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score
)
from utils.logger import get_logger

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = 'mlruns'
EXPERIMENT_NAME     = 'FinShield-FraudDetection'


def setup_mlflow() -> None:
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment   : {EXPERIMENT_NAME}")


def log_run(
    model,
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    params: dict = None
) -> str:
    """
    Log a complete model run to MLflow:
    - Parameters
    - Metrics
    - Model artifact
    - Feature importance
    """
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # Log parameters
        if params:
            mlflow.log_params(params)
        else:
            # Auto extract params from model
            try:
                model_params = model.get_params()
                # Limit to avoid too many params
                safe_params = {
                    k: str(v) for k, v in model_params.items()
                    if v is not None
                }
                mlflow.log_params(safe_params)
            except Exception:
                pass

        # Log dataset info
        mlflow.log_params({
            'train_size':    len(X_train),
            'test_size':     len(X_test),
            'n_features':    X_train.shape[1],
            'fraud_rate_train': round(y_train.mean(), 4),
            'fraud_rate_test':  round(y_test.mean(), 4),
        })

        # Compute metrics
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'f1_score':        round(f1_score(y_test, preds), 4),
            'precision':       round(precision_score(y_test, preds), 4),
            'recall':          round(recall_score(y_test, preds), 4),
            'roc_auc':         round(roc_auc_score(y_test, proba), 4),
            'accuracy':        round(accuracy_score(y_test, preds), 4),
            'fraud_caught':    int(preds[y_test == 1].sum()),
            'fraud_missed':    int((y_test == 1).sum() - preds[y_test == 1].sum()),
            'false_positives': int(preds[y_test == 0].sum()),
        }

        mlflow.log_metrics(metrics)

        logger.info(f"Metrics logged:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")

        # Log feature importance if available
        actual_model = model
        if hasattr(model, 'estimators_'):
            for est_tuple in model.estimators_:
                est = est_tuple if not isinstance(
                    est_tuple, tuple) else est_tuple[1]
                if hasattr(est, 'feature_importances_'):
                    actual_model = est
                    break

        if hasattr(actual_model, 'feature_importances_'):
            importance = dict(zip(
                X_train.columns,
                actual_model.feature_importances_
            ))
            importance_sorted = dict(
                sorted(importance.items(),
                       key=lambda x: x[1],
                       reverse=True)
            )
            mlflow.log_params({
                f'fi_{k}': round(v, 4)
                for k, v in list(importance_sorted.items())[:5]
            })

        # Log model artifact
        try:
            if 'xgboost' in model_name.lower():
                mlflow.xgboost.log_model(model, 'model')
            else:
                mlflow.sklearn.log_model(model, 'model')
            logger.info("Model artifact logged to MLflow")
        except Exception as e:
            logger.error(f"Model logging failed: {e}")

        # Log plots as artifacts
        plots_dir = 'plots'
        if os.path.exists(plots_dir):
            mlflow.log_artifacts(plots_dir, artifact_path='plots')
            logger.info("Plots logged to MLflow")

        logger.info(f"MLflow run complete: {run_id}")
        return run_id


def log_all_models(
    models: dict,
    X_train,
    X_test,
    y_train,
    y_test
) -> dict:
    """Log all trained models to MLflow."""
    logger.info("Logging all models to MLflow...")
    run_ids = {}

    for name, model in models.items():
        logger.info(f"Logging {name}...")
        run_id = log_run(
            model=model,
            model_name=name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        run_ids[name] = run_id

    return run_ids


def get_best_run() -> dict:
    """
    Retrieve the best run from MLflow
    based on ROC AUC score.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if not experiment:
        logger.error("No experiment found")
        return {}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['metrics.roc_auc DESC'],
        max_results=1
    )

    if not runs:
        logger.info("No runs found")
        return {}

    best = runs[0]
    logger.info("-- Best Run ----------------------------")
    logger.info(f"  Run ID    : {best.info.run_id}")
    logger.info(f"  Model     : {best.info.run_name}")
    logger.info(f"  ROC AUC   : {best.data.metrics.get('roc_auc', 'N/A')}")
    logger.info(f"  F1 Score  : {best.data.metrics.get('f1_score', 'N/A')}")
    logger.info(f"  Precision : {best.data.metrics.get('precision', 'N/A')}")
    logger.info(f"  Recall    : {best.data.metrics.get('recall', 'N/A')}")
    logger.info("----------------------------------------")

    return {
        'run_id':    best.info.run_id,
        'model':     best.info.run_name,
        'roc_auc':   best.data.metrics.get('roc_auc'),
        'f1_score':  best.data.metrics.get('f1_score'),
        'precision': best.data.metrics.get('precision'),
        'recall':    best.data.metrics.get('recall'),
    }


def detect_drift(
    current_metrics: dict,
    threshold: float = 0.05
) -> bool:
    """
    Compare current model metrics against
    the best historical run to detect drift.
    """
    client     = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if not experiment:
        return False

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['metrics.roc_auc DESC'],
        max_results=5
    )

    if len(runs) < 2:
        logger.info("Not enough runs for drift detection")
        return False

    # Compare against best historical run
    best_roc_auc = max(
        r.data.metrics.get('roc_auc', 0) for r in runs
    )
    current_roc  = current_metrics.get('roc_auc', 0)
    drift        = best_roc_auc - current_roc

    logger.info("-- Drift Detection ---------------------")
    logger.info(f"  Best historical ROC AUC : {best_roc_auc:.4f}")
    logger.info(f"  Current ROC AUC         : {current_roc:.4f}")
    logger.info(f"  Drift                   : {drift:.4f}")
    logger.info(f"  Threshold               : {threshold}")

    if drift > threshold:
        logger.info("  Status: DRIFT DETECTED - retrain recommended")
        return True
    else:
        logger.info("  Status: No significant drift detected")
        return False