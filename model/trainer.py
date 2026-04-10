import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


def split_data(X, y):
    """Split into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logger.info(f"Train size: {X_train.shape[0]}")
    logger.info(f"Test size : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def tune_xgboost(X_train, y_train) -> XGBClassifier:
    logger.info("Tuning XGBoost hyperparameters...")

    scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])

    param_grid = {
        'n_estimators':     [100, 200, 300],
        'max_depth':        [3, 4, 5],
        'learning_rate':    [0.01, 0.05, 0.1],
        'subsample':        [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    model = XGBClassifier(
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)
    logger.info(f"Best XGBoost params : {grid.best_params_}")
    logger.info(f"Best XGBoost CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_


def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    logger.info("Tuning Random Forest hyperparameters...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth':    [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
    }

    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)
    logger.info(f"Best RF params : {grid.best_params_}")
    logger.info(f"Best RF CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_ensemble(X_train, y_train,
                   xgb_model, rf_model) -> VotingClassifier:
    logger.info("Training ensemble model...")

    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf',  rf_model)
        ],
        voting='soft'
    )

    ensemble.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        ensemble, X_train, y_train,
        cv=cv, scoring='f1'
    )
    logger.info(f"Ensemble CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    return ensemble


def train_models(X_train, y_train) -> dict:
    logger.info("Starting model training pipeline...")
    logger.info("-" * 40)

    # Step 1 — Tune XGBoost
    xgb_model = tune_xgboost(X_train, y_train)

    # Step 2 — Tune Random Forest
    rf_model = tune_random_forest(X_train, y_train)

    # Step 3 — Train Ensemble
    ensemble = train_ensemble(X_train, y_train, xgb_model, rf_model)

    models = {
        'xgboost':        xgb_model,
        'random_forest':  rf_model,
        'ensemble':       ensemble
    }

    logger.info("-" * 40)
    return models


def save_best_model(models: dict, X_test, y_test) -> str:
    from sklearn.metrics import f1_score

    logger.info("Evaluating all models on test set...")
    best_name  = None
    best_score = 0

    for name, model in models.items():
        preds = model.predict(X_test)
        score = f1_score(y_test, preds)
        logger.info(f"  {name} Test F1: {score:.4f}")

        if score > best_score:
            best_score = score
            best_name  = name

    logger.info(f"Best model: {best_name} (F1={best_score:.4f})")

    # Save best model
    path = os.path.join(MODEL_DIR, f'{best_name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(models[best_name], f)
    logger.info(f"Model saved to {path}")

    return best_name