import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def train_models(X_train, y_train) -> dict:
    logger.info("Training models...")

    models = {
        'logistic_regression': LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ),
        'random_forest': RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    }

    trained = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        logger.info(f"Training {name}...")

        # Scale for logistic regression
        if name == 'logistic_regression':
            scaler  = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)

            # Cross validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=cv, scoring='f1'
            )
        else:
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='f1'
            )

        logger.info(f"  {name} CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        trained[name] = model

    return trained


def save_best_model(models: dict, X_test, y_test) -> str:
    from sklearn.metrics import f1_score

    best_name  = None
    best_score = 0

    for name, model in models.items():
        if name == 'logistic_regression':
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)
            preds = model.predict(X_test_scaled)
        else:
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