import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)
from utils.logger import get_logger

logger = get_logger(__name__)

PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Full evaluation of the best model."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    f1        = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall    = recall_score(y_test, preds)
    roc_auc   = roc_auc_score(y_test, proba)

    logger.info("-- Model Evaluation --------------------")
    logger.info(f"  Model     : {model_name}")
    logger.info(f"  F1 Score  : {f1:.4f}")
    logger.info(f"  Precision : {precision:.4f}")
    logger.info(f"  Recall    : {recall:.4f}")
    logger.info(f"  ROC AUC   : {roc_auc:.4f}")
    logger.info("----------------------------------------")
    logger.info("\n" + classification_report(
        y_test, preds,
        target_names=['Legit', 'Fraud']
    ))

    return {
        'model_name': model_name,
        'f1':         f1,
        'precision':  precision,
        'recall':     recall,
        'roc_auc':    roc_auc,
        'preds':      preds,
        'proba':      proba
    }


def plot_confusion_matrix(y_test, preds) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=['Legit', 'Fraud'],
        yticklabels=['Legit', 'Fraud']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/06_confusion_matrix.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 06_confusion_matrix.png")


def plot_roc_curve(y_test, proba) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc         = roc_auc_score(y_test, proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#F44336',
             label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/07_roc_curve.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 07_roc_curve.png")


def plot_feature_importance(model, feature_cols: list) -> None:
    """Plot feature importance for tree models."""

    actual_model = model

    # Handle ensemble — extract best estimator from it
    if hasattr(model, 'estimators_'):
        for est_tuple in model.estimators_:
            # VotingClassifier stores as list of fitted estimators
            est = est_tuple if not isinstance(est_tuple, tuple) else est_tuple[1]
            if hasattr(est, 'feature_importances_'):
                actual_model = est
                logger.info(f"Extracting feature importance from ensemble")
                break

    if not hasattr(actual_model, 'feature_importances_'):
        logger.info("Feature importance not available")
        return

    importance = pd.DataFrame({
        'feature':    feature_cols,
        'importance': actual_model.feature_importances_
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(
        importance['feature'],
        importance['importance'],
        color='#2196F3',
        alpha=0.8
    )
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/08_feature_importance.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 08_feature_importance.png")


def plot_precision_recall(y_test, proba) -> None:
    """Plot precision recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#2196F3')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/09_precision_recall.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 09_precision_recall.png")


def plot_model_comparison(models: dict, X_test, y_test) -> None:
    """Compare all models visually."""
    from sklearn.metrics import f1_score, roc_auc_score

    names   = []
    f1s     = []
    aucs    = []

    for name, model in models.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        names.append(name)
        f1s.append(f1_score(y_test, preds))
        aucs.append(roc_auc_score(y_test, proba))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # F1 comparison
    axes[0].bar(names, f1s, color=['#2196F3', '#F44336', '#4CAF50'])
    axes[0].set_title('Model F1 Score Comparison')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(f1s):
        axes[0].text(i, v + 0.01, f'{v:.4f}',
                    ha='center', fontweight='bold')

    # AUC comparison
    axes[1].bar(names, aucs, color=['#2196F3', '#F44336', '#4CAF50'])
    axes[1].set_title('Model ROC AUC Comparison')
    axes[1].set_ylabel('ROC AUC')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(aucs):
        axes[1].text(i, v + 0.01, f'{v:.4f}',
                    ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/10_model_comparison.png', dpi=150)
    plt.close()
    logger.info("Plot saved: 10_model_comparison.png")


def run_evaluation(model, X_test, y_test,
                   model_name: str,
                   feature_cols: list,
                   all_models: dict = None) -> dict:
    """Run full evaluation pipeline."""
    logger.info("Starting model evaluation...")

    results = evaluate_model(model, X_test, y_test, model_name)
    plot_confusion_matrix(y_test, results['preds'])
    plot_roc_curve(y_test, results['proba'])
    plot_feature_importance(model, feature_cols)
    plot_precision_recall(y_test, results['proba'])

    if all_models:
        plot_model_comparison(all_models, X_test, y_test)

    return results