import sys
from utils.logger import get_logger

logger = get_logger('finshield.main')

def run_phase1() -> None:
    from ingestion.plaid_client import get_plaid_client
    from ingestion.fetch_transactions import fetch_transactions
    from ingestion.transform import (
        transform_transactions,
        validate_dataframe,
        get_summary_stats
    )
    from ingestion.bigquery_loader import load_to_bigquery, verify_bigquery_load

    logger.info("FinShield - Phase 1: Ingestion Started")
    logger.info("=" * 50)

    client       = get_plaid_client()
    transactions = fetch_transactions(client)
    df           = transform_transactions(transactions)

    if not validate_dataframe(df):
        logger.error("Validation failed - pipeline stopped")
        return

    get_summary_stats(df)
    load_to_bigquery(df)
    verify_bigquery_load()

    logger.info("=" * 50)
    logger.info("FinShield - Phase 1: Complete")


def run_phase2() -> None:
    from eda.cleaner import load_from_bigquery, clean_data, save_cleaned_to_bigquery
    from eda.explorer import run_eda
    from eda.hypothesis import run_hypothesis_tests

    logger.info("FinShield - Phase 2: EDA + Hypothesis Testing Started")
    logger.info("=" * 50)

    df_raw   = load_from_bigquery()
    df_clean = clean_data(df_raw)
    save_cleaned_to_bigquery(df_clean)
    run_eda(df_clean)
    run_hypothesis_tests(df_clean)

    logger.info("=" * 50)
    logger.info("FinShield - Phase 2: Complete")

def run_phase3() -> None:
    from model.features import load_cleaned_data, engineer_features, get_feature_matrix
    from model.trainer import split_data, train_models, save_best_model
    from model.evaluator import run_evaluation
    import pickle
    import os

    logger.info("FinShield - Phase 3: ML Model Started")
    logger.info("=" * 50)

    # Step 1 - Load cleaned data
    df = load_cleaned_data()

    # Step 2 - Engineer features
    df = engineer_features(df)

    # Step 3 - Get feature matrix
    X, y = get_feature_matrix(df)

    feature_cols = list(X.columns)

    # Step 4 - Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5 - Train models
    models = train_models(X_train, y_train)

    # Step 6 - Save best model
    best_name = save_best_model(models, X_test, y_test)

    # Step 7 - Evaluate best model
    best_model = models[best_name]
    run_evaluation(best_model, X_test, y_test, best_name, feature_cols)

    logger.info("=" * 50)
    logger.info("FinShield - Phase 3: Complete")
    logger.info("Model saved in: models/")
    logger.info("Plots saved in: plots/")

if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "1"

    if phase == "1":
        run_phase1()
    elif phase == "2":
        run_phase2()
    elif phase == "3":
        run_phase3()
    else:
        logger.error(f"Unknown phase: {phase}")