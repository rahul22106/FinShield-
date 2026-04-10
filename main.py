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


if __name__ == "__main__":
    # Pass phase number as argument
    # python main.py 1  -> runs phase 1
    # python main.py 2  -> runs phase 2

    phase = sys.argv[1] if len(sys.argv) > 1 else "1"

    if phase == "1":
        run_phase1()
    elif phase == "2":
        run_phase2()
    else:
        logger.error(f"Unknown phase: {phase}")