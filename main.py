from ingestion.plaid_client import get_plaid_client
from ingestion.fetch_transactions import get_access_token, fetch_transactions
from ingestion.transform import (
    transform_transactions,
    validate_dataframe,
    get_summary_stats
)
from ingestion.bigquery_loader import load_to_bigquery, verify_bigquery_load
from utils.logger import get_logger

logger = get_logger('finshield.main')

def run_phase1() -> None:
    logger.info("🚀 FinShield — Phase 1: Ingestion Started")
    logger.info("=" * 50)

    client       = get_plaid_client()
    access_token = get_access_token(client)
    transactions = fetch_transactions(client, access_token)
    df           = transform_transactions(transactions)

    if not validate_dataframe(df):
        logger.error("❌ Validation failed — pipeline stopped")
        return

    get_summary_stats(df)
    load_to_bigquery(df)
    verify_bigquery_load()

    logger.info("=" * 50)
    logger.info("✅ FinShield — Phase 1: Complete")

if __name__ == "__main__":
    run_phase1()