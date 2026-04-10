import pandas as pd
import pandas_gbq
from google.cloud import bigquery
from config.settings import (
    GCP_PROJECT_ID, BQ_DESTINATION,
    BQ_DATASET, BQ_TABLE
)
from utils.logger import get_logger

logger = get_logger(__name__)

def load_to_bigquery(df: pd.DataFrame) -> None:
    try:
        logger.info(f"Uploading {len(df)} rows to BigQuery...")
        pandas_gbq.to_gbq(
            df,
            destination_table=BQ_DESTINATION,
            project_id=GCP_PROJECT_ID,
            if_exists='replace',
            progress_bar=True
        )
        logger.info(f"✅ Loaded to {GCP_PROJECT_ID}.{BQ_DESTINATION}")
    except Exception as e:
        logger.error(f"❌ BigQuery load failed: {e}")
        raise

def verify_bigquery_load() -> None:
    try:
        client = bigquery.Client(project=GCP_PROJECT_ID)
        query = f"""
            SELECT
                COUNT(*)               AS total_rows,
                SUM(Class)             AS fraud_count,
                ROUND(AVG(amount), 2)  AS avg_amount,
                MIN(date)              AS from_date,
                MAX(date)              AS to_date
            FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        """
        result = client.query(query).to_dataframe()
        logger.info("── BigQuery Verification ──────────")
        logger.info(f"  Total rows  : {result['total_rows'][0]}")
        logger.info(f"  Fraud count : {result['fraud_count'][0]}")
        logger.info(f"  Avg amount  : ${result['avg_amount'][0]}")
        logger.info(f"  Date range  : {result['from_date'][0]} → {result['to_date'][0]}")
        logger.info("───────────────────────────────────")
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        raise