import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


def load_from_bigquery() -> pd.DataFrame:
    """Load raw transactions from BigQuery."""
    from google.cloud import bigquery
    from config.settings import GCP_PROJECT_ID, BQ_DATASET, BQ_TABLE

    client = bigquery.Client(project=GCP_PROJECT_ID)
    query  = f"""
        SELECT *
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df)} rows from BigQuery")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    - Remove duplicates
    - Handle nulls
    - Fix data types
    - Add engineered features
    """
    logger.info("Starting data cleaning...")
    original_shape = df.shape

    # Remove duplicates
    df = df.drop_duplicates(subset=['transaction_id'])
    logger.info(f"After dedup: {len(df)} rows")

    # Fill nulls
    df['merchant_name']   = df['merchant_name'].fillna('Unknown')
    df['category']        = df['category'].fillna('Uncategorized')
    df['payment_channel'] = df['payment_channel'].fillna('unknown')

    # Fix data types
    df['date']   = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(float)
    df['Class']  = df['Class'].astype(int)

    # Use absolute amount
    df['amount'] = df['amount'].abs()

    # Feature engineering
    df['day_of_week']  = df['date'].dt.dayofweek
    df['month']        = df['date'].dt.month
    df['hour']         = df['date'].dt.hour
    df['is_weekend']   = df['day_of_week'].isin([5, 6]).astype(int)
    df['amount_log']   = df['amount'].apply(lambda x: __import__('math').log1p(x))
    df['amount_range'] = pd.cut(
        df['amount'],
        bins=[0, 50, 200, 500, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    logger.info(f"Cleaning complete: {original_shape} -> {df.shape}")
    return df


def save_cleaned_to_bigquery(df: pd.DataFrame) -> None:
    """Save cleaned data to BigQuery as a new table."""
    import pandas_gbq
    from config.settings import GCP_PROJECT_ID, BQ_DATASET

    destination = f'{BQ_DATASET}.cleaned_transactions'
    pandas_gbq.to_gbq(
        df,
        destination_table=destination,
        project_id=GCP_PROJECT_ID,
        if_exists='replace'
    )
    logger.info(f"Cleaned data saved to BigQuery: {destination}")