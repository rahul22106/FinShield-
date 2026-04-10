import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.logger import get_logger

logger = get_logger(__name__)


def load_cleaned_data() -> pd.DataFrame:
    from google.cloud import bigquery
    from config.settings import GCP_PROJECT_ID, BQ_DATASET

    client = bigquery.Client(project=GCP_PROJECT_ID)
    query  = f"""
        SELECT *
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.cleaned_transactions`
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df)} rows from BigQuery")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    logger.info("Engineering features...")

    # Encode payment_channel
    le = LabelEncoder()
    df['payment_channel_enc'] = le.fit_transform(
        df['payment_channel'].astype(str)
    )

    # Encode amount_range
    amount_map = {
        'Low': 0, 'Medium': 1,
        'High': 2, 'Very High': 3
    }
    df['amount_range_enc'] = df['amount_range'].map(amount_map)

    # Transaction velocity — how many transactions same day
    df['date_str'] = df['date'].astype(str).str[:10]
    daily_counts   = df.groupby('date_str')['transaction_id'].transform('count')
    df['daily_tx_count'] = daily_counts

    # High amount flag
    df['is_high_amount'] = (df['amount'] > 900).astype(int)

    # Online transaction flag
    df['is_online'] = (df['payment_channel'] == 'online').astype(int)

    logger.info(f"Features engineered: {df.shape}")
    return df


def get_feature_matrix(df: pd.DataFrame):
    feature_cols = [
        'amount',
        'amount_log',
        'amount_range_enc',
        'payment_channel_enc',
        'is_high_amount',
        'is_online',
        'daily_tx_count',
        'month',
    ]

    X = df[feature_cols]
    y = df['Class']

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    return X, y