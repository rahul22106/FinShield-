import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.logger import get_logger

logger = get_logger(__name__)


def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned transactions from BigQuery."""
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

    # Transaction velocity
    df['date_str']       = df['date'].astype(str).str[:10]
    daily_counts         = df.groupby('date_str')['transaction_id'].transform('count')
    df['daily_tx_count'] = daily_counts

    # High amount flag
    df['is_high_amount'] = (df['amount'] > 900).astype(int)

    # Online transaction flag
    df['is_online'] = (df['payment_channel'] == 'online').astype(int)

    # NEW — Amount deviation from daily average
    daily_avg = df.groupby('date_str')['amount'].transform('mean')
    df['amount_vs_daily_avg'] = df['amount'] / (daily_avg + 1)

    # NEW — Is amount a round number (fraud signal)
    df['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)

    # NEW — Merchant frequency (rare merchants = risky)
    merchant_counts       = df['merchant_name'].map(
        df['merchant_name'].value_counts()
    )
    df['merchant_frequency'] = merchant_counts

    # NEW — Amount zscore (how unusual is this amount)
    df['amount_zscore'] = (
        (df['amount'] - df['amount'].mean()) / df['amount'].std()
    )

    # NEW — Is very high amount
    df['is_very_high_amount'] = (df['amount'] > 2000).astype(int)

    logger.info(f"Features engineered: {df.shape}")
    return df


def get_feature_matrix(df: pd.DataFrame):
    feature_cols = [
        'amount',
        'amount_log',
        'amount_range_enc',
        'payment_channel_enc',
        'is_high_amount',
        'is_very_high_amount',
        'is_online',
        'daily_tx_count',
        'month',
        'amount_vs_daily_avg',
        'is_round_amount',
        'merchant_frequency',
        'amount_zscore',
    ]

    X = df[feature_cols]
    y = df['Class']

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    return X, y