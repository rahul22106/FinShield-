import random
import pandas as pd
from config.settings import REQUIRED_COLUMNS
from utils.logger import get_logger

logger = get_logger(__name__)


def assign_fraud_label(t) -> int:

    score = 0

    # Signal 1 — High amount
    if abs(t.amount) > 500:
        score += 2
    elif abs(t.amount) > 200:
        score += 1

    # Signal 2 — Risky payment channel
    if t.payment_channel == 'online':
        score += 2
    elif t.payment_channel == 'other':
        score += 1

    # Signal 3 — Unknown merchant
    if not t.merchant_name:
        score += 2

    # Signal 4 — Risky category
    risky_categories = [
        'Travel', 'Airlines and Aviation Services',
        'Shops', 'Digital Purchase',
        'Service', 'Subscription'
    ]
    if t.category and any(
        cat in str(t.category) for cat in risky_categories
    ):
        score += 1

    # Signal 5 — Random noise
    score += random.randint(0, 2)

    # Fraud if score >= 5
    return 1 if score >= 5 else 0


def transform_transactions(transactions: list) -> pd.DataFrame:
    random.seed(42)  # reproducibility
    records = []

    for t in transactions:
        records.append({
            'transaction_id':  t.transaction_id,
            'date':            str(t.date),
            'amount':          t.amount,
            'merchant_name':   t.merchant_name or 'Unknown',
            'category':        t.category[0] if t.category else 'Uncategorized',
            'payment_channel': t.payment_channel,
            'Class':           assign_fraud_label(t)
        })

    df = pd.DataFrame(records)
    logger.info(f"DataFrame created: {df.shape}")
    logger.info(f"Fraud rate: {df['Class'].mean()*100:.1f}%")
    return df


def validate_dataframe(df: pd.DataFrame) -> bool:
    checks = {
        'has_rows':         len(df) > 0,
        'required_columns': all(c in df.columns for c in REQUIRED_COLUMNS),
        'binary_class':     df['Class'].isin([0, 1]).all(),
        'no_null_amounts':  df['amount'].isnull().sum() == 0,
        'no_null_ids':      df['transaction_id'].isnull().sum() == 0,
        'valid_amounts':    df['amount'].notna().all(),
        'has_fraud':        df['Class'].sum() > 0,
        'has_legit':        (df['Class'] == 0).sum() > 0,
    }

    all_passed = True
    for check, result in checks.items():
        status = "OK" if result else "FAIL"
        logger.info(f"  {status} {check}: {result}")
        if not result:
            all_passed = False
    return all_passed


def get_summary_stats(df: pd.DataFrame) -> None:
    """Log summary statistics of the DataFrame."""
    logger.info("-- Summary Stats ------------------")
    logger.info(f"  Total rows  : {len(df)}")
    logger.info(f"  Fraud       : {df['Class'].sum()}")
    logger.info(f"  Legit       : {(df['Class']==0).sum()}")
    logger.info(f"  Fraud rate  : {df['Class'].mean()*100:.1f}%")
    logger.info(f"  Avg amount  : ${df['amount'].mean():.2f}")
    logger.info(f"  Max amount  : ${df['amount'].max():.2f}")
    logger.info(f"  Date range  : {df['date'].min()} to {df['date'].max()}")
    logger.info("-----------------------------------")