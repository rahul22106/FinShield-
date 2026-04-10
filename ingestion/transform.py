import pandas as pd
from config.settings import FRAUD_AMOUNT_THRESHOLD, REQUIRED_COLUMNS
from utils.logger import get_logger

logger = get_logger(__name__)

def transform_transactions(transactions: list) -> pd.DataFrame:
    records = []
    for t in transactions:
        records.append({
            'transaction_id':  t.transaction_id,
            'date':            str(t.date),
            'amount':          t.amount,
            'merchant_name':   t.merchant_name or 'Unknown',
            'category':        t.category[0] if t.category else 'Uncategorized',
            'payment_channel': t.payment_channel,
            'Class':           1 if abs(t.amount) > FRAUD_AMOUNT_THRESHOLD else 0
        })
    df = pd.DataFrame(records)
    logger.info(f"DataFrame created: {df.shape}")
    return df

def validate_dataframe(df: pd.DataFrame) -> bool:
    checks = {
        'has_rows':         len(df) > 0,
        'required_columns': all(c in df.columns for c in REQUIRED_COLUMNS),
        'binary_class':     df['Class'].isin([0, 1]).all(),
        'no_null_amounts':  df['amount'].isnull().sum() == 0,
        'no_null_ids':      df['transaction_id'].isnull().sum() == 0,
        'valid_amounts':    df['amount'].notna().all(),
    }
    all_passed = True
    for check, result in checks.items():
        status = "OK" if result else "FAIL"
        logger.info(f"  {status} {check}: {result}")
        if not result:
            all_passed = False
    return all_passed

def get_summary_stats(df: pd.DataFrame) -> None:
    logger.info("-- Summary Stats ------------------")
    logger.info(f"  Total rows  : {len(df)}")
    logger.info(f"  Fraud       : {df['Class'].sum()}")
    logger.info(f"  Legit       : {(df['Class']==0).sum()}")
    logger.info(f"  Avg amount  : ${df['amount'].mean():.2f}")
    logger.info(f"  Max amount  : ${df['amount'].max():.2f}")
    logger.info(f"  Date range  : {df['date'].min()} to {df['date'].max()}")
    logger.info("-----------------------------------")