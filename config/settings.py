import os
from dotenv import load_dotenv

load_dotenv()

PLAID_CLIENT_ID        = os.getenv('PLAID_CLIENT_ID')
PLAID_SECRET           = os.getenv('PLAID_SECRET')
PLAID_ENV              = os.getenv('PLAID_ENV', 'sandbox')
PLAID_HOST             = "https://sandbox.plaid.com"
PLAID_INSTITUTION_ID   = 'ins_109508'

GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'finshield-492903')
BQ_DATASET             = 'fraud_data'
BQ_TABLE               = 'raw_transactions'
BQ_DESTINATION         = f'{BQ_DATASET}.{BQ_TABLE}'

START_DATE             = '2010-01-01'
TRANSACTION_COUNT      = 500
FRAUD_AMOUNT_THRESHOLD = 900

REQUIRED_COLUMNS = [
    'transaction_id',
    'date',
    'amount',
    'merchant_name',
    'category',
    'payment_channel',
    'Class'
]