import datetime
import time
from plaid.model.sandbox_public_token_create_request import SandboxPublicTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.products import Products
from config.settings import (
    PLAID_INSTITUTION_ID,
    START_DATE, TRANSACTION_COUNT
)
from utils.logger import get_logger

logger = get_logger(__name__)


def get_access_token(client) -> str:
    try:
        pub_request = SandboxPublicTokenCreateRequest(
            institution_id=PLAID_INSTITUTION_ID,
            initial_products=[Products('transactions')]
        )
        public_token = client.sandbox_public_token_create(
            pub_request
        ).public_token
        logger.info("✅ Public token created")

        exchange_request = ItemPublicTokenExchangeRequest(
            public_token=public_token
        )
        access_token = client.item_public_token_exchange(
            exchange_request
        ).access_token
        logger.info("✅ Access token obtained")
        return access_token
    except Exception as e:
        logger.error(f"❌ Token exchange failed: {e}")
        raise

import datetime
import time  # ADD THIS

def fetch_transactions(client, access_token: str) -> list:
    try:
        # ADD THIS — wait for sandbox to prepare transactions
        logger.info("Waiting for sandbox transactions to load...")
        time.sleep(10)

        start = datetime.date.fromisoformat(START_DATE)
        end   = datetime.date.today()
        request = TransactionsGetRequest(
            access_token=access_token,
            start_date=start,
            end_date=end,
            options=TransactionsGetRequestOptions(
                count=TRANSACTION_COUNT
            )
        )
        transactions = client.transactions_get(request).transactions
        logger.info(f"✅ {len(transactions)} transactions fetched")
        logger.info(f"   Range: {start} → {end}")
        return transactions
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        raise