import datetime
import time
from plaid.model.sandbox_public_token_create_request import SandboxPublicTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.products import Products
from config.settings import START_DATE, TRANSACTION_COUNT
from utils.logger import get_logger

logger = get_logger(__name__)

SANDBOX_INSTITUTIONS = [
    'ins_109508',
    'ins_109509',
    'ins_109510',
    'ins_109511',
    'ins_109512',
    'ins_109513',
    'ins_109514',
    'ins_109515',
    'ins_109516',
    'ins_109517',
    'ins_109518',
    'ins_109519',
    'ins_109520',
    'ins_109521',
    'ins_109522',
    'ins_109523',
    'ins_109524',
    'ins_109525',
    'ins_109526',
    'ins_109527',
]


def get_access_token(client, institution_id: str) -> str:
    try:
        pub_request = SandboxPublicTokenCreateRequest(
            institution_id=institution_id,
            initial_products=[Products('transactions')]
        )
        public_token = client.sandbox_public_token_create(
            pub_request
        ).public_token

        exchange_request = ItemPublicTokenExchangeRequest(
            public_token=public_token
        )
        access_token = client.item_public_token_exchange(
            exchange_request
        ).access_token
        logger.info(f"Access token obtained for {institution_id}")
        return access_token
    except Exception as e:
        logger.error(f"Token exchange failed for {institution_id}: {e}")
        raise


def fetch_transactions_for_token(client, access_token: str) -> list:
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
    return client.transactions_get(request).transactions


def fetch_transactions(client, access_token: str = None) -> list:
    """
    Fetch transactions from multiple sandbox institutions
    and combine into one list.
    """
    all_transactions = []

    for institution_id in SANDBOX_INSTITUTIONS:
        try:
            logger.info(f"Fetching from {institution_id}...")
            token = get_access_token(client, institution_id)

            logger.info("Waiting for sandbox to prepare transactions...")
            time.sleep(10)

            transactions = fetch_transactions_for_token(client, token)
            all_transactions.extend(transactions)
            logger.info(f"{len(transactions)} transactions from {institution_id}")

        except Exception as e:
            logger.error(f"Skipping {institution_id}: {e}")
            continue

    logger.info(f"Total transactions fetched: {len(all_transactions)}")
    return all_transactions