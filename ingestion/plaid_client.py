from plaid.api import plaid_api
from plaid.configuration import Configuration
from plaid.api_client import ApiClient
from config.settings import PLAID_CLIENT_ID, PLAID_SECRET, PLAID_HOST
from utils.logger import get_logger

logger = get_logger(__name__)

def get_plaid_client() -> plaid_api.PlaidApi:
    if not PLAID_CLIENT_ID or not PLAID_SECRET:
        raise ValueError("❌ Plaid credentials missing in .env")
    configuration = Configuration(
        host=PLAID_HOST,
        api_key={
            'clientId': PLAID_CLIENT_ID,
            'secret':   PLAID_SECRET,
        }
    )
    client = plaid_api.PlaidApi(ApiClient(configuration))
    logger.info("✅ Plaid client initialized")
    return client