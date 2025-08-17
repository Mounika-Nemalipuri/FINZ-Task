import requests
import base64
import os
from dotenv import load_dotenv
load_dotenv()
# === QuickBooks Config ===
QB_CLIENT_ID = os.getenv("QB_CLIENT_ID")          # Your QBO App's Client ID
QB_CLIENT_SECRET = os.getenv("QB_CLIENT_SECRET")  # Your QBO App's Client Secret
QB_REFRESH_TOKEN = os.getenv("QB_REFRESH_TOKEN")  # Refresh token obtained from OAuth flow
QB_REALM_ID = os.getenv("QB_REALM_ID")            # Company ID
QB_BASE_URL = "https://quickbooks.api.intuit.com/v3/company"

print("QuickBooks Online configuration loaded.", QB_BASE_URL, QB_REALM_ID, "Client ID:", QB_CLIENT_ID, QB_REFRESH_TOKEN)

def get_qb_access_token():
    """Refresh the QuickBooks Online access token."""
    token_url = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
    auth_header = base64.b64encode(f"{QB_CLIENT_ID}:{QB_CLIENT_SECRET}".encode()).decode()

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": QB_REFRESH_TOKEN
    }
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    print('payload:', payload)

    response = requests.post(token_url, headers=headers, data=payload)
    if response.status_code != 200:
        raise Exception(f"Error refreshing token: {response.text}")

    access_token = response.json()["access_token"]
    return access_token

def post_bill_to_quickbooks(vendor_name, amount, invoice_date, memo, currency="USD"):
    """Post a Bill to QuickBooks."""
    access_token = get_qb_access_token()
    bill_url = f"{QB_BASE_URL}/{QB_REALM_ID}/bill"

    # Note: In production, VendorRef should be looked up from QBO Vendor list
    bill_payload = {
        "VendorRef": {"name": vendor_name},
        "Line": [
            {
                "Amount": amount,
                "DetailType": "AccountBasedExpenseLineDetail",
                "AccountBasedExpenseLineDetail": {
                    "AccountRef": {"value": "1"}  # Replace with real Account ID
                }
            }
        ],
        "TxnDate": invoice_date,
        "CurrencyRef": {"value": currency},
        "PrivateNote": memo
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(bill_url, headers=headers, json=bill_payload)

    if response.status_code not in (200, 201):
        raise Exception(f"Error posting bill: {response.text}")

    return response.json()

# === Example Usage ===
if __name__ == "__main__":
    try:
        bill_response = post_bill_to_quickbooks(
            vendor_name="PG&E",
            amount=90,
            invoice_date="2025-08-10",
            memo="Laptop chargers",
            currency="USD"
        )
        print("Bill posted successfully:", bill_response)
        # get_qb_access_token()
    except Exception as e:
        print("Error:", e)
