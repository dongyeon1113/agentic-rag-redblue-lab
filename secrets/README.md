# Local secrets

Place the Google service-account key at
`secrets/google-service-account.json`. JSON key files in this directory are
ignored by Git and must never be committed.

Share the target Drive folder with the service account's `client_email` as a
viewer. Then set these values in `.env`:

```text
DRIVE_SYNC_ENABLED=true
GOOGLE_DRIVE_FOLDER_ID=<folder ID>
GOOGLE_SERVICE_ACCOUNT_FILE=./secrets/google-service-account.json
```

## Gmail OAuth

Place the desktop OAuth client JSON at
`secrets/google-gmail-oauth-client.json`, then authorize the dummy Gmail
account once:

```bash
python3 scripts/authorize_gmail.py
```

This creates the ignored `secrets/google-gmail-token.json`. Enable Gmail sync
in `.env`:

```text
GMAIL_SYNC_ENABLED=true
GMAIL_DATA_FILE=/app/data/google_gmail_messages.json
GMAIL_QUERY=in:anywhere
GMAIL_MAX_MESSAGES=100
GOOGLE_GMAIL_TOKEN_FILE=./secrets/google-gmail-token.json
```
