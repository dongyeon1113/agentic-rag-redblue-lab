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
