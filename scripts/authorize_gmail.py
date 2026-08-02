import argparse
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

from services.gmail_agent.google_gmail import READ_ONLY_SCOPE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Authorize the dummy Gmail account for read-only sync."
    )
    parser.add_argument(
        "--client",
        type=Path,
        default=Path("secrets/google-gmail-oauth-client.json"),
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=Path("secrets/google-gmail-token.json"),
    )
    args = parser.parse_args()

    if not args.client.is_file():
        raise SystemExit(f"OAuth client JSON not found: {args.client}")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(args.client),
        [READ_ONLY_SCOPE],
    )
    credentials = flow.run_local_server(port=0)
    args.token.parent.mkdir(parents=True, exist_ok=True)
    args.token.write_text(credentials.to_json(), encoding="utf-8")
    args.token.chmod(0o600)
    print(f"Gmail OAuth token saved to {args.token}")


if __name__ == "__main__":
    main()
