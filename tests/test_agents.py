from fastapi.testclient import TestClient

from services.drive_agent.app import app as drive_app
from services.gmail_agent.app import app as gmail_app
from services.local_db_agent.app import app as local_db_app
from services.orchestrator.app import app as orchestrator_app


def test_all_services_report_health() -> None:
    services = [
        (orchestrator_app, "orchestrator"),
        (local_db_app, "local-db-agent"),
        (gmail_app, "gmail-agent"),
        (drive_app, "drive-agent"),
    ]

    for app, expected_name in services:
        response = TestClient(app).get("/health")
        assert response.status_code == 200
        assert response.json()["service"] == expected_name


def test_local_db_search_endpoint() -> None:
    response = TestClient(local_db_app).post(
        "/search",
        json={"query": "capital of France", "limit": 3},
    )

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert len(hits) == 3
    assert hits[0]["document_id"] == "nq-sample-001"
    assert "Paris" in hits[0]["text"]
