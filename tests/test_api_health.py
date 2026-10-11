from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_returns_spec_fields():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_version"]
    assert body["timestamp"]


def test_swagger_docs_available():
    assert client.get("/docs").status_code == 200
