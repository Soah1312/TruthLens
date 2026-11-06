import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client fixture for invoking FastAPI endpoints."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Verify that the root endpoint returns HTTP 200 and service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["application"] == "TruthLens"
    assert data["version"] == "1.0.0"
    assert data["status"] == "online"
    assert "documentation" in data
    assert "health" in data


def test_health_endpoint(client: TestClient):
    """Verify the exact expected health endpoint contract required by Phase 1."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["application"] == "TruthLens"
    assert data["version"] == "1.0.0"


def test_openapi_docs_available(client: TestClient):
    """Verify Swagger UI and OpenAPI schema availability."""
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    assert "swagger" in docs_response.text.lower() or "html" in docs_response.headers.get("content-type", "")

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    schema = openapi_response.json()
    assert schema["info"]["title"] == "TruthLens"
    assert "/api/health" in schema["paths"]


def test_cors_headers(client: TestClient):
    """Verify CORS middleware permits requests from the frontend origin."""
    response = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_error_handling_not_found(client: TestClient):
    """Verify 404 errors return a structured JSON response."""
    response = client.get("/api/nonexistent-route")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
