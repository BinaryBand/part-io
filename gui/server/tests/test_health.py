"""Smoke test for the /health route."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_calls_real_partio_core() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["plan_cut_smoke_test"] == [
        {"start_seconds": 0.0, "end_seconds": 10.0},
        {"start_seconds": 25.0, "end_seconds": 30.0},
    ]
