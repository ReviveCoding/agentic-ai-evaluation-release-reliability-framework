from fastapi.testclient import TestClient

from agentic_eval_framework.api.app import app


def test_api_uses_shared_engine_and_persists_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_TRACE_DB", str(tmp_path / "api.sqlite"))
    monkeypatch.setenv("AGENTIC_MODEL_DIR", str(tmp_path / "missing_model"))
    monkeypatch.setenv("AGENTIC_SERVICE_DOCS", str(tmp_path / "missing_service_docs.jsonl"))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        created = client.post(
            "/runs",
            json={
                "scenario_id": "api_test",
                "user_request": "Show hotels in Boston",
                "service": "Hotels_1",
                "intent": "SearchHotel",
                "known_slots": {"location": "Boston"},
                "missing_slots": [],
                "required_slots": ["location"],
            },
        )
        assert created.status_code == 200
        payload = created.json()
        assert payload["expected_tools"] == ["search_places", "final_answer"]
        assert payload["release_decision"] == "PASS"

        fetched = client.get(f"/runs/{payload['run_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["run_id"] == payload["run_id"]


def test_api_readiness_and_explicit_oracle(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_TRACE_DB", str(tmp_path / "api_ready.sqlite"))
    monkeypatch.setenv("AGENTIC_MODEL_DIR", str(tmp_path / "missing_model"))
    monkeypatch.setenv("AGENTIC_SERVICE_DOCS", str(tmp_path / "missing_service_docs.jsonl"))
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["database_integrity"] == "ok"

        created = client.post(
            "/runs",
            json={
                "scenario_id": "explicit_oracle",
                "user_request": "Find policy documents",
                "service": "Documents_1",
                "intent": "SearchDocument",
                "known_slots": {"topic": "policy"},
                "required_slots": ["topic"],
                "expected_tools": ["search_docs", "final_answer"],
                "release_gate_expected": "PASS",
            },
        )
        assert created.status_code == 200
        assert created.json()["expected_tools"] == ["search_docs", "final_answer"]
        assert created.json()["model_fingerprint"]


def test_api_rejects_unknown_oracle_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_TRACE_DB", str(tmp_path / "api_invalid.sqlite"))
    monkeypatch.setenv("AGENTIC_MODEL_DIR", str(tmp_path / "missing_model"))
    monkeypatch.setenv("AGENTIC_SERVICE_DOCS", str(tmp_path / "missing_service_docs.jsonl"))
    with TestClient(app) as client:
        response = client.post(
            "/runs",
            json={
                "user_request": "Do something",
                "expected_tools": ["invented_tool"],
            },
        )
        assert response.status_code == 422
