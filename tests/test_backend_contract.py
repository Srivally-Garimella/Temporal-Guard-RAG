import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

import api as legacy_api
from src.agents import calculation_agent
from backend.main import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get('/health')
    assert resp.status_code == 200
    payload = resp.json()
    assert payload['status'] == 'healthy'


def test_tickers_endpoint():
    client = TestClient(app)
    resp = client.get('/api/tickers')
    assert resp.status_code == 200
    payload = resp.json()
    assert 'tickers' in payload
    assert len(payload['tickers']) > 0


def test_analyze_queue_accepts_valid_request():
    client = TestClient(app)
    resp = client.post(
        '/api/analyze',
        json={
            'query': 'What was Apple revenue in Q4 2023?',
            'ticker': 'AAPL',
            'analysis_date': '20240115',
            'provider': 'ollama',
            'model_name': 'llama3.2',
            'fast_mode': False,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert 'job_id' in payload
    assert payload['status'] == 'queued'


def test_analyze_rejects_impossible_analysis_date():
    client = TestClient(app)
    response = client.post(
        '/api/analyze',
        json={
            'query': 'What was Apple revenue in Q3 2023?',
            'ticker': 'AAPL',
            'analysis_date': '20240230',
        },
    )

    assert response.status_code == 422


def test_missing_job_returns_404():
    client = TestClient(app)
    resp = client.get('/api/jobs/not-a-real-job-id')
    assert resp.status_code == 404


def test_root_api_calculation_route_uses_calculation_agent_contract(monkeypatch):
    class FakeCalculationAgent:
        def calculate(self, metric, ticker, date):
            return {"output": f"{metric}|{ticker}|{date}"}

    monkeypatch.setattr(calculation_agent, "CalculationAgent", FakeCalculationAgent)
    client = TestClient(legacy_api.app)

    response = client.post(
        "/api/v1/calculate",
        json={"ticker": "AAPL", "ratio": "ROE", "date": "20231001"},
    )

    assert response.status_code == 200
    assert response.json()["result"] == "ROE|AAPL|20231001"


def test_root_api_rejects_impossible_calculation_date():
    client = TestClient(legacy_api.app)
    response = client.post(
        "/api/v1/calculate",
        json={"ticker": "AAPL", "ratio": "ROE", "date": "20240230"},
    )

    assert response.status_code == 422
