import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

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


def test_missing_job_returns_404():
    client = TestClient(app)
    resp = client.get('/api/jobs/not-a-real-job-id')
    assert resp.status_code == 404
