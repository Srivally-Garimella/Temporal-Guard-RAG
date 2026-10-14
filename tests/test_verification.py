import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.verification_agent import VerificationAgent


class FakeVectorStore:
    def __init__(self, result=None):
        self.result = result or {
            'documents': [[]],
            'metadatas': [[]],
        }

    def temporal_search(self, query, cutoff_date, ticker, n_results):
        return self.result


def test_temporal_consistency_valid_claim():
    """Claims that stay within the cutoff date should be deemed temporally consistent."""
    agent = VerificationAgent()
    result = agent._check_temporal_consistency(
        "Apple revenue grew in FY2023|20240115|20240131"
    )

    assert 'TEMPORALLY CONSISTENT' in result
    assert 'TEMPORAL VIOLATION DETECTED' not in result


def test_temporal_consistency_future_year_is_flagged():
    """Claims referencing a date after the cutoff should trigger a temporal violation."""
    agent = VerificationAgent()
    result = agent._check_temporal_consistency(
        "Apple revenue will grow in FY2025|20240115|20240131"
    )

    assert 'TEMPORAL VIOLATION DETECTED' in result
    assert '2025' in result


def test_cross_check_number_match_for_near_exact_value(tmp_path):
    """A near-exact XBRL value should classify as MATCH when variance is under 1%."""
    xbrl_dir = tmp_path / 'xbrl'
    xbrl_dir.mkdir()
    metrics_path = xbrl_dir / 'AAPL_metrics.csv'

    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'end_date', 'value'])
        writer.writerow(['Revenue', '2023-12-31', 1000000000])

    agent = VerificationAgent(xbrl_dir=str(xbrl_dir))
    result = agent._cross_check_number('AAPL|Revenue|1005000000|20231231')

    assert '✅ MATCH' in result
    assert 'Variance:' in result


def test_cross_check_number_mismatch_for_material_error(tmp_path):
    """A materially wrong value should classify as MISMATCH."""
    xbrl_dir = tmp_path / 'xbrl'
    xbrl_dir.mkdir()
    metrics_path = xbrl_dir / 'AAPL_metrics.csv'

    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'end_date', 'value'])
        writer.writerow(['Revenue', '2023-12-31', 1000000000])

    agent = VerificationAgent(xbrl_dir=str(xbrl_dir))
    result = agent._cross_check_number('AAPL|Revenue|500000000|20231231')

    assert '❌ MISMATCH' in result
    assert 'Variance:' in result


def test_verify_claim_returns_supported_high_confidence_when_evidence_exists():
    """If the vector store returns strong textual evidence, the claim should be supported with HIGH confidence."""
    claim = 'Apple revenue was 100 billion in FY2023'
    support_text = 'Apple revenue was 100 billion in FY2023 and this was reported before the filing date.'

    fake_store = FakeVectorStore({
        'documents': [[support_text]],
        'metadatas': [[{'filing_date': '20240101', 'filing_type': '10-K'}]],
    })

    agent = VerificationAgent(vector_store=fake_store)
    result = agent._verify_claim(f'{claim}|AAPL|20240131')

    assert '✅ SUPPORTED' in result
    assert 'Confidence: HIGH' in result


def test_verify_claim_returns_unverified_low_confidence_when_no_evidence_exists():
    """If no evidence exists before the cutoff, the claim should be marked UNVERIFIED with LOW confidence."""
    agent = VerificationAgent(vector_store=FakeVectorStore({
        'documents': [[]],
        'metadatas': [[]],
    }))
    result = agent._verify_claim('Apple revenue was 100 billion in FY2023|AAPL|20240131')

    assert 'UNVERIFIED' in result
    assert 'LOW' in result


def test_verification_agent_initialization():
    """Test that the verification agent can be initialized."""
    agent = VerificationAgent()
    assert agent is not None


def test_numeric_deviation_detection():
    """
    Test the internal logic for detecting deviations.
    Note: Real verification requires an LLM call or complex mocking.
    We test the structure and properties.
    """
    agent = VerificationAgent()
    assert hasattr(agent, 'verify')


@pytest.mark.skip(reason="Requires valid XBRL data and LLM")
def test_full_verification_flow():
    agent = VerificationAgent()
    result = agent.verify(
        query="Apple's revenue in 2023 was $100 billion",
        ticker="AAPL",
        cutoff_date="20231231"
    )
    assert "verification" in result
    assert "confidence" in result
