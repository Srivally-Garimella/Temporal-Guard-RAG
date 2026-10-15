import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.temporal_agent import TemporalAgent
from src.agents.orchestrator import MultiAgentOrchestrator
from src.rag_system.vector_store import TemporalVectorStore


def test_temporal_vector_store_filters_by_cutoff(tmp_path):
    """The temporal filter must exclude post-cutoff documents in an isolated collection."""
    store = TemporalVectorStore(
        persist_dir=str(tmp_path / "vector_db"),
        collection_name="regression_temporal_filter_test",
    )

    docs = [
        "Apple revenue Q4 2023 report before cutoff",
        "Apple revenue Q1 2024 report after cutoff",
    ]
    metadatas = [
        {"ticker": "AAPL", "filing_date": 20231231, "filing_type": "10-Q"},
        {"ticker": "AAPL", "filing_date": 20240430, "filing_type": "10-Q"},
    ]
    ids = ["pre_cutoff_doc", "post_cutoff_doc"]
    embeddings = [store.embedder.encode(doc).tolist() for doc in docs]

    store.collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=docs,
        metadatas=metadatas,
    )

    results = store.temporal_search(
        query="Apple revenue",
        cutoff_date="20240101",
        ticker="AAPL",
        n_results=5,
        query_embeddings=store.embedder.encode("Apple revenue").tolist(),
    )

    returned_docs = results.get("documents", [[]])[0]
    returned_metas = results.get("metadatas", [[]])[0]

    assert returned_docs, "vector store should return the pre-cutoff document"
    assert returned_metas, "metadata should be returned alongside documents"
    assert len(returned_docs) == 1, "post-cutoff documents must be excluded"
    assert int(returned_metas[0]["filing_date"]) < 20240101
    assert "Apple revenue Q4 2023 report before cutoff" in returned_docs[0]


def test_historical_point_in_time_query_is_allowed():
    """A Q3 query is valid after the 10-Q filing lag has elapsed."""
    agent = TemporalAgent()
    result = agent.validate_query(
        "What was Apple revenue in Q3 2023?",
        "20231115",
    )

    assert result["is_valid"] is True
    assert result["has_violations"] is False


def test_q4_query_is_rejected_before_10k_data_is_available():
    """Q4 uses the 10-K lag, so mid-February is too early for Q4 2023."""
    agent = TemporalAgent()
    result = agent.validate_query(
        "What was Apple revenue in Q4 2023?",
        "20240215",
    )

    assert result["is_valid"] is False
    assert result["has_violations"] is True
    assert "Q4 2023" in result["bias_detection"]


def test_q4_query_is_allowed_after_10k_filing_lag():
    """Q4 is valid once the conservative 10-K lag has elapsed."""
    agent = TemporalAgent()
    result = agent.validate_query(
        "What was Apple revenue in Q4 2023?",
        "20240301",
    )

    assert result["is_valid"] is True
    assert result["has_violations"] is False


def test_orchestrator_blocks_temporally_invalid_query_before_downstream_agents():
    """An unavailable Q4 query must be stopped before retrieval or analysis."""
    orchestrator = object.__new__(MultiAgentOrchestrator)
    orchestrator.temporal_agent = TemporalAgent()
    orchestrator.model_name = None

    result = orchestrator.process_query(
        query="What was Apple revenue in Q4 2023?",
        ticker="AAPL",
        analysis_date="20240215",
        verbose=False,
    )

    assert result["stages"]["temporal"]["is_valid"] is False
    assert result["metadata"]["temporal_rejected"] is True
    assert result["metadata"]["agents_used"] == ["temporal"]
    assert "Analysis blocked" in result["final_answer"]


def test_orchestrator_fails_closed_when_temporal_validation_raises():
    """A temporal validation error must not permit downstream analysis."""
    class BrokenTemporalAgent:
        def validate_query(self, query, analysis_date, allow_future_target=False):
            raise RuntimeError("validation unavailable")

    orchestrator = object.__new__(MultiAgentOrchestrator)
    orchestrator.temporal_agent = BrokenTemporalAgent()
    orchestrator.model_name = None

    result = orchestrator.process_query(
        query="Summarize AAPL historical revenue",
        ticker="AAPL",
        analysis_date="20240101",
        verbose=False,
    )

    assert result["stages"]["temporal"]["is_valid"] is False
    assert result["metadata"]["temporal_rejected"] is True
    assert result["metadata"]["agents_used"] == ["temporal"]
    assert "Temporal validation failed" in result["final_answer"]


def test_explicit_projection_target_is_allowed_without_future_evidence():
    orchestrator = object.__new__(MultiAgentOrchestrator)
    orchestrator.temporal_agent = TemporalAgent()
    orchestrator.model_name = None
    routed = {}

    def capture_analysis_route(**kwargs):
        routed.update(kwargs)
        return {"stages": {"temporal": kwargs["temporal_result"]}}

    orchestrator._process_analysis_query = capture_analysis_route
    result = orchestrator.process_query(
        query="Forecast Apple revenue in 2027",
        ticker="AAPL",
        analysis_date="20240101",
        verbose=False,
    )

    assert routed["query_type"] == "projection"
    assert result["stages"]["temporal"]["is_valid"] is True


def test_future_period_without_projection_intent_is_rejected():
    agent = TemporalAgent()
    result = agent.validate_query(
        "What was Apple revenue in 2027?",
        "20240101",
    )

    assert result["is_valid"] is False
    assert result["has_violations"] is True


def test_historical_market_data_lookup_does_not_fetch_live_data():
    from contextvars import ContextVar
    from src.agents.analysis_agent import FinancialAnalysisAgent

    class UnexpectedMarketAccess:
        def get_current_price(self, ticker):
            raise AssertionError("live market data must not be used for historical analysis")

    agent = object.__new__(FinancialAnalysisAgent)
    agent._analysis_date_context = ContextVar("historical_test_date", default=None)
    agent._analysis_date = "20240101"
    agent.stock_loader = UnexpectedMarketAccess()

    result = agent._get_market_data("AAPL")

    assert "unavailable for historical as-of date 20240101" in result


def test_future_reference_is_flagged_as_look_ahead():
    """A future-year query should trigger explicit look-ahead detection."""
    agent = TemporalAgent()
    result = agent.validate_query(
        "What will Apple revenue be in 2025?",
        "20231001",
    )

    assert result["has_violations"] is True or "LOOK-AHEAD BIAS" in result["bias_detection"]
