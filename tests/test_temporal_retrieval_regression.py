import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.temporal_agent import TemporalAgent
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
    """A quarter with actual filing lag elapsed should be valid at the analysis date."""
    agent = TemporalAgent()
    result = agent.validate_query(
        "What was Apple revenue in Q4 2023?",
        "20240215",
    )

    assert result["is_valid"] is True
    assert result["has_violations"] is False


def test_historical_point_in_time_query_is_rejected_before_data_is_available():
    """A quarter whose 10-Q filing lag has not elapsed should be rejected."""
    agent = TemporalAgent()
    result = agent.validate_query(
        "What was Apple revenue in Q4 2023?",
        "20240115",
    )

    assert result["is_valid"] is False
    assert result["has_violations"] is True
    assert "Q4 2023" in result["bias_detection"]


def test_future_reference_is_flagged_as_look_ahead():
    """A future-year query should trigger explicit look-ahead detection."""
    agent = TemporalAgent()
    result = agent.validate_query(
        "What will Apple revenue be in 2025?",
        "20231001",
    )

    assert result["has_violations"] is True or "LOOK-AHEAD BIAS" in result["bias_detection"]
