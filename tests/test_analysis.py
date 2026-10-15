"""
Test the new Financial Analysis capabilities.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
sys.path.insert(0, '.')

from src.agents.analysis_agent import FinancialAnalysisAgent
from src.agents import MultiAgentOrchestrator


def test_analysis_as_of_date_is_isolated_between_worker_threads():
    agent = object.__new__(FinancialAnalysisAgent)
    agent._analysis_date_context = ContextVar("test_analysis_date", default=None)

    def get_date(date):
        agent._analysis_date = date
        return agent._analysis_date

    with ThreadPoolExecutor(max_workers=2) as pool:
        dates = list(pool.map(get_date, ["20240101", "20250101"]))

    assert dates == ["20240101", "20250101"]

def test_projection_query():
    """Projection language should route to the forecast analysis path."""
    orchestrator = object.__new__(MultiAgentOrchestrator)
    assert orchestrator._classify_query("Project cash flow for Apple in 2027") == "projection"

def test_historical_extreme():
    """Historical extreme language should use the analysis path."""
    orchestrator = object.__new__(MultiAgentOrchestrator)
    assert orchestrator._classify_query("When was Microsoft's highest EPS recorded?") == "historical_extreme"

def test_ratio_query():
    """Ratio language should route to the structured analysis path."""
    orchestrator = object.__new__(MultiAgentOrchestrator)
    assert orchestrator._classify_query("What is Apple's ROE?") == "ratio"
