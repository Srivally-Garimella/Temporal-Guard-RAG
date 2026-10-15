
import pytest
from src.agents.calculation_agent import CalculationAgent
from pathlib import Path


def _empty_xbrl_dir() -> str:
    return str(Path(__file__).parent / "fixtures" / "empty_xbrl")


def test_mock_helpers_include_warning():
    """Mock helpers must clearly disclose mock usage."""
    agent = CalculationAgent(xbrl_dir=_empty_xbrl_dir(), provider="ollama")

    mock_xbrl = agent._mock_xbrl_data("TESTCO")
    assert "[WARNING: MOCK DATA USED" in mock_xbrl

    mock_metric = agent._mock_metric_value("TESTCO", "Revenue", "20230630")
    assert "[WARNING: MOCK DATA USED" in mock_metric


def test_calculate_ratio_logic_uses_mock_when_no_xbrl():
    """Ratio calculation should be deterministic in fallback mode when no XBRL exists."""
    agent = CalculationAgent(xbrl_dir=_empty_xbrl_dir(), provider="ollama")

    # Force non-LLM fallback path.
    agent.agent = None

    result = agent._calculate_ratio("ROE|TESTCO|20230630")
    assert "Financial Ratio Calculation" in result
    assert "Result: ROE =" in result
    # With mock values: NetIncome 24.16B, Equity 56.727B -> ~42.59%
    assert "42.59%" in result
    assert "[WARNING: MOCK DATA USED" in result


def test_fallback_mechanism_sets_mode():
    """calculate() should mark fallback mode when the agent is disabled."""
    agent = CalculationAgent(xbrl_dir=_empty_xbrl_dir(), provider="ollama")
    agent.agent = None

    out = agent.calculate(metric="profit_margin", ticker="TESTCO", date="20231231")
    assert out["mode"] == "fallback"
    assert "profit_margin" in out["output"]
    assert "[WARNING: MOCK DATA USED" in out["output"]


def test_metric_lookup_excludes_results_filed_after_as_of_date(tmp_path):
    """A period end before the cutoff is insufficient if its filing came later."""
    metrics_path = tmp_path / "AAPL_metrics.csv"
    metrics_path.write_text(
        "ticker,metric,value,end_date,filed_date,form\n"
        "AAPL,Revenue,100,2022-09-30,2022-11-01,10-K\n"
        "AAPL,Revenue,200,2023-09-30,2024-11-01,10-K\n",
        encoding="utf-8",
    )
    agent = CalculationAgent(xbrl_dir=str(tmp_path), provider="ollama")

    result = agent._get_metric_value("AAPL|Revenue|20240101")

    assert "Value: $100" in result
    assert "Filed: 2022-11-01" in result
    assert "$200" not in result
