"""
Tests for Temporal Agent

Tests temporal consistency validation and look-ahead bias detection.
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.temporal_agent import TemporalAgent
from src.analysis.data_loader import FinancialDataLoader


class TestTemporalAgent:
    """Test suite for TemporalAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a temporal agent instance."""
        return TemporalAgent()
    
    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert len(agent.FILING_LAGS) > 0
        assert len(agent.tools) > 0
        
    def test_parse_date_valid(self, agent):
        """Test date parsing with valid dates."""
        # YYYYMMDD format
        result = agent._parse_date("20231015")
        assert result.year == 2023
        assert result.month == 10
        assert result.day == 15
        
        # With dashes
        result = agent._parse_date("2023-10-15")
        assert result.year == 2023
        
    def test_parse_date_invalid(self, agent):
        """Test date parsing with invalid dates."""
        with pytest.raises(ValueError):
            agent._parse_date("invalid")
            
    def test_calculate_information_cutoff(self, agent):
        """Test information cutoff calculation."""
        result = agent._calculate_information_cutoff("20231001")
        
        assert "INFORMATION CUTOFF ANALYSIS" in result
        assert "10-K" in result
        assert "10-Q" in result
        
    def test_check_document_availability_available(self, agent):
        """Test document availability when document should be available."""
        # 10-Q for Q1 (ends March 31) should be available by mid-May
        result = agent._check_document_availability("20230331|10-Q|20230601")
        
        assert "AVAILABLE" in result
        
    def test_check_document_availability_not_available(self, agent):
        """Test document availability when document not yet available."""
        # 10-Q for Q1 (ends March 31) should NOT be available on April 1
        result = agent._check_document_availability("20230331|10-Q|20230401")
        
        assert "NOT AVAILABLE" in result
        
    def test_validate_date_range_valid(self, agent):
        """Test valid date range."""
        result = agent._validate_date_range("20220101|20221231|20231001")
        
        assert "VALID" in result
        
    def test_validate_date_range_future_end(self, agent):
        """Test date range with future end date."""
        result = agent._validate_date_range("20220101|20240101|20231001")
        
        assert "LOOK-AHEAD BIAS" in result
        
    def test_detect_look_ahead_bias_clean(self, agent):
        """Test query with no look-ahead bias."""
        result = agent._detect_look_ahead_bias(
            "What was Apple revenue in Q2 2022?|20231001"
        )
        
        assert "CLEAN" in result or "No look-ahead bias" in result
        
    def test_detect_look_ahead_bias_detected(self, agent):
        """Test query with look-ahead bias."""
        result = agent._detect_look_ahead_bias(
            "What will Apple revenue be in 2025?|20231001"
        )
        
        # Should detect future reference or forward-looking language
        assert "WARNINGS" in result or "BIAS" in result
        
    def test_validate_query_valid(self, agent):
        """Test full query validation - valid query."""
        result = agent.validate_query(
            "What was revenue growth in fiscal 2022?",
            "20231001"
        )
        
        assert result['is_valid'] is True
        assert 'cutoff_analysis' in result
        assert 'bias_detection' in result
        
    def test_validate_query_with_future_reference(self, agent):
        """Test full query validation - query with future year."""
        result = agent.validate_query(
            "What will revenue be in 2025?",
            "20231001"
        )
        
        # May have warnings for forward-looking language
        assert 'has_warnings' in result or 'has_violations' in result

    def test_validate_query_rejects_invalid_analysis_date(self, agent):
        result = agent.validate_query("What was revenue?", "20240230")

        assert result['is_valid'] is False
        assert result['has_violations'] is True
        assert 'INVALID ANALYSIS DATE' in result['bias_detection']
        
    def test_get_cutoff_date(self, agent):
        """Test cutoff date calculation."""
        # 10-Q has 40 day lag
        cutoff = agent.get_cutoff_date("20231001", "10-Q")
        
        # Should be approximately 40 days before
        cutoff_dt = datetime.strptime(cutoff, '%Y%m%d')
        analysis_dt = datetime(2023, 10, 1)
        
        diff = (analysis_dt - cutoff_dt).days
        assert 35 <= diff <= 45  # Allow some flexibility
        
    def test_quarter_determination(self, agent):
        """Test quarter end dates are correct."""
        assert agent.QUARTER_ENDS[1] == (3, 31)   # Q1 ends March 31
        assert agent.QUARTER_ENDS[2] == (6, 30)   # Q2 ends June 30
        assert agent.QUARTER_ENDS[3] == (9, 30)   # Q3 ends September 30
        assert agent.QUARTER_ENDS[4] == (12, 31)  # Q4 ends December 31


class TestTemporalIntegration:
    """Integration tests for temporal consistency."""
    
    @pytest.fixture
    def agent(self):
        return TemporalAgent()
        
    def test_realistic_financial_query(self, agent):
        """Test with realistic financial query."""
        result = agent.validate_query(
            "What was Apple's revenue and profit margin in Q3 2023? "
            "How does it compare to Q3 2022?",
            "20231115"
        )
        
        # Q3 2023 10-Q should be available by mid-November
        assert result['is_valid'] is True
        
    def test_boundary_case_just_filed(self, agent):
        """Test boundary case where document just became available."""
        # Q2 ends June 30, 10-Q filed ~August 10 (40 days later)
        result = agent.validate_query(
            "What was Q2 2023 revenue?",
            "20230815"  # Just after typical filing date
        )
        
        assert result['is_valid'] is True


def test_metric_history_uses_filing_date_for_as_of_filter(tmp_path):
    metrics_path = tmp_path / "AAPL_metrics.csv"
    metrics_path.write_text(
        "metric,end_date,value,filed_date,form\n"
        "Revenue,2022-09-30,100,2022-11-01,10-K\n"
        "Revenue,2023-09-30,200,2024-11-01,10-K\n",
        encoding="utf-8",
    )
    loader = FinancialDataLoader(str(tmp_path))

    history = loader.get_metric_history("AAPL", "Revenue", as_of_date="20240101")

    assert history == [("2022-09-30", 100)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
