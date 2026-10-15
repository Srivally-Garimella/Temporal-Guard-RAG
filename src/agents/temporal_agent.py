"""
Temporal Agent for TemporalGuard-RAG

Specialized agent for enforcing temporal consistency and preventing look-ahead bias.
Acts as the temporal gatekeeper for all data access and analysis.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
import os
import re

from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import warnings

from .llm_provider import get_llm

# Suppress deprecation warnings for langgraph
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langgraph')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalAgent:
    """
    Temporal consistency enforcement agent.
    
    Key Responsibilities:
    - Validate date ranges for queries
    - Calculate point-in-time cutoffs
    - Detect temporal violations
    - Enforce filing lag constraints
    - Guard against look-ahead bias
    """
    
    SYSTEM_PROMPT = """You are a temporal consistency specialist responsible for preventing look-ahead bias in financial analysis.

Your primary mission is to ensure that NO analysis uses information that would not have been available at the specified point in time.

CRITICAL RULES:
1. ALWAYS validate dates before any data access
2. NEVER allow access to documents filed after the analysis date
3. Account for filing delays (10-K: ~60 days, 10-Q: ~40 days, earnings: typically delayed)
4. Explicitly state the effective information cutoff date
5. Flag any potential look-ahead bias violations
6. Consider market data availability (closing prices available next day)

Filing Lag Guidelines:
- 10-K Annual Reports: Available ~60-90 days after fiscal year end
- 10-Q Quarterly Reports: Available ~40-45 days after quarter end
- 8-K Current Reports: Available within days of event
- Earnings Calls: Transcripts typically available 1-2 days after call

When validating:
1. Parse the query date
2. Calculate the effective information cutoff
3. Identify what data would actually be available
4. Warn about any temporal inconsistencies
5. Recommend corrections if needed
"""

    # Filing lag constants (in days)
    FILING_LAGS = {
        "10-K": 60,   # Annual report ~60-90 days after FYE
        "10-Q": 40,   # Quarterly report ~40-45 days after quarter end
        "8-K": 4,     # Current report within 4 business days
        "earnings_call": 2,  # Transcript typically 1-2 days after call
        "stock_price": 1     # Available at market close same day, typically used next day
    }
    
    # Quarter end dates
    QUARTER_ENDS = {
        1: (3, 31),   # Q1 ends March 31
        2: (6, 30),   # Q2 ends June 30
        3: (9, 30),   # Q3 ends September 30
        4: (12, 31)   # Q4 ends December 31
    }

    def __init__(self, model_name: str = None, provider: str = None):
        """
        Initialize Temporal Agent.
        
        Args:
            model_name: LLM model name (default: auto-select)
            provider: 'openai' or 'ollama' (default: auto-detect)
        """
        self.model_name = model_name
        self.provider = provider
        
        # Initialize LLM using provider factory
        self.llm = get_llm(provider=provider, model_name=model_name, temperature=0)
        
        # Create tools using decorator (langgraph style)
        @tool
        def calculate_information_cutoff(analysis_date: str) -> str:
            """Calculate the effective information cutoff date.
            Input: analysis_date
            Format: YYYYMMDD (e.g., 20230630)
            Returns: Cutoff dates for different document types.
            """
            return self._calculate_information_cutoff(analysis_date)
        
        @tool
        def check_document_availability(query_string: str) -> str:
            """Check if a specific document would be available at a point in time.
            Input: document_date|filing_type|analysis_date
            Example: 20230331|10-Q|20230501
            Returns: Availability status and explanation.
            """
            return self._check_document_availability(query_string)
        
        @tool
        def validate_date_range(query_string: str) -> str:
            """Validate a date range for temporal consistency.
            Input: start_date|end_date|analysis_date
            Format: YYYYMMDD|YYYYMMDD|YYYYMMDD
            Returns: Validation result and warnings.
            """
            return self._validate_date_range(query_string)
        
        @tool
        def detect_look_ahead_bias(query_string: str) -> str:
            """Analyze a query for potential look-ahead bias.
            Input: query|analysis_date
            Returns: Bias detection results and recommendations.
            """
            return self._detect_look_ahead_bias(query_string)
        
        self.tools = [calculate_information_cutoff, check_document_availability, 
                      validate_date_range, detect_look_ahead_bias]
        
        # Create ReAct agent using langgraph (works with any LLM)
        try:
            self.agent = create_react_agent(
                self.llm, 
                self.tools,
                prompt=self.SYSTEM_PROMPT
            )
            logger.info(f"Initialized Temporal Agent")
        except Exception as e:
            logger.warning(f"Could not create agent: {e}")
            self.agent = None
        
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string in various formats."""
        date_str = date_str.strip().replace('-', '').replace('/', '')
        
        # Try different formats with correct string lengths
        formats = [
            ('%Y%m%d', 8),   # 20230630
            ('%Y%m', 6),     # 202306
            ('%Y', 4)        # 2023
        ]
        
        for fmt, expected_len in formats:
            if len(date_str) >= expected_len:
                try:
                    return datetime.strptime(date_str[:expected_len], fmt)
                except ValueError:
                    continue
                
        raise ValueError(f"Could not parse date: {date_str}")
        
    def _calculate_information_cutoff(self, analysis_date: str) -> str:
        """Calculate effective information cutoff for each document type."""
        try:
            date = self._parse_date(analysis_date)
        except ValueError as e:
            return f"Error: {e}"
            
        cutoffs = {}
        
        for doc_type, lag_days in self.FILING_LAGS.items():
            # For a document to be available at analysis_date,
            # it must have been filed at least lag_days before analysis_date
            filing_cutoff = date - timedelta(days=lag_days)
            cutoffs[doc_type] = filing_cutoff.strftime('%Y-%m-%d')
            
        # Determine what quarters would be available
        available_quarters = self._get_available_quarters(date)
        
        return f"""INFORMATION CUTOFF ANALYSIS
Analysis Date: {date.strftime('%Y-%m-%d')}

Document Availability Cutoffs:
{'=' * 50}
- 10-K Annual Reports:  Filed on or before {cutoffs['10-K']}
  (Reflects fiscal years ended ~{(date - timedelta(days=self.FILING_LAGS['10-K'] + 60)).year})

- 10-Q Quarterly Reports: Filed on or before {cutoffs['10-Q']}
  (Reflects quarters ended ~40-45 days before filing)

- 8-K Current Reports: Filed on or before {cutoffs['8-K']}
  (Near real-time events)

- Earnings Transcripts: Filed on or before {cutoffs['earnings_call']}
  (1-2 days after earnings call)

- Stock Prices: As of {cutoffs['stock_price']}
  (End of day prices)

Available Quarterly Data:
{self._format_available_quarters(available_quarters)}

IMPORTANT: Only use documents filed BEFORE these dates to avoid look-ahead bias."""
    
    def _get_available_quarters(self, analysis_date: datetime) -> List[str]:
        """Determine which quarterly reports would be available."""
        available = []
        
        # Check last 8 quarters
        for i in range(8):
            # Calculate quarter end
            months_back = i * 3
            check_date = analysis_date - timedelta(days=months_back * 30)
            
            quarter = (check_date.month - 1) // 3 + 1
            year = check_date.year
            
            quarter_month, quarter_day = self.QUARTER_ENDS[quarter]
            quarter_end = datetime(year, quarter_month, quarter_day)
            
            # When would filing be available?
            filing_available = quarter_end + timedelta(days=self.FILING_LAGS["10-Q"])
            
            if filing_available < analysis_date:
                available.append(f"Q{quarter} {year}")
                
        return available[:6]  # Return last 6 available
        
    def _format_available_quarters(self, quarters: List[str]) -> str:
        """Format available quarters for display."""
        if not quarters:
            return "No quarterly data would be available."
        return "  " + ", ".join(quarters)
        
    def _check_document_availability(self, query_string: str) -> str:
        """Check if a document would be available at a given date."""
        parts = query_string.split('|')
        
        if len(parts) != 3:
            return "Error: Input must be document_date|filing_type|analysis_date"
            
        doc_date_str, filing_type, analysis_date_str = [p.strip() for p in parts]
        
        try:
            doc_date = self._parse_date(doc_date_str)
            analysis_date = self._parse_date(analysis_date_str)
        except ValueError as e:
            return f"Error parsing dates: {e}"
            
        filing_type = filing_type.upper()
        
        # Get filing lag
        if filing_type in ['10-K', '10K']:
            lag = self.FILING_LAGS['10-K']
            doc_type_name = "10-K Annual Report"
        elif filing_type in ['10-Q', '10Q']:
            lag = self.FILING_LAGS['10-Q']
            doc_type_name = "10-Q Quarterly Report"
        elif filing_type in ['8-K', '8K']:
            lag = self.FILING_LAGS['8-K']
            doc_type_name = "8-K Current Report"
        else:
            lag = 30  # Default assumption
            doc_type_name = f"Document ({filing_type})"
            
        # When would document be available?
        estimated_available = doc_date + timedelta(days=lag)
        
        is_available = estimated_available <= analysis_date
        days_difference = (analysis_date - estimated_available).days
        
        status = "✅ AVAILABLE" if is_available else "❌ NOT AVAILABLE"
        
        return f"""DOCUMENT AVAILABILITY CHECK
{'-' * 50}
Document: {doc_type_name}
Document Period End: {doc_date.strftime('%Y-%m-%d')}
Analysis Date: {analysis_date.strftime('%Y-%m-%d')}

Filing Lag (typical): {lag} days
Estimated Availability: {estimated_available.strftime('%Y-%m-%d')}

Status: {status}
{f'Available {abs(days_difference)} days before analysis date' if is_available else f'Would not be available for {abs(days_difference)} more days'}

{'This document CAN be used in analysis.' if is_available else '⚠️ LOOK-AHEAD BIAS: Using this document would be anachronistic!'}"""
    
    def _validate_date_range(self, query_string: str) -> str:
        """Validate date range for temporal consistency."""
        parts = query_string.split('|')
        
        if len(parts) != 3:
            return "Error: Input must be start_date|end_date|analysis_date"
            
        start_str, end_str, analysis_str = [p.strip() for p in parts]
        
        try:
            start_date = self._parse_date(start_str)
            end_date = self._parse_date(end_str)
            analysis_date = self._parse_date(analysis_str)
        except ValueError as e:
            return f"Error parsing dates: {e}"
            
        issues = []
        warnings = []
        
        # Check for basic validity
        if start_date > end_date:
            issues.append("Start date is after end date")
            
        # Check for look-ahead bias
        if end_date > analysis_date:
            issues.append(f"⚠️ LOOK-AHEAD BIAS: End date ({end_date.strftime('%Y-%m-%d')}) is after analysis date ({analysis_date.strftime('%Y-%m-%d')})")
            
        # Check if data for end date would actually be available
        implied_availability = end_date + timedelta(days=self.FILING_LAGS['10-Q'])
        if implied_availability > analysis_date:
            warnings.append(f"Data through {end_date.strftime('%Y-%m-%d')} may not be fully available at analysis date")
            
        # Analyze range
        range_days = (end_date - start_date).days
        
        status = "✅ VALID" if not issues else "❌ INVALID"
        
        result = f"""DATE RANGE VALIDATION
{'-' * 50}
Start Date: {start_date.strftime('%Y-%m-%d')}
End Date: {end_date.strftime('%Y-%m-%d')}
Analysis Date: {analysis_date.strftime('%Y-%m-%d')}
Range: {range_days} days

Status: {status}
"""
        
        if issues:
            result += f"\nISSUES:\n" + "\n".join(f"  - {i}" for i in issues)
        if warnings:
            result += f"\n\nWARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings)
            
        if not issues:
            result += "\n\nThis date range is temporally consistent for analysis."
            
        return result
        
    def _detect_look_ahead_bias(self, query_string: str,
                                allow_future_target: bool = False) -> str:
        """Detect potential look-ahead bias in a query."""
        parts = query_string.split('|')
        
        if len(parts) != 2:
            return "Error: Input must be query|analysis_date"
            
        query, analysis_date_str = parts
        query = query.strip()
        
        try:
            analysis_date = self._parse_date(analysis_date_str)
        except ValueError as e:
            return f"Error parsing date: {e}"
            
        analysis_year = analysis_date.year
        
        issues = []
        warnings = []
        
        # Detect future year references
        year_pattern = r'20\d{2}'
        found_years = re.findall(year_pattern, query)
        
        for year_str in found_years:
            year = int(year_str)
            if year > analysis_year and not allow_future_target:
                issues.append(f"Reference to future year {year} (analysis year: {analysis_year})")
            elif year == analysis_year:
                warnings.append(f"Reference to analysis year {year} - verify data availability")
                
        # Detect quarter references that may be future
        quarter_patterns = [
            r'Q([1-4])\s*20(\d{2})',
            r'(\d{4})\s*Q([1-4])',
        ]
        
        for pattern in quarter_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if pattern.startswith(r'Q'):
                    quarter, year = int(match[0]), int('20' + match[1])
                else:
                    year, quarter = int(match[0]), int(match[1])
                    
                # Calculate quarter end
                quarter_month, quarter_day = self.QUARTER_ENDS[quarter]
                quarter_end = datetime(year, quarter_month, quarter_day)
                
                # Q4 is reported in the annual 10-K; Q1-Q3 use 10-Q filings.
                filing_type = '10-K' if quarter == 4 else '10-Q'
                availability = quarter_end + timedelta(days=self.FILING_LAGS[filing_type])
                
                if availability > analysis_date and not (
                    allow_future_target and year > analysis_year
                ):
                    issues.append(f"Q{quarter} {year} data would not be available at analysis date")
                    
        # Detect suspicious keywords
        future_indicators = [
            'will', 'forecast', 'predict', 'expected',
            'guidance', 'outlook', 'projection', 'future'
        ]
        
        query_lower = query.lower()
        for indicator in future_indicators:
            if indicator in query_lower:
                warnings.append(f"Forward-looking keyword detected: '{indicator}'")
                
        status = "❌ BIAS DETECTED" if issues else "⚠️ WARNINGS" if warnings else "✅ CLEAN"
        
        result = f"""LOOK-AHEAD BIAS DETECTION
{'-' * 50}
Query: "{query}"
Analysis Date: {analysis_date.strftime('%Y-%m-%d')}

Status: {status}
"""
        
        if issues:
            result += f"\n🚨 LOOK-AHEAD BIAS ISSUES:\n" + "\n".join(f"  - {i}" for i in issues)
            result += "\n\n❌ This query contains temporal violations and should be modified."
            
        if warnings:
            result += f"\n⚠️ WARNINGS (review carefully):\n" + "\n".join(f"  - {w}" for w in warnings)
            
        if not issues and not warnings:
            result += "\n✅ No look-ahead bias detected. Query appears temporally consistent."
            
        return result
        
    def validate_query(self, query: str, analysis_date: str = None,
                       allow_future_target: bool = False) -> Dict:
        """
        Main method to validate a query for temporal consistency.
        
        Args:
            query: Query string to validate
            analysis_date: Analysis date in YYYYMMDD format (defaults to today)
            
        Returns:
            Validation results dictionary
        """
        # Default to today if not specified
        if analysis_date is None:
            analysis_date = datetime.now().strftime('%Y%m%d')

        normalized_date = str(analysis_date).strip().replace('-', '').replace('/', '')
        try:
            if len(normalized_date) != 8 or not normalized_date.isdigit():
                raise ValueError("Expected YYYYMMDD")
            datetime.strptime(normalized_date, '%Y%m%d')
            analysis_date = normalized_date
        except ValueError as e:
            message = f"❌ INVALID ANALYSIS DATE: {e}"
            return {
                'query': query,
                'analysis_date': str(analysis_date),
                'cutoff_analysis': message,
                'bias_detection': message,
                'has_violations': True,
                'has_warnings': False,
                'is_valid': False,
                'agent': 'TemporalAgent',
                'processing_time': 0.0,
                'timestamp': datetime.now().isoformat(),
            }
            
        start_time = datetime.now()
        
        # Run standard checks
        cutoff_result = self._calculate_information_cutoff(analysis_date)
        bias_result = self._detect_look_ahead_bias(
            f"{query}|{analysis_date}",
            allow_future_target=allow_future_target,
        )
        
        # Determine overall status
        has_violations = '❌' in bias_result or 'BIAS DETECTED' in bias_result
        has_warnings = '⚠️' in bias_result and not has_violations
        
        return {
            'query': query,
            'analysis_date': analysis_date,
            'cutoff_analysis': cutoff_result,
            'bias_detection': bias_result,
            'has_violations': has_violations,
            'has_warnings': has_warnings,
            'is_valid': not has_violations,
            'agent': 'TemporalAgent',
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
        
    def get_cutoff_date(self, analysis_date: str, filing_type: str = "10-Q") -> str:
        """
        Get the effective cutoff date for a filing type.
        
        Args:
            analysis_date: Analysis date (YYYYMMDD)
            filing_type: Type of filing
            
        Returns:
            Cutoff date string
        """
        try:
            date = self._parse_date(analysis_date)
        except ValueError:
            return analysis_date
            
        lag = self.FILING_LAGS.get(filing_type, 30)
        cutoff = date - timedelta(days=lag)
        
        return cutoff.strftime('%Y%m%d')


# Usage
if __name__ == "__main__":
    # Initialize agent
    agent = TemporalAgent()
    
    # Test query validation
    result = agent.validate_query(
        query="What was Apple's revenue growth in Q2 2024?",
        analysis_date="20240115"
    )
    
    print("\nTemporalAgent Analysis:")
    print("=" * 60)
    print(result['cutoff_analysis'])
    print("=" * 60)
    print(result['bias_detection'])
    print("=" * 60)
    print(f"Valid: {result['is_valid']}")
