"""
Calculation Agent for TemporalGuard-RAG

Specialized agent for financial calculations using structured XBRL data.
Computes ratios, growth rates, and other metrics with full transparency.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pandas as pd

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


class CalculationAgent:
    """
    Financial calculation agent using XBRL structured data.
    
    Key Responsibilities:
    - Calculate financial ratios
    - Compute growth rates
    - Verify numerical accuracy
    - Show calculation steps
    """
    
    SYSTEM_PROMPT = """You are a financial calculation specialist working with structured XBRL data.

Your job is to compute accurate financial metrics from structured data while showing your work.

CRITICAL RULES:
1. Always use XBRL structured data for calculations, not narrative text
2. Show all calculation steps clearly
3. Cite data sources with exact dates
4. Flag any missing or uncertain data
5. Use appropriate formulas for financial ratios
6. Round results appropriately (typically 2 decimal places for ratios, percentages)

Common Financial Ratios:
- ROE (Return on Equity) = Net Income / Shareholders' Equity
- ROA (Return on Assets) = Net Income / Total Assets
- Debt Ratio = Total Liabilities / Total Assets
- Profit Margin = Net Income / Revenue
- Revenue Growth = (Current Revenue - Prior Revenue) / Prior Revenue × 100

When providing results:
- Show the formula used
- List input values with their filing dates
- Show the calculation
- Provide the final result with appropriate units
"""

    # Supported ratio calculations
    RATIO_FORMULAS = {
        'ROE': {
            'formula': 'Net Income / Shareholders Equity',
            'numerator': 'NetIncome',
            'denominator': 'Equity'
        },
        'ROA': {
            'formula': 'Net Income / Total Assets',
            'numerator': 'NetIncome',
            'denominator': 'TotalAssets'
        },
        'debt_ratio': {
            'formula': 'Total Liabilities / Total Assets',
            'numerator': 'TotalLiabilities',
            'denominator': 'TotalAssets'
        },
        'profit_margin': {
            'formula': 'Net Income / Revenue',
            'numerator': 'NetIncome',
            'denominator': 'Revenue'
        },
        'current_ratio': {
            'formula': 'Current Assets / Current Liabilities',
            'numerator': 'CurrentAssets',
            'denominator': 'CurrentLiabilities'
        }
    }
    
    def __init__(self, 
                 xbrl_dir: str = "data/raw/xbrl_structured",
                 model_name: str = None,
                 provider: str = None):
        """
        Initialize Calculation Agent.
        
        Args:
            xbrl_dir: Directory containing XBRL data files
            model_name: LLM model name (default: auto-select)
            provider: 'openai' or 'ollama' (default: auto-detect)
        """
        self.xbrl_dir = Path(xbrl_dir)
        self.model_name = model_name
        self.provider = provider
        
        # Initialize LLM using provider factory
        self.llm = get_llm(provider=provider, model_name=model_name, temperature=0)
        
        # Create tools using decorator (langgraph style)
        @tool
        def calculate_financial_ratio(query_string: str) -> str:
            """Calculate financial ratios from XBRL data.
            Input: ratio_name|ticker|date
            Supported ratios: ROE, ROA, debt_ratio, profit_margin, current_ratio, revenue_growth
            Example: ROE|AAPL|20230630
            """
            return self._calculate_ratio(query_string)
        
        @tool
        def load_xbrl_data(ticker: str) -> str:
            """Load structured financial data for a company.
            Input: ticker
            Example: AAPL
            """
            return self._load_xbrl_data(ticker)
        
        @tool
        def get_metric_value(query_string: str) -> str:
            """Get a specific financial metric value.
            Input: ticker|metric|date
            Metrics: Revenue, NetIncome, TotalAssets, TotalLiabilities, Equity
            Example: AAPL|Revenue|20230630
            """
            return self._get_metric_value(query_string)
        
        self.tools = [calculate_financial_ratio, load_xbrl_data, get_metric_value]
        
        # Create ReAct agent using langgraph (works with any LLM)
        try:
            self.agent = create_react_agent(
                self.llm, 
                self.tools,
                prompt=self.SYSTEM_PROMPT
            )
            logger.info(f"Initialized Calculation Agent")
        except Exception as e:
            logger.warning(f"Could not create agent: {e}")
            self.agent = None
        
    def _load_xbrl_data(self, ticker: str) -> str:
        """Load XBRL financial data for a company."""
        ticker = ticker.strip().upper()
        
        # Try to load facts file
        facts_path = self.xbrl_dir / f"{ticker}_facts.json"
        metrics_path = self.xbrl_dir / f"{ticker}_metrics.csv"
        
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                
                # Get summary of available data
                metrics = df['metric'].unique().tolist()
                date_range = f"{df['end_date'].min()} to {df['end_date'].max()}"
                
                return f"""XBRL Data loaded for {ticker}:
- Available metrics: {', '.join(metrics)}
- Date range: {date_range}
- Total records: {len(df)}

Use get_metric_value to retrieve specific values."""
                
            except Exception as e:
                return f"Error loading metrics data: {e}"
                
        elif facts_path.exists():
            try:
                with open(facts_path, 'r') as f:
                    data = json.load(f)
                    
                return f"XBRL facts loaded for {ticker}. Use get_metric_value to retrieve specific values."
                
            except Exception as e:
                return f"Error loading facts data: {e}"
        else:
            return self._mock_xbrl_data(ticker)
            
    def _mock_xbrl_data(self, ticker: str) -> str:
        """Return mock XBRL data for testing."""
        return f"""[MOCK DATA] XBRL Data for {ticker}:
- Revenue (2023-03-31): $94,836,000,000
- Revenue (2022-03-31): $97,278,000,000
- Net Income (2023-03-31): $24,160,000,000
- Net Income (2022-03-31): $25,010,000,000
- Total Assets (2023-03-31): $346,747,000,000
- Total Liabilities (2023-03-31): $290,020,000,000
- Shareholders' Equity (2023-03-31): $56,727,000,000

Note: This is mock data. Connect XBRL collector for real data.
[WARNING: MOCK DATA USED - Results are for demonstration only]"""
            
    def _get_metric_value(self, query_string: str) -> str:
        """Get specific metric value."""
        parts = query_string.split('|')
        
        if len(parts) != 3:
            return "Error: Input must be ticker|metric|date"
            
        ticker, metric, date = [p.strip() for p in parts]
        ticker = ticker.upper()
        
        metrics_path = self.xbrl_dir / f"{ticker}_metrics.csv"
        
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                df['end_date'] = pd.to_datetime(df['end_date'])
                if 'filed_date' not in df.columns:
                    return f"No {metric} data with filing-date provenance found for {ticker}"
                df['filed_date'] = pd.to_datetime(df['filed_date'], errors='coerce')
                
                target_date = pd.to_datetime(date)
                
                # Filter by metric and date (on or before)
                filtered = df[
                    (df['metric'] == metric) & 
                    (df['end_date'] <= target_date) &
                    (df['filed_date'] <= target_date)
                ].sort_values(['end_date', 'filed_date'], ascending=False)
                
                if filtered.empty:
                    return f"No {metric} data found for {ticker} before {date}"
                    
                latest = filtered.iloc[0]
                
                return f"""{metric} for {ticker}:
- Value: ${latest['value']:,.0f}
- As of: {latest['end_date'].strftime('%Y-%m-%d')}
- Filed: {latest.get('filed_date', 'Unknown')}
- Form: {latest.get('form', 'Unknown')}"""
                
            except Exception as e:
                return f"Error retrieving metric: {e}"
        else:
            return self._mock_metric_value(ticker, metric, date)
            
    def _mock_metric_value(self, ticker: str, metric: str, date: str) -> str:
        """Return mock metric value."""
        mock_values = {
            'Revenue': 94836000000,
            'NetIncome': 24160000000,
            'TotalAssets': 346747000000,
            'TotalLiabilities': 290020000000,
            'Equity': 56727000000,
            'CurrentAssets': 135405000000,
            'CurrentLiabilities': 153982000000
        }
        
        value = mock_values.get(metric, 10000000000)
        
        return f"""[MOCK] {metric} for {ticker}:
- Value: ${value:,.0f}
- As of: {date[:4]}-{date[4:6]}-{date[6:]}
- Source: Mock XBRL data
[WARNING: MOCK DATA USED - Results are for demonstration only]"""
        
    def _calculate_ratio(self, query_string: str) -> str:
        """Calculate financial ratio."""
        parts = query_string.split('|')
        
        if len(parts) != 3:
            return "Error: Input must be ratio_name|ticker|date"
            
        ratio_name, ticker, date = [p.strip() for p in parts]
        ticker = ticker.upper()
        
        # Handle revenue growth specially
        if ratio_name.lower() == 'revenue_growth':
            return self._calculate_growth(ticker, 'Revenue', date)
            
        # Get ratio formula
        if ratio_name not in self.RATIO_FORMULAS:
            return f"Error: Unknown ratio '{ratio_name}'. Supported: {list(self.RATIO_FORMULAS.keys())}"
            
        formula = self.RATIO_FORMULAS[ratio_name]
        
        # Get component values
        num_result = self._get_metric_value(f"{ticker}|{formula['numerator']}|{date}")
        den_result = self._get_metric_value(f"{ticker}|{formula['denominator']}|{date}")

        used_mock = (
            "[WARNING: MOCK DATA USED" in num_result
            or "[WARNING: MOCK DATA USED" in den_result
            or num_result.strip().startswith("[MOCK")
            or den_result.strip().startswith("[MOCK")
        )
        
        # Extract values (simplified parsing)
        try:
            # Parse numerator
            num_line = [l for l in num_result.split('\n') if 'Value:' in l][0]
            num_value = float(num_line.split('$')[1].split()[0].replace(',', ''))
            
            # Parse denominator
            den_line = [l for l in den_result.split('\n') if 'Value:' in l][0]
            den_value = float(den_line.split('$')[1].split()[0].replace(',', ''))
            
            # Calculate ratio
            if den_value == 0:
                return f"Error: Cannot calculate {ratio_name} - denominator is zero"
                
            ratio = num_value / den_value
            percentage = ratio * 100
            
            output = f"""Financial Ratio Calculation for {ticker}:

Ratio: {ratio_name}
Formula: {formula['formula']}

Inputs:
- {formula['numerator']}: ${num_value:,.0f}
- {formula['denominator']}: ${den_value:,.0f}
- As of date: {date}

Calculation:
{num_value:,.0f} / {den_value:,.0f} = {ratio:.4f}

Result: {ratio_name} = {percentage:.2f}%"""

            if used_mock:
                output += "\n[WARNING: MOCK DATA USED - Results are for demonstration only]"

            return output
            
        except Exception as e:
            return f"Error calculating ratio: {e}\n\nNumerator data:\n{num_result}\n\nDenominator data:\n{den_result}"
            
    def _calculate_growth(self, ticker: str, metric: str, date: str) -> str:
        """Calculate growth rate for a metric."""
        # Current period
        current_result = self._get_metric_value(f"{ticker}|{metric}|{date}")
        
        # Prior period (1 year ago)
        try:
            prior_date = f"{int(date[:4])-1}{date[4:]}"
        except:
            return "Error: Invalid date format"
            
        prior_result = self._get_metric_value(f"{ticker}|{metric}|{prior_date}")
        
        try:
            # Parse current
            curr_line = [l for l in current_result.split('\n') if 'Value:' in l][0]
            curr_value = float(curr_line.split('$')[1].split()[0].replace(',', ''))
            
            # Parse prior
            prior_line = [l for l in prior_result.split('\n') if 'Value:' in l][0]
            prior_value = float(prior_line.split('$')[1].split()[0].replace(',', ''))
            
            if prior_value == 0:
                return f"Error: Cannot calculate growth - prior period value is zero"
                
            growth = ((curr_value - prior_value) / prior_value) * 100
            
            return f"""{metric} Growth Calculation for {ticker}:

Current Period ({date}): ${curr_value:,.0f}
Prior Period ({prior_date}): ${prior_value:,.0f}

Calculation:
({curr_value:,.0f} - {prior_value:,.0f}) / {prior_value:,.0f} × 100

Result: {metric} Growth = {growth:.2f}%"""
            
        except Exception as e:
            return f"Error calculating growth: {e}"
            
    def calculate(self, metric: str, ticker: str, date: str) -> Dict:
        """
        Main calculation method.
        
        Args:
            metric: Ratio or metric to calculate
            ticker: Company ticker symbol
            date: As-of date (YYYYMMDD)
            
        Returns:
            Dictionary with calculation results
        """
        start_time = datetime.now()
        
        if self.agent:
            try:
                input_message = f"""Calculate {metric} for {ticker} as of {date}.
                    
Show all calculation steps and cite data sources."""

                result = self.agent.invoke({
                    "messages": [HumanMessage(content=input_message)]
                })
                
                # Extract output from messages
                output = result.get('messages', [])[-1].content if result.get('messages') else ''
                
                return {
                    'output': output,
                    'metric': metric,
                    'ticker': ticker,
                    'date': date,
                    'agent': 'CalculationAgent',
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Agent error: {e}")
                return self._fallback_calculate(metric, ticker, date)
        else:
            return self._fallback_calculate(metric, ticker, date)
            
    def _fallback_calculate(self, metric: str, ticker: str, date: str) -> Dict:
        """Fallback calculation without agent."""
        result = self._calculate_ratio(f"{metric}|{ticker}|{date}")
        
        return {
            'output': result,
            'metric': metric,
            'ticker': ticker,
            'date': date,
            'agent': 'CalculationAgent',
            'mode': 'fallback',
            'timestamp': datetime.now().isoformat()
        }


# Usage
if __name__ == "__main__":
    # Initialize agent
    agent = CalculationAgent()
    
    # Test calculation
    result = agent.calculate(
        metric="ROE",
        ticker="AAPL",
        date="20230630"
    )
    
    print("\nCalculation Result:")
    print("=" * 60)
    print(result['output'])
    print("=" * 60)
