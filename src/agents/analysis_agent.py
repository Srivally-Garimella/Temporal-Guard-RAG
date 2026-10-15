"""
Financial Analysis Agent for TemporalGuard-RAG

A comprehensive agent that handles:
- Projections and forecasting (projected cash flow for 2027)
- Historical analysis (when was highest EPS?)
- Financial calculations (profit margin, ROE, etc.)
- Valuation analysis (DCF, comparable analysis)
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from contextvars import ContextVar
import json
import logging
import re

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import warnings

from .llm_provider import get_llm
from ..analysis.formulas import FinancialFormulas, list_formulas
from ..analysis.projections import ProjectionEngine, Projection
from ..analysis.historical import HistoricalAnalyzer
from ..analysis.data_loader import FinancialDataLoader, StockDataLoader

warnings.filterwarnings('ignore', category=DeprecationWarning, module='langgraph')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialAnalysisAgent:
    """
    Comprehensive financial analysis agent.
    
    Capabilities:
    - Project future metrics (revenue, FCF, EPS, etc.)
    - Find historical extremes (highest/lowest values)
    - Calculate financial ratios
    - Perform trend analysis
    - DCF valuation
    """
    
    SYSTEM_PROMPT = """You are a senior financial analyst with expertise in:
- Financial statement analysis (Balance Sheet, Income Statement, Cash Flow)
- Financial projections and forecasting
- Valuation methods (DCF, multiples, comparables)
- Historical trend analysis

You have access to tools that can:
1. Load comprehensive financial data from SEC XBRL filings
2. Project future values using various methods (CAGR, linear trend, growth rate)
3. Find historical maxima/minima
4. Calculate financial ratios and metrics
5. Get current stock market data

ANALYSIS GUIDELINES:
1. Always cite data sources and dates
2. For projections, clearly state assumptions used
3. Provide confidence levels: HIGH/MEDIUM/LOW
4. When data is insufficient, acknowledge limitations
5. For future projections, use multiple methods when possible

PROJECTION METHODS:
- CAGR-based: Good for stable, historical trends
- Linear regression: Good for consistent patterns
- Growth rate: Good when using analyst estimates
- Scenario analysis: Provide bull/base/bear cases

When asked about future metrics:
1. Load historical data
2. Analyze the trend
3. Apply appropriate projection method
4. Provide range estimates (confidence interval)
5. Note key assumptions and risks
"""

    def __init__(self,
                 xbrl_dir: str = "data/raw/xbrl_structured",
                 model_name: str = None,
                 provider: str = None):
        """Initialize the Financial Analysis Agent."""
        self.xbrl_dir = xbrl_dir
        self.model_name = model_name
        self.provider = provider
        
        # Initialize components
        self.data_loader = FinancialDataLoader(xbrl_dir)
        self.stock_loader = StockDataLoader()
        self.projection_engine = ProjectionEngine()
        self.historical_analyzer = HistoricalAnalyzer(xbrl_dir)
        self.formulas = FinancialFormulas()
        
        # Initialize LLM
        self.llm = get_llm(provider=provider, model_name=model_name, temperature=0)

        # Request-scoped context (set per analyze call)
        self._analysis_date_context: ContextVar[Optional[str]] = ContextVar(
            f"analysis_date_{id(self)}",
            default=None,
        )
        
        # Create tools
        self._setup_tools()
        
        # Create agent
        try:
            self.agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=self.SYSTEM_PROMPT
            )
            logger.info("Initialized Financial Analysis Agent")
        except Exception as e:
            logger.warning(f"Could not create agent: {e}")
            self.agent = None

    @property
    def _analysis_date(self) -> Optional[str]:
        return self._analysis_date_context.get()

    @_analysis_date.setter
    def _analysis_date(self, value: Optional[str]) -> None:
        self._analysis_date_context.set(value)
    
    def _setup_tools(self):
        """Setup tools for the agent."""
        
        @tool
        def load_financial_data(ticker: str) -> str:
            """Load all financial data for a company from SEC XBRL filings.
            Input: ticker symbol (e.g., AAPL, MSFT)
            Returns: Summary of available financial data including Balance Sheet, 
                    Income Statement, Cash Flow metrics.
            """
            return self._load_financial_data(ticker)
        
        @tool
        def project_metric(query_string: str) -> str:
            """Project a financial metric to a future year.
            Input format: ticker|metric|target_year
            Examples: 
              AAPL|Revenue|2027
              MSFT|FreeCashFlow|2028
              JPM|NetIncome|2026
            Supported metrics: Revenue, NetIncome, EPS, FreeCashFlow, OperatingCashFlow
            """
            return self._project_metric(query_string)
        
        @tool
        def find_historical_extreme(query_string: str) -> str:
            """Find when a metric was at its highest or lowest.
            Input format: ticker|metric|type
            type can be: maximum, minimum
            Examples:
              AAPL|EPS_Diluted|maximum
              MSFT|Revenue|maximum
              JPM|NetIncome|minimum
            """
            return self._find_extreme(query_string)
        
        @tool
        def calculate_ratio(query_string: str) -> str:
            """Calculate a financial ratio for a company.
            Input format: ticker|ratio_name
            Supported ratios: ROE, ROA, NetMargin, CurrentRatio, DebtToEquity, 
                            GrossMargin, FCFMargin, QuickRatio, DebtRatio
            Example: AAPL|ROE
            """
            return self._calculate_ratio(query_string)
        
        @tool
        def analyze_trend(query_string: str) -> str:
            """Analyze the historical trend of a metric.
            Input format: ticker|metric
            Example: AAPL|Revenue
            Returns: Trend direction, strength, CAGR, volatility analysis
            """
            return self._analyze_trend(query_string)
        
        @tool
        def scenario_projection(query_string: str) -> str:
            """Generate bull/base/bear case projections.
            Input format: ticker|metric|target_year
            Example: AAPL|Revenue|2027
            Returns projections under optimistic, base, and pessimistic scenarios.
            """
            return self._scenario_projection(query_string)
        
        @tool
        def dcf_valuation(ticker: str) -> str:
            """Perform a DCF valuation for a company.
            Input: ticker symbol
            Returns: Fair value estimate based on discounted cash flow analysis.
            """
            return self._dcf_valuation(ticker)
        
        @tool
        def get_market_data(ticker: str) -> str:
            """Get current stock market data including price, P/E, analyst estimates.
            Input: ticker symbol
            Returns: Current price, market cap, P/E ratios, analyst targets
            """
            return self._get_market_data(ticker)
        
        @tool
        def list_available_metrics(ticker: str) -> str:
            """List all available financial metrics for a company.
            Input: ticker symbol
            Returns: List of metrics that can be analyzed or projected.
            """
            return self._list_metrics(ticker)
        
        self.tools = [
            load_financial_data,
            project_metric,
            find_historical_extreme,
            calculate_ratio,
            analyze_trend,
            scenario_projection,
            dcf_valuation,
            get_market_data,
            list_available_metrics
        ]

    def _filter_history_asof(self, history: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        return [(int(str(year)[:4]), value) for year, value in history]

    def _latest_value_asof(self, ticker: str, metric: str) -> Tuple[Optional[int], Optional[float]]:
        history_raw = self.data_loader.get_metric_history(
            ticker, metric, as_of_date=self._analysis_date
        )
        history = [(int(y), v) for y, v in history_raw] if history_raw else []
        history = self._filter_history_asof(history)
        if not history:
            return None, None
        year, value = max(history, key=lambda x: x[0])
        return year, value
    
    # ═══════════════════════════════════════════════════════════════════
    # TOOL IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def _load_financial_data(self, ticker: str) -> str:
        """Load and summarize financial data."""
        ticker = ticker.strip().upper()
        
        try:
            summary = self.data_loader.get_financial_summary(
                ticker, as_of_date=self._analysis_date
            )
            
            lines = [
                f"📊 Financial Data for {summary.get('company', ticker)}",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"Available Metrics: {summary.get('available_metrics', 0)}",
                "",
                "📈 Latest Key Figures:"
            ]
            
            for metric, info in summary.get("latest_data", {}).items():
                value = info.get("value", 0)
                year = info.get("year", "N/A")
                if abs(value) >= 1e9:
                    value_str = f"${value/1e9:.2f}B"
                elif abs(value) >= 1e6:
                    value_str = f"${value/1e6:.2f}M"
                else:
                    value_str = f"{value:,.2f}"
                lines.append(f"  • {metric}: {value_str} ({year})")
            
            lines.append("")
            lines.append("📊 Key Ratios:")
            for ratio, info in summary.get("ratios", {}).items():
                value = info.get("value", 0)
                lines.append(f"  • {ratio}: {value:.2f}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error loading data for {ticker}: {e}"
    
    def _project_metric(self, query_string: str) -> str:
        """Project a metric to a future year."""
        parts = query_string.split("|")
        if len(parts) != 3:
            return "Error: Input must be ticker|metric|target_year (e.g., AAPL|Revenue|2027)"
        
        ticker, metric, target_year = [p.strip() for p in parts]
        ticker = ticker.upper()
        
        try:
            target_year = int(target_year)
        except ValueError:
            return f"Error: Invalid year '{target_year}'"
        
        # Map common metric names
        metric_map = {
            "revenue": "Revenue",
            "netincome": "NetIncome",
            "net income": "NetIncome",
            "eps": "EPS_Diluted",
            "earnings": "NetIncome",
            "fcf": "FreeCashFlow",
            "free cash flow": "FreeCashFlow",
            "freecashflow": "FreeCashFlow",
            "cash flow": "OperatingCashFlow",
            "cashflow": "OperatingCashFlow",
            "operating cash flow": "OperatingCashFlow",
            "operatingcashflow": "OperatingCashFlow",
            "ocf": "OperatingCashFlow",
        }
        
        std_metric = metric_map.get(metric.lower(), metric)
        
        # Load historical data
        history_raw = self.data_loader.get_metric_history(
            ticker, std_metric, as_of_date=self._analysis_date
        )
        history = [(int(y), v) for y, v in history_raw] if history_raw else []
        history = self._filter_history_asof(history)
        
        if not history:
            # Try to calculate FCF from OCF and CapEx
            if std_metric in ["FreeCashFlow", "FCF"]:
                ocf_history_raw = self.data_loader.get_metric_history(
                    ticker, "OperatingCashFlow", as_of_date=self._analysis_date
                )
                capex_history_raw = self.data_loader.get_metric_history(
                    ticker, "CapitalExpenditures", as_of_date=self._analysis_date
                )
                ocf_history = self._filter_history_asof([(int(y), v) for y, v in ocf_history_raw]) if ocf_history_raw else []
                capex_history = self._filter_history_asof([(int(y), v) for y, v in capex_history_raw]) if capex_history_raw else []
                
                if ocf_history and capex_history:
                    # Calculate FCF = OCF - CapEx
                    history = []
                    capex_dict = dict(capex_history)
                    for year, ocf in ocf_history:
                        if year in capex_dict:
                            fcf = ocf - abs(capex_dict[year])
                            history.append((year, fcf))
        
        if not history:
            return f"Error: No historical data found for {std_metric} for {ticker}. " \
                   f"Try 'list_available_metrics' to see what's available."
        
        # Generate projections using multiple methods
        cagr_proj = self.projection_engine.cagr_projection(
            [(int(y), v) for y, v in history],
            target_year,
            std_metric
        )
        
        linear_proj = self.projection_engine.linear_trend_projection(
            [(int(y), v) for y, v in history],
            target_year,
            std_metric
        )
        
        # Format output
        lines = [
            f"📊 {std_metric} Projection for {ticker}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Target Year: {target_year}",
            "",
            "📈 Historical Data:",
        ]
        
        for year, value in sorted(history)[-5:]:  # Last 5 years
            if abs(value) >= 1e9:
                val_str = f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                val_str = f"${value/1e6:.2f}M"
            else:
                val_str = f"${value:,.2f}"
            lines.append(f"  {year}: {val_str}")
        
        lines.extend([
            "",
            "🎯 Projections:",
            "",
            f"Method 1: CAGR-Based ({cagr_proj.confidence_level} confidence)",
        ])
        
        if abs(cagr_proj.projected_value) >= 1e9:
            proj_str = f"${cagr_proj.projected_value/1e9:.2f}B"
        elif abs(cagr_proj.projected_value) >= 1e6:
            proj_str = f"${cagr_proj.projected_value/1e6:.2f}M"
        else:
            proj_str = f"${cagr_proj.projected_value:,.2f}"
        
        lines.append(f"  Projected {std_metric}: {proj_str}")
        lines.append(f"  {cagr_proj.explanation}")
        
        if cagr_proj.confidence_interval:
            ci_low, ci_high = cagr_proj.confidence_interval
            lines.append(f"  Range: ${ci_low/1e9:.2f}B to ${ci_high/1e9:.2f}B")
        
        lines.extend([
            "",
            f"Method 2: Linear Trend ({linear_proj.confidence_level} confidence)",
        ])
        
        if abs(linear_proj.projected_value) >= 1e9:
            proj_str2 = f"${linear_proj.projected_value/1e9:.2f}B"
        else:
            proj_str2 = f"${linear_proj.projected_value/1e6:.2f}M"
        
        lines.append(f"  Projected {std_metric}: {proj_str2}")
        lines.append(f"  {linear_proj.explanation}")
        
        lines.extend([
            "",
            "⚠️ Assumptions & Caveats:",
            f"  • Based on {len(history)} years of historical data",
            "  • Assumes historical growth patterns continue",
            "  • Does not account for economic cycles or disruptions",
            "  • Actual results may vary significantly",
        ])
        
        return "\n".join(lines)
    
    def _find_extreme(self, query_string: str) -> str:
        """Find historical maximum or minimum."""
        parts = query_string.split("|")
        if len(parts) != 3:
            return "Error: Input must be ticker|metric|type (e.g., AAPL|EPS_Diluted|maximum)"
        
        ticker, metric, extreme_type = [p.strip() for p in parts]
        ticker = ticker.upper()
        extreme_type = extreme_type.lower()
        
        # Map metric names
        metric_map = {
            "eps": "EPS_Diluted",
            "revenue": "Revenue",
            "netincome": "NetIncome",
            "profit": "NetIncome",
        }
        std_metric = metric_map.get(metric.lower(), metric)
        
        # Load data
        history_raw = self.data_loader.get_metric_history(
            ticker, std_metric, as_of_date=self._analysis_date
        )
        history = [(int(y), v) for y, v in history_raw] if history_raw else []
        history = self._filter_history_asof(history)
        
        if not history:
            return f"No historical data found for {std_metric} for {ticker}"
        
        if extreme_type == "maximum":
            finding = self.historical_analyzer.find_maximum(history, std_metric, ticker)
        elif extreme_type == "minimum":
            finding = self.historical_analyzer.find_minimum(history, std_metric, ticker)
        else:
            return f"Unknown type '{extreme_type}'. Use 'maximum' or 'minimum'"
        
        return self.historical_analyzer.format_finding(finding)
    
    def _calculate_ratio(self, query_string: str) -> str:
        """Calculate a financial ratio."""
        parts = query_string.split("|")
        if len(parts) != 2:
            return "Error: Input must be ticker|ratio_name (e.g., AAPL|ROE)"
        
        ticker, ratio_name = [p.strip() for p in parts]
        ticker = ticker.upper()
        
        def get_latest_value(metric: str) -> float:
            _, value = self._latest_value_asof(ticker, metric)
            return float(value or 0.0)
        
        ratio_lower = ratio_name.lower()
        
        try:
            if ratio_lower == "roe":
                net_income = get_latest_value("NetIncome")
                equity = get_latest_value("StockholdersEquity")
                result = self.formulas.return_on_equity(net_income, equity)
            elif ratio_lower == "roa":
                net_income = get_latest_value("NetIncome")
                assets = get_latest_value("Assets")
                result = self.formulas.return_on_assets(net_income, assets)
            elif ratio_lower in ["netmargin", "net margin", "profit margin"]:
                net_income = get_latest_value("NetIncome")
                revenue = get_latest_value("Revenue")
                result = self.formulas.net_profit_margin(net_income, revenue)
            elif ratio_lower in ["currentratio", "current ratio"]:
                current_assets = get_latest_value("CurrentAssets")
                current_liab = get_latest_value("CurrentLiabilities")
                result = self.formulas.current_ratio(current_assets, current_liab)
            elif ratio_lower in ["debttoequity", "debt to equity", "d/e"]:
                debt = get_latest_value("LongTermDebt") or get_latest_value("Liabilities")
                equity = get_latest_value("StockholdersEquity")
                result = self.formulas.debt_to_equity(debt, equity)
            elif ratio_lower in ["grossmargin", "gross margin"]:
                revenue = get_latest_value("Revenue")
                cogs = get_latest_value("CostOfRevenue")
                result = self.formulas.gross_profit_margin(revenue, cogs)
            elif ratio_lower in ["fcfmargin", "fcf margin"]:
                ocf = get_latest_value("OperatingCashFlow")
                capex = abs(get_latest_value("CapitalExpenditures") or 0)
                revenue = get_latest_value("Revenue")
                fcf = ocf - capex
                result = self.formulas.fcf_margin(fcf, revenue)
            else:
                return f"Unknown ratio '{ratio_name}'. Supported: ROE, ROA, NetMargin, CurrentRatio, DebtToEquity, GrossMargin, FCFMargin"
            
            lines = [
                f"📊 {result.name} for {ticker}",
                f"━━━━━━━━━━━━━━━━━━━━━━━",
                f"Value: {result.value:.2f}{' %' if result.unit == 'percent' else 'x' if result.unit == 'ratio' else ''}",
                f"Formula: {result.formula}",
                "",
                "Inputs:",
            ]
            for key, val in result.inputs.items():
                if abs(val) >= 1e9:
                    val_str = f"${val/1e9:.2f}B"
                elif abs(val) >= 1e6:
                    val_str = f"${val/1e6:.2f}M"
                else:
                    val_str = f"{val:,.2f}"
                lines.append(f"  • {key}: {val_str}")
            
            lines.extend([
                "",
                f"💡 {result.interpretation}",
                f"Confidence: {result.confidence}"
            ])
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error calculating {ratio_name} for {ticker}: {e}"
    
    def _analyze_trend(self, query_string: str) -> str:
        """Analyze trend for a metric."""
        parts = query_string.split("|")
        if len(parts) != 2:
            return "Error: Input must be ticker|metric (e.g., AAPL|Revenue)"
        
        ticker, metric = [p.strip() for p in parts]
        ticker = ticker.upper()
        
        history_raw = self.data_loader.get_metric_history(
            ticker, metric, as_of_date=self._analysis_date
        )
        history = [(int(y), v) for y, v in history_raw] if history_raw else []
        history = self._filter_history_asof(history)
        
        if not history:
            return f"No data found for {metric} for {ticker}"
        
        finding = self.historical_analyzer.analyze_trend(history, metric, ticker)
        return self.historical_analyzer.format_finding(finding)
    
    def _scenario_projection(self, query_string: str) -> str:
        """Generate scenario-based projections."""
        parts = query_string.split("|")
        if len(parts) != 3:
            return "Error: Input must be ticker|metric|target_year"
        
        ticker, metric, target_year = [p.strip() for p in parts]
        ticker = ticker.upper()
        
        try:
            target_year = int(target_year)
        except ValueError:
            return f"Invalid year: {target_year}"
        
        # Load data
        history_raw = self.data_loader.get_metric_history(
            ticker, metric, as_of_date=self._analysis_date
        )
        history = [(int(y), v) for y, v in history_raw] if history_raw else []
        history = self._filter_history_asof(history)
        
        if not history:
            return f"No data found for {metric} for {ticker}"
        
        # Get the most recent value
        base_year, base_value = max(history, key=lambda x: x[0])
        
        # Calculate historical CAGR for base case
        trend = self.historical_analyzer.analyze_trend(history, metric, ticker)
        historical_cagr = trend.context.get("cagr", 5) / 100
        
        # Define scenarios
        scenario = self.projection_engine.scenario_projection(
            base_value=base_value,
            base_year=int(base_year),
            target_year=target_year,
            metric_name=metric,
            base_growth=historical_cagr,
            bull_growth=historical_cagr * 1.5,
            bear_growth=max(0, historical_cagr * 0.5)
        )
        
        return self.projection_engine.format_scenario(scenario)
    
    def _dcf_valuation(self, ticker: str) -> str:
        """Perform DCF valuation."""
        ticker = ticker.strip().upper()
        
        # Get required data
        ocf_history_raw = self.data_loader.get_metric_history(
            ticker, "OperatingCashFlow", as_of_date=self._analysis_date
        )
        capex_history_raw = self.data_loader.get_metric_history(
            ticker, "CapitalExpenditures", as_of_date=self._analysis_date
        )
        ocf_history = self._filter_history_asof([(int(y), v) for y, v in ocf_history_raw]) if ocf_history_raw else []
        capex_history = self._filter_history_asof([(int(y), v) for y, v in capex_history_raw]) if capex_history_raw else []
        
        if not ocf_history:
            return f"Insufficient data for DCF valuation of {ticker}"
        
        # Calculate current FCF
        latest_ocf = ocf_history[-1][1] if ocf_history else 0
        latest_capex = abs(capex_history[-1][1]) if capex_history else latest_ocf * 0.15
        current_fcf = latest_ocf - latest_capex
        
        # Get market data for shares outstanding
        market_data = (
            self.stock_loader.get_current_price(ticker)
            if not self._analysis_date or self._analysis_date == datetime.now().strftime("%Y%m%d")
            else {}
        )
        market_cap = market_data.get("market_cap", 0)
        current_price = market_data.get("price", 0)
        
        # Estimate shares outstanding from market cap / price
        if current_price and market_cap:
            shares = market_cap / current_price
        else:
            # Use reported shares if available
            shares = 15e9  # Default estimate
            _, reported_shares = self._latest_value_asof(ticker, "SharesOutstanding")
            if reported_shares:
                shares = reported_shares
        
        # DCF assumptions - Dynamic Estimation
        # 1. Estimate Growth Rate from historical FCF trend
        historical_fcf = []
        capex_dict = dict(capex_history)
        for year, ocf in ocf_history:
            if year in capex_dict:
                historical_fcf.append((int(year), ocf - abs(capex_dict[year])))
        
        if len(historical_fcf) >= 3:
            trend = self.historical_analyzer.analyze_trend(historical_fcf, "Free Cash Flow", ticker)
            # Cap growth between 0% and 15% for conservative DCF
            growth_rate = max(0.02, min(0.15, trend.context.get("cagr", 8.0) / 100))
            growth_method = f"Estimated from {len(historical_fcf)}-year CAGR"
        else:
            growth_rate = 0.07  # Conservative default
            growth_method = "Default (insufficient history)"

        # 2. Terminal Growth (typically 2-3% reflecting long-term GDP growth)
        terminal_growth = 0.025

        # 3. Discount Rate (WACC) - Defaulting to 9% for tech, 8% otherwise
        # Future: Estimate from Beta and Risk-Free Rate
        is_tech = any(t in ticker for t in ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'])
        discount_rate = 0.09 if is_tech else 0.08
        rate_method = "Sector-adjusted average"
        
        # Calculate DCF
        dcf_result = self.projection_engine.dcf_valuation(
            current_fcf=current_fcf,
            growth_rate_5yr=growth_rate,
            terminal_growth=terminal_growth,
            discount_rate=discount_rate,
            shares_outstanding=shares
        )
        
        lines = [
            f"📊 DCF Valuation for {ticker}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "📈 Inputs:",
            f"  Current FCF: ${current_fcf/1e9:.2f}B",
            f"  Growth Rate (5yr): {growth_rate*100:.1f}% ({growth_method})",
            f"  Terminal Growth: {terminal_growth*100:.1f}%",
            f"  Discount Rate (WACC): {discount_rate*100:.1f}% ({rate_method})",
            "",
            "💰 Valuation:",
            f"  PV of FCFs: ${dcf_result['pv_of_fcfs']/1e9:.2f}B",
            f"  Terminal Value: ${dcf_result['terminal_value']/1e9:.2f}B",
            f"  PV of Terminal: ${dcf_result['pv_terminal_value']/1e9:.2f}B",
            f"  Enterprise Value: ${dcf_result['enterprise_value']/1e9:.2f}B",
            "",
            f"🎯 Fair Value per Share: ${dcf_result['fair_value_per_share']:.2f}",
            f"   Current Price: ${current_price:.2f}" if current_price else "",
        ]
        
        if current_price:
            upside = (dcf_result['fair_value_per_share'] / current_price - 1) * 100
            lines.append(f"   Implied Upside: {upside:+.1f}%")
        
        lines.extend([
            "",
            "⚠️ Disclaimer:",
            "  This is a simplified DCF model for illustrative purposes.",
            "  Actual valuation requires detailed analysis of WACC,",
            "  growth assumptions, and company-specific factors."
        ])
        
        return "\n".join([l for l in lines if l])
    
    def _get_market_data(self, ticker: str) -> str:
        """Get current market data."""
        ticker = ticker.strip().upper()

        if self._analysis_date and self._analysis_date != datetime.now().strftime("%Y%m%d"):
            return f"Current market data is unavailable for historical as-of date {self._analysis_date}."
        
        data = self.stock_loader.get_current_price(ticker)
        estimates = self.stock_loader.get_analyst_estimates(ticker)
        
        if "error" in data:
            return f"Error getting market data: {data['error']}"
        
        lines = [
            f"📈 Market Data for {ticker}",
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "💵 Price & Valuation:",
            f"  Current Price: ${data.get('price', 0):.2f}",
            f"  Market Cap: ${data.get('market_cap', 0)/1e9:.2f}B" if data.get('market_cap') else "",
            f"  P/E (Trailing): {data.get('pe_ratio', 'N/A')}",
            f"  P/E (Forward): {data.get('forward_pe', 'N/A')}",
            f"  EPS: ${data.get('eps', 0):.2f}" if data.get('eps') else "",
            "",
            "📊 52-Week Range:",
            f"  High: ${data.get('52_week_high', 0):.2f}",
            f"  Low: ${data.get('52_week_low', 0):.2f}",
        ]
        
        if estimates and "error" not in estimates:
            lines.extend([
                "",
                "🎯 Analyst Estimates:",
                f"  Target Price (Mean): ${estimates.get('target_mean', 0):.2f}",
                f"  Target Range: ${estimates.get('target_low', 0):.2f} - ${estimates.get('target_high', 0):.2f}",
                f"  Recommendation: {estimates.get('recommendation', 'N/A').upper()}",
                f"  Number of Analysts: {estimates.get('num_analysts', 'N/A')}",
                f"  Forward EPS: ${estimates.get('forward_eps', 0):.2f}" if estimates.get('forward_eps') else "",
            ])
        
        return "\n".join([l for l in lines if l])
    
    def _list_metrics(self, ticker: str) -> str:
        """List available metrics."""
        ticker = ticker.strip().upper()
        
        metrics = self.data_loader.get_all_metrics(ticker)
        
        if not metrics:
            return f"No data found for {ticker}"
        
        lines = [
            f"📋 Available Metrics for {ticker}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Total: {len(metrics)} metrics",
            "",
        ]
        
        # Group by category
        categories = {
            "Income Statement": ["Revenue", "GrossProfit", "OperatingIncome", "NetIncome", "EPS"],
            "Balance Sheet": ["Assets", "Liabilities", "StockholdersEquity", "Cash", "Debt"],
            "Cash Flow": ["OperatingCashFlow", "InvestingCashFlow", "FinancingCashFlow", "CapitalExpenditures"],
        }
        
        for category, key_metrics in categories.items():
            found = [m for m in metrics if any(k in m for k in key_metrics)]
            if found:
                lines.append(f"📊 {category}:")
                for m in found[:8]:
                    lines.append(f"  • {m}")
                if len(found) > 8:
                    lines.append(f"  ... and {len(found)-8} more")
                lines.append("")
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════════
    # MAIN ANALYSIS METHOD
    # ═══════════════════════════════════════════════════════════════════
    
    def analyze(self, query: str, ticker: str = None, analysis_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a financial analysis query.
        
        Args:
            query: Natural language query
            ticker: Optional ticker (extracted from query if not provided)
            
        Returns:
            Analysis results
        """
        start_time = datetime.now()
        self._analysis_date = analysis_date or datetime.now().strftime("%Y%m%d")
        
        # Try to extract ticker from query if not provided
        if not ticker:
            ticker_match = re.search(r'\b(AAPL|MSFT|GOOGL|AMZN|META|JPM|GS|XOM|CVX)\b', query.upper())
            if ticker_match:
                ticker = ticker_match.group(1)
        
        if self.agent:
            try:
                input_message = f"""Analyze this financial query:
Query: {query}
Ticker: {ticker or 'Please identify from query'}
Analysis Date (as-of): {analysis_date or 'N/A'}

Use the available tools to gather data and provide a comprehensive analysis.
For projections, use multiple methods and provide ranges.
For historical questions, cite specific dates and values.
Always note assumptions and confidence levels."""

                result = self.agent.invoke({
                    "messages": [HumanMessage(content=input_message)]
                })
                
                output = result.get('messages', [])[-1].content if result.get('messages') else ''
                
                return {
                    'output': output,
                    'query': query,
                    'ticker': ticker,
                    'agent': 'FinancialAnalysisAgent',
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Agent error: {e}")
        
        # Fallback: direct tool usage based on query analysis
        return self._fallback_analysis(query, ticker)
    
    def _fallback_analysis(self, query: str, ticker: str = None) -> Dict[str, Any]:
        """Fallback analysis without LLM agent."""
        query_lower = query.lower()
        
        # Detect query type
        if any(word in query_lower for word in ['project', 'forecast', 'predict', '2026', '2027', '2028', 'future']):
            # Projection query
            year_match = re.search(r'20\d{2}', query)
            target_year = int(year_match.group()) if year_match else 2027
            
            metric = "Revenue"  # Default
            if "cash flow" in query_lower or "cashflow" in query_lower or "fcf" in query_lower:
                metric = "OperatingCashFlow"
            elif "profit" in query_lower or "income" in query_lower or "earnings" in query_lower:
                metric = "NetIncome"
            elif "eps" in query_lower:
                metric = "EPS_Diluted"
            
            if ticker:
                output = self._project_metric(f"{ticker}|{metric}|{target_year}")
            else:
                output = "Please specify a ticker symbol (e.g., AAPL, MSFT)"
                
        elif any(word in query_lower for word in ['highest', 'maximum', 'max', 'peak', 'record']):
            # Historical maximum query
            metric = "EPS_Diluted" if "eps" in query_lower else "Revenue" if "revenue" in query_lower else "NetIncome"
            if ticker:
                output = self._find_extreme(f"{ticker}|{metric}|maximum")
            else:
                output = "Please specify a ticker symbol"
                
        elif any(word in query_lower for word in ['lowest', 'minimum', 'min']):
            # Historical minimum query
            metric = "EPS_Diluted" if "eps" in query_lower else "Revenue"
            if ticker:
                output = self._find_extreme(f"{ticker}|{metric}|minimum")
            else:
                output = "Please specify a ticker symbol"
                
        elif any(word in query_lower for word in ['ratio', 'roe', 'roa', 'margin', 'debt']):
            # Ratio calculation
            ratio = "ROE"
            if "roa" in query_lower:
                ratio = "ROA"
            elif "margin" in query_lower:
                ratio = "NetMargin"
            elif "debt" in query_lower:
                ratio = "DebtToEquity"
                
            if ticker:
                output = self._calculate_ratio(f"{ticker}|{ratio}")
            else:
                output = "Please specify a ticker symbol"
        
        else:
            # Default: load data and provide summary
            if ticker:
                output = self._load_financial_data(ticker)
            else:
                output = "Please provide more details. I can help with:\n" \
                        "• Projections (e.g., 'projected cash flow for AAPL in 2027')\n" \
                        "• Historical analysis (e.g., 'when was MSFT highest EPS?')\n" \
                        "• Financial ratios (e.g., 'what is Apple's ROE?')\n" \
                        "• Market data (e.g., 'current price and P/E for JPM')"
        
        return {
            'output': output,
            'query': query,
            'ticker': ticker,
            'agent': 'FinancialAnalysisAgent',
            'mode': 'fallback',
            'timestamp': datetime.now().isoformat()
        }
