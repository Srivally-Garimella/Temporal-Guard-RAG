"""
Comprehensive Financial Data Loader for TemporalGuard-RAG

Extracts and organizes:
- Balance Sheet items
- Income Statement items
- Cash Flow Statement items
- Per-share metrics
- Financial ratios
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """
    Loads and organizes financial data from XBRL files.
    
    Provides structured access to:
    - Balance Sheet (Assets, Liabilities, Equity)
    - Income Statement (Revenue, Expenses, Profit)
    - Cash Flow Statement (Operating, Investing, Financing)
    - Key metrics and ratios
    """
    
    # XBRL concept mappings for standardization
    BALANCE_SHEET_CONCEPTS = {
        # Assets
        'Assets': ['Assets'],
        'CurrentAssets': ['AssetsCurrent'],
        'Cash': ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsAndShortTermInvestments'],
        'AccountsReceivable': ['AccountsReceivableNetCurrent', 'ReceivablesNetCurrent'],
        'Inventory': ['InventoryNet'],
        'PrepaidExpenses': ['PrepaidExpenseAndOtherAssetsCurrent'],
        'NonCurrentAssets': ['AssetsNoncurrent'],
        'PropertyPlantEquipment': ['PropertyPlantAndEquipmentNet'],
        'Goodwill': ['Goodwill'],
        'IntangibleAssets': ['IntangibleAssetsNetExcludingGoodwill'],
        'Investments': ['Investments', 'LongTermInvestments'],
        
        # Liabilities
        'Liabilities': ['Liabilities'],
        'CurrentLiabilities': ['LiabilitiesCurrent'],
        'AccountsPayable': ['AccountsPayableCurrent'],
        'ShortTermDebt': ['ShortTermBorrowings', 'DebtCurrent'],
        'DeferredRevenue': ['DeferredRevenueCurrent', 'ContractWithCustomerLiabilityCurrent'],
        'AccruedLiabilities': ['AccruedLiabilitiesCurrent'],
        'NonCurrentLiabilities': ['LiabilitiesNoncurrent'],
        'LongTermDebt': ['LongTermDebtNoncurrent', 'LongTermDebt'],
        'DeferredTaxLiabilities': ['DeferredIncomeTaxLiabilitiesNet'],
        
        # Equity
        'StockholdersEquity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
        'CommonStock': ['CommonStockValue'],
        'RetainedEarnings': ['RetainedEarningsAccumulatedDeficit'],
        'TreasuryStock': ['TreasuryStockValue'],
        'AccumulatedOCI': ['AccumulatedOtherComprehensiveIncomeLossNetOfTax'],
    }
    
    INCOME_STATEMENT_CONCEPTS = {
        # Revenue
        'Revenue': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 
                   'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax'],
        'CostOfRevenue': ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfGoodsSold'],
        'GrossProfit': ['GrossProfit'],
        
        # Operating Expenses
        'OperatingExpenses': ['OperatingExpenses'],
        'ResearchAndDevelopment': ['ResearchAndDevelopmentExpense'],
        'SellingGeneralAdmin': ['SellingGeneralAndAdministrativeExpense'],
        'Depreciation': ['DepreciationAndAmortization', 'Depreciation'],
        'OperatingIncome': ['OperatingIncomeLoss'],
        
        # Other Income/Expense
        'InterestExpense': ['InterestExpense', 'InterestAndDebtExpense'],
        'InterestIncome': ['InterestIncome', 'InvestmentIncomeInterest'],
        'OtherIncome': ['OtherNonoperatingIncomeExpense', 'NonoperatingIncomeExpense'],
        
        # Taxes and Net Income
        'IncomeBeforeTax': ['IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'],
        'IncomeTaxExpense': ['IncomeTaxExpenseBenefit'],
        'NetIncome': ['NetIncomeLoss', 'ProfitLoss'],
        
        # EPS
        'EPS_Basic': ['EarningsPerShareBasic'],
        'EPS_Diluted': ['EarningsPerShareDiluted'],
        'SharesOutstanding': ['CommonStockSharesOutstanding', 'WeightedAverageNumberOfSharesOutstandingBasic'],
    }
    
    CASH_FLOW_CONCEPTS = {
        # Operating Activities
        'OperatingCashFlow': ['NetCashProvidedByUsedInOperatingActivities'],
        'DepreciationAmortization': ['DepreciationDepletionAndAmortization'],
        'StockBasedCompensation': ['ShareBasedCompensation'],
        'DeferredTaxes': ['DeferredIncomeTaxExpenseBenefit'],
        'WorkingCapitalChanges': ['IncreaseDecreaseInOperatingCapital'],
        
        # Investing Activities
        'InvestingCashFlow': ['NetCashProvidedByUsedInInvestingActivities'],
        'CapitalExpenditures': ['PaymentsToAcquirePropertyPlantAndEquipment'],
        'Acquisitions': ['PaymentsToAcquireBusinessesNetOfCashAcquired'],
        'InvestmentPurchases': ['PaymentsToAcquireInvestments'],
        'InvestmentSales': ['ProceedsFromSaleOfInvestments'],
        
        # Financing Activities
        'FinancingCashFlow': ['NetCashProvidedByUsedInFinancingActivities'],
        'DebtRepayment': ['RepaymentsOfDebt', 'RepaymentsOfLongTermDebt'],
        'DebtIssuance': ['ProceedsFromIssuanceOfDebt', 'ProceedsFromIssuanceOfLongTermDebt'],
        'ShareRepurchase': ['PaymentsForRepurchaseOfCommonStock'],
        'DividendsPaid': ['PaymentsOfDividends', 'PaymentsOfDividendsCommonStock'],
        
        # Net Change
        'NetCashChange': ['CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect'],
    }
    
    def __init__(self, xbrl_dir: str = "data/raw/xbrl_structured"):
        """Initialize with path to XBRL data."""
        self.xbrl_dir = Path(xbrl_dir)
        self.cache = {}  # Cache loaded data
        logger.info(f"Initialized Financial Data Loader with dir: {xbrl_dir}")
    
    def load_company_data(self, ticker: str) -> Dict:
        """
        Load all financial data for a company.
        
        Returns dict with:
        - balance_sheet: Dict of metrics over time
        - income_statement: Dict of metrics over time
        - cash_flow: Dict of metrics over time
        - ratios: Calculated financial ratios
        - metadata: Company info
        """
        if ticker in self.cache:
            return self.cache[ticker]
        
        facts_path = self.xbrl_dir / f"{ticker}_facts.json"
        metrics_path = self.xbrl_dir / f"{ticker}_metrics.csv"
        
        data = {
            "ticker": ticker,
            "balance_sheet": {},
            "income_statement": {},
            "cash_flow": {},
            "ratios": {},
            "metadata": {},
            "time_series": {}
        }
        
        if facts_path.exists():
            data = self._load_from_facts(ticker, facts_path, data)
        elif metrics_path.exists():
            data = self._load_from_metrics(ticker, metrics_path, data)
        else:
            logger.warning(f"No XBRL data found for {ticker}")
        
        self.cache[ticker] = data
        return data
    
    def _load_from_facts(self, ticker: str, facts_path: Path, data: Dict) -> Dict:
        """Load comprehensive data from XBRL facts JSON."""
        try:
            with open(facts_path, 'r') as f:
                facts = json.load(f)
            
            # Get company metadata
            data["metadata"]["cik"] = facts.get("cik")
            data["metadata"]["entity_name"] = facts.get("entityName")
            
            # Process us-gaap facts
            us_gaap = facts.get("facts", {}).get("us-gaap", {})
            
            # Extract Balance Sheet items
            for std_name, xbrl_names in self.BALANCE_SHEET_CONCEPTS.items():
                for xbrl_name in xbrl_names:
                    if xbrl_name in us_gaap:
                        values = self._extract_values(us_gaap[xbrl_name])
                        if values:
                            data["balance_sheet"][std_name] = values
                            data["time_series"][std_name] = values
                            break
            
            # Extract Income Statement items
            for std_name, xbrl_names in self.INCOME_STATEMENT_CONCEPTS.items():
                for xbrl_name in xbrl_names:
                    if xbrl_name in us_gaap:
                        values = self._extract_values(us_gaap[xbrl_name])
                        if values:
                            data["income_statement"][std_name] = values
                            data["time_series"][std_name] = values
                            break
            
            # Extract Cash Flow items
            for std_name, xbrl_names in self.CASH_FLOW_CONCEPTS.items():
                for xbrl_name in xbrl_names:
                    if xbrl_name in us_gaap:
                        values = self._extract_values(us_gaap[xbrl_name])
                        if values:
                            data["cash_flow"][std_name] = values
                            data["time_series"][std_name] = values
                            break
            
            # Calculate key ratios
            data["ratios"] = self._calculate_ratios(data)
            
            logger.info(f"Loaded {len(data['time_series'])} metrics for {ticker}")
            
        except Exception as e:
            logger.error(f"Error loading facts for {ticker}: {e}")
        
        return data
    
    def _load_from_metrics(self, ticker: str, metrics_path: Path, data: Dict) -> Dict:
        """Load data from pre-extracted metrics CSV."""
        try:
            df = pd.read_csv(metrics_path)

            for metric, metric_df in df.groupby('metric', sort=False):
                metric_df = metric_df.sort_values('end_date')
                values = {}

                for row in metric_df.itertuples(index=False):
                    date_key = str(getattr(row, 'end_date', ''))[:10]
                    values[date_key] = {
                        "value": getattr(row, 'value'),
                        "filed": str(getattr(row, 'filed_date', ''))[:10] if pd.notna(getattr(row, 'filed_date', None)) else None,
                        "form": getattr(row, 'form', '10-K')
                    }

                data["time_series"][metric] = values
            
        except Exception as e:
            logger.error(f"Error loading metrics for {ticker}: {e}")
        
        return data
    
    def _extract_values(self, concept_data: Dict) -> Dict[str, Dict]:
        """Extract values from XBRL concept data."""
        values = {}
        
        # Get USD values for annual periods
        units = concept_data.get("units", {})
        
        # Try USD first, then shares
        usd_values = units.get("USD", units.get("USD/shares", units.get("shares", [])))
        
        for entry in usd_values:
            # Only get annual data (10-K) or instant values
            form = entry.get("form", "")
            if form not in ["10-K", "10-K/A"]:
                continue
            
            end_date = entry.get("end")
            if not end_date:
                continue
            
            # Use fiscal year end as key
            year = end_date[:4]
            
            if year not in values or entry.get("filed", "") > values[year].get("filed", ""):
                values[year] = {
                    "value": entry.get("val"),
                    "filed": entry.get("filed"),
                    "end_date": end_date,
                    "form": form
                }
        
        return values
    
    def _calculate_ratios(self, data: Dict) -> Dict:
        """Calculate financial ratios from loaded data."""
        ratios = {}
        
        bs = data.get("balance_sheet", {})
        inc = data.get("income_statement", {})
        cf = data.get("cash_flow", {})
        
        # Get latest year data
        def get_latest(metric_dict: Dict) -> Tuple[Optional[str], Optional[float]]:
            if not metric_dict:
                return None, None
            latest_year = max(metric_dict.keys())
            return latest_year, metric_dict[latest_year].get("value")
        
        # Profitability Ratios
        year, net_income = get_latest(inc.get("NetIncome", {}))
        _, revenue = get_latest(inc.get("Revenue", {}))
        _, equity = get_latest(bs.get("StockholdersEquity", {}))
        _, assets = get_latest(bs.get("Assets", {}))
        
        if net_income and revenue:
            ratios["NetProfitMargin"] = {"value": net_income / revenue * 100, "year": year}
        if net_income and equity:
            ratios["ROE"] = {"value": net_income / equity * 100, "year": year}
        if net_income and assets:
            ratios["ROA"] = {"value": net_income / assets * 100, "year": year}
        
        # Liquidity Ratios
        _, current_assets = get_latest(bs.get("CurrentAssets", {}))
        _, current_liab = get_latest(bs.get("CurrentLiabilities", {}))
        _, inventory = get_latest(bs.get("Inventory", {}))
        _, cash = get_latest(bs.get("Cash", {}))
        
        if current_assets and current_liab:
            ratios["CurrentRatio"] = {"value": current_assets / current_liab, "year": year}
        if current_assets and current_liab and inventory:
            ratios["QuickRatio"] = {"value": (current_assets - inventory) / current_liab, "year": year}
        
        # Leverage Ratios
        _, liabilities = get_latest(bs.get("Liabilities", {}))
        _, long_term_debt = get_latest(bs.get("LongTermDebt", {}))
        
        if liabilities and assets:
            ratios["DebtRatio"] = {"value": liabilities / assets, "year": year}
        if long_term_debt and equity:
            ratios["DebtToEquity"] = {"value": long_term_debt / equity, "year": year}
        
        # Cash Flow Ratios
        _, ocf = get_latest(cf.get("OperatingCashFlow", {}))
        _, capex = get_latest(cf.get("CapitalExpenditures", {}))
        
        if ocf and capex:
            ratios["FreeCashFlow"] = {"value": ocf - abs(capex), "year": year}
            if revenue:
                ratios["FCFMargin"] = {"value": (ocf - abs(capex)) / revenue * 100, "year": year}
        
        return ratios
    
    # ═══════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_metric_history(self, ticker: str, metric: str) -> List[Tuple[str, float]]:
        """Get time series for a specific metric."""
        data = self.load_company_data(ticker)
        
        metric_data = data.get("time_series", {}).get(metric, {})
        
        if not metric_data:
            # Try alternate names
            for category in [data.get("balance_sheet"), data.get("income_statement"), data.get("cash_flow")]:
                if category and metric in category:
                    metric_data = category[metric]
                    break
        
        result = []
        for year, info in sorted(metric_data.items()):
            if isinstance(info, dict):
                result.append((year, info.get("value", 0)))
            else:
                result.append((year, info))
        
        return result
    
    def get_latest_value(self, ticker: str, metric: str) -> Tuple[Optional[str], Optional[float]]:
        """Get the most recent value for a metric."""
        history = self.get_metric_history(ticker, metric)
        if history:
            return history[-1]
        return None, None
    
    def get_all_metrics(self, ticker: str) -> List[str]:
        """Get list of all available metrics for a company."""
        data = self.load_company_data(ticker)
        return list(data.get("time_series", {}).keys())
    
    def get_financial_summary(self, ticker: str) -> Dict:
        """Get a summary of key financial metrics."""
        data = self.load_company_data(ticker)
        
        summary = {
            "ticker": ticker,
            "company": data.get("metadata", {}).get("entity_name", ticker),
            "latest_data": {},
            "ratios": data.get("ratios", {}),
            "available_metrics": len(data.get("time_series", {})),
        }
        
        # Get latest values for key metrics
        key_metrics = ["Revenue", "NetIncome", "Assets", "StockholdersEquity", 
                      "OperatingCashFlow", "EPS_Diluted"]
        
        for metric in key_metrics:
            year, value = self.get_latest_value(ticker, metric)
            if value is not None:
                summary["latest_data"][metric] = {"year": year, "value": value}
        
        return summary
    
    def export_to_dataframe(self, ticker: str) -> pd.DataFrame:
        """Export all time series data as a DataFrame."""
        data = self.load_company_data(ticker)
        
        records = []
        for metric, values in data.get("time_series", {}).items():
            for year, info in values.items():
                records.append({
                    "ticker": ticker,
                    "metric": metric,
                    "year": year,
                    "value": info.get("value") if isinstance(info, dict) else info,
                    "filed": info.get("filed") if isinstance(info, dict) else None,
                })
        
        return pd.DataFrame(records)


# Stock data integration using yfinance (if available)
class StockDataLoader:
    """Load market data for stocks."""
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except ImportError:
            logger.warning("yfinance not installed - stock data unavailable")
            self.available = False
    
    def get_current_price(self, ticker: str) -> Dict:
        """Get current stock price and basic info."""
        if not self.available:
            return {"error": "yfinance not installed"}
        
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info
            
            return {
                "ticker": ticker,
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "analyst_target": info.get("targetMeanPrice"),
                "recommendation": info.get("recommendationKey"),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_price_history(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """Get historical price data."""
        if not self.available:
            return pd.DataFrame()
        
        try:
            stock = self.yf.Ticker(ticker)
            return stock.history(period=period)
        except Exception as e:
            logger.error(f"Error getting price history: {e}")
            return pd.DataFrame()
    
    def get_analyst_estimates(self, ticker: str) -> Dict:
        """Get analyst estimates and forecasts."""
        if not self.available:
            return {"error": "yfinance not installed"}
        
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info
            
            return {
                "ticker": ticker,
                "forward_eps": info.get("forwardEps"),
                "target_low": info.get("targetLowPrice"),
                "target_mean": info.get("targetMeanPrice"),
                "target_high": info.get("targetHighPrice"),
                "recommendation": info.get("recommendationKey"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
                "earnings_growth": info.get("earningsGrowth"),
                "revenue_growth": info.get("revenueGrowth"),
            }
        except Exception as e:
            return {"error": str(e)}
