"""
Market Data Agent for TemporalGuard-RAG

Real-time stock market data integration using yfinance.
Provides live prices, historical data, and company fundamentals.
"""

import yfinance as yf
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataAgent:
    """
    Real-time market data agent using Yahoo Finance.
    
    Capabilities:
    - Live stock prices and changes
    - Historical price data
    - Company fundamentals
    - Financial statements
    - Key statistics
    """
    
    def __init__(self):
        """Initialize Market Data Agent."""
        self.cache = {}  # Simple cache for repeated lookups
        self.cache_ttl = 60  # Cache TTL in seconds
        logger.info("Initialized Market Data Agent")
    
    def _cache_get(self, key: str):
        """Return cached value if it is still fresh."""
        cached = self.cache.get(key)
        if cached is None:
            return None

        value, expires_at = cached
        if time.monotonic() >= expires_at:
            self.cache.pop(key, None)
            return None

        return value
    
    def _cache_set(self, key: str, value: Any):
        """Store fetched payload with a TTL."""
        self.cache[key] = (value, time.monotonic() + self.cache_ttl)
    
    def get_live_price(self, ticker: str) -> Dict[str, Any]:
        """
        Get real-time stock price and daily change.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary with price data
        """
        cache_key = f"live_price:{ticker.upper()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            
            result = {
                'ticker': ticker,
                'price': round(info.last_price, 2) if info.last_price else None,
                'previous_close': round(info.previous_close, 2) if info.previous_close else None,
                'change': round(info.last_price - info.previous_close, 2) if info.last_price and info.previous_close else None,
                'change_percent': round(((info.last_price - info.previous_close) / info.previous_close) * 100, 2) if info.last_price and info.previous_close else None,
                'market_cap': info.market_cap,
                'volume': info.last_volume,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error fetching live price for {ticker}: {e}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            }
    
    def get_historical_prices(self, ticker: str, period: str = "1mo", 
                             interval: str = "1d") -> Dict[str, Any]:
        """
        Get historical price data.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary with historical data
        """
        cache_key = f"historical:{ticker.upper()}:{period}:{interval}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                result = {
                    'ticker': ticker,
                    'status': 'error',
                    'error': 'No data available'
                }
                self._cache_set(cache_key, result)
                return result
            
            # Convert to records
            records = []
            for date, row in hist.iterrows():
                records.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(row['Open'], 2),
                    'high': round(row['High'], 2),
                    'low': round(row['Low'], 2),
                    'close': round(row['Close'], 2),
                    'volume': int(row['Volume'])
                })
            
            # Calculate summary stats
            result = {
                'ticker': ticker,
                'period': period,
                'interval': interval,
                'data_points': len(records),
                'start_date': records[0]['date'] if records else None,
                'end_date': records[-1]['date'] if records else None,
                'start_price': records[0]['close'] if records else None,
                'end_price': records[-1]['close'] if records else None,
                'period_change': round(records[-1]['close'] - records[0]['close'], 2) if records else None,
                'period_change_percent': round(((records[-1]['close'] - records[0]['close']) / records[0]['close']) * 100, 2) if records else None,
                'high': max(r['high'] for r in records) if records else None,
                'low': min(r['low'] for r in records) if records else None,
                'avg_volume': int(sum(r['volume'] for r in records) / len(records)) if records else None,
                'prices': records[-30:],  # Last 30 data points
                'status': 'success'
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            }
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get company information and fundamentals.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company info
        """
        cache_key = f"company_info:{ticker.upper()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            result = {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary', '')[:500] + '...' if info.get('longBusinessSummary') else None,
                'employees': info.get('fullTimeEmployees'),
                'country': info.get('country'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'revenue': info.get('totalRevenue'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {e}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            }
    
    def get_financials(self, ticker: str) -> Dict[str, Any]:
        """
        Get financial statements (annual).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with financial data
        """
        cache_key = f"financials:{ticker.upper()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get income statement
            income = stock.income_stmt
            balance = stock.balance_sheet
            cashflow = stock.cashflow
            
            def df_to_dict(df, name):
                if df is None or df.empty:
                    return {}
                result = {}
                for col in df.columns[:4]:  # Last 4 periods
                    period = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)
                    result[period] = {}
                    for idx in df.index:
                        val = df.loc[idx, col]
                        if pd.notna(val):
                            result[period][str(idx)] = float(val)
                return result
            
            result = {
                'ticker': ticker,
                'income_statement': df_to_dict(income, 'income'),
                'balance_sheet': df_to_dict(balance, 'balance'),
                'cash_flow': df_to_dict(cashflow, 'cashflow'),
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error fetching financials for {ticker}: {e}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            }
    
    def get_analyst_recommendations(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst recommendations and price targets.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with analyst data
        """
        cache_key = f"analyst:{ticker.upper()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get recommendations
            recs = stock.recommendations
            rec_summary = None
            if recs is not None and not recs.empty:
                recent = recs.tail(10).to_dict('records')
                rec_summary = recent
            
            # Get info for targets
            info = stock.info
            
            result = {
                'ticker': ticker,
                'target_high': info.get('targetHighPrice'),
                'target_low': info.get('targetLowPrice'),
                'target_mean': info.get('targetMeanPrice'),
                'target_median': info.get('targetMedianPrice'),
                'recommendation': info.get('recommendationKey'),
                'recommendation_mean': info.get('recommendationMean'),
                'num_analysts': info.get('numberOfAnalystOpinions'),
                'recent_recommendations': rec_summary,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error fetching analyst data for {ticker}: {e}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            }
    
    def get_earnings_estimates(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings estimates and forward projections.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with earnings estimates
        """
        cache_key = f"earnings_estimates:{ticker.upper()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get earnings estimates
            earnings_estimate = None
            revenue_estimate = None
            
            try:
                # Earnings calendar
                calendar = stock.calendar
                if calendar is not None and not calendar.empty if hasattr(calendar, 'empty') else calendar:
                    earnings_estimate = calendar
            except:
                pass
            
            # Get analyst earnings estimates
            try:
                earnings_trend = stock.earnings_estimate
                if earnings_trend is not None and not earnings_trend.empty:
                    earnings_estimate = earnings_trend.to_dict()
            except:
                pass
            
            # Get revenue estimates
            try:
                rev_est = stock.revenue_estimate
                if rev_est is not None and not rev_est.empty:
                    revenue_estimate = rev_est.to_dict()
            except:
                pass
            
            result = {
                'ticker': ticker,
                'current_eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'current_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                'revenue_growth': info.get('revenueGrowth'),
                'next_earnings_date': str(info.get('earningsDate', [None])[0]) if info.get('earningsDate') else None,
                'earnings_estimate': earnings_estimate,
                'revenue_estimate': revenue_estimate,
                'analyst_growth_expectations': {
                    'earnings_growth_5y': info.get('earningsGrowth'),
                    'revenue_growth_estimate': info.get('revenueGrowth'),
                },
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error fetching earnings estimates for {ticker}: {e}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            }
    
    def get_forward_guidance(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive forward-looking data for projections.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with forward guidance and projections
        """
        cache_key = f"forward_guidance:{ticker.upper()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Current price for calculating upside/downside
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            target_mean = info.get('targetMeanPrice')
            
            upside_potential = None
            if current_price and target_mean:
                upside_potential = round(((target_mean - current_price) / current_price) * 100, 2)
            
            result = {
                'ticker': ticker,
                'company_name': info.get('longName'),
                
                # Current metrics
                'current_price': current_price,
                'market_cap': info.get('marketCap'),
                
                # Valuation
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'price_to_book': info.get('priceToBook'),
                'enterprise_to_ebitda': info.get('enterpriseToEbitda'),
                
                # Growth metrics
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                
                # Analyst targets
                'target_low': info.get('targetLowPrice'),
                'target_mean': target_mean,
                'target_high': info.get('targetHighPrice'),
                'upside_potential_percent': upside_potential,
                
                # Recommendation
                'recommendation': info.get('recommendationKey'),
                'num_analysts': info.get('numberOfAnalystOpinions'),
                
                # EPS
                'trailing_eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'eps_growth_expected': round(((info.get('forwardEps', 0) - info.get('trailingEps', 0)) / info.get('trailingEps', 1)) * 100, 2) if info.get('trailingEps') and info.get('forwardEps') else None,
                
                # Dividends
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                
                # Risk
                'beta': info.get('beta'),
                
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                
                # Disclaimer
                'disclaimer': 'Forward-looking data is based on analyst estimates and may not reflect actual future performance.'
            }
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error fetching forward guidance for {ticker}: {e}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            }
    
    def compare_stocks(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Compare multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary with comparison data
        """
        try:
            results = []
            for ticker in tickers[:10]:  # Limit to 10 stocks
                info = self.get_company_info(ticker)
                price = self.get_live_price(ticker)
                
                results.append({
                    'ticker': ticker,
                    'name': info.get('name'),
                    'price': price.get('price'),
                    'change_percent': price.get('change_percent'),
                    'market_cap': info.get('market_cap'),
                    'pe_ratio': info.get('trailing_pe'),
                    'profit_margin': info.get('profit_margin'),
                    'revenue_growth': info.get('revenue_growth'),
                    'dividend_yield': info.get('dividend_yield')
                })
            
            return {
                'tickers': tickers,
                'comparison': results,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error comparing stocks: {e}")
            return {
                'tickers': tickers,
                'status': 'error',
                'error': str(e)
            }
    
    def answer_market_question(self, question: str, ticker: str = None) -> Dict[str, Any]:
        """
        Answer a market-related question using real-time data.
        
        Args:
            question: Natural language question
            ticker: Optional specific ticker
            
        Returns:
            Dictionary with answer and supporting data
        """
        question_lower = question.lower()
        
        # Determine what data is needed
        needs_price = any(w in question_lower for w in ['price', 'trading', 'worth', 'cost', 'today', 'now'])
        needs_history = any(w in question_lower for w in ['history', 'trend', 'week', 'month', 'year', 'performance', 'return'])
        needs_fundamentals = any(w in question_lower for w in ['pe', 'ratio', 'margin', 'revenue', 'profit', 'growth', 'fundamental'])
        needs_analyst = any(w in question_lower for w in ['analyst', 'target', 'recommend', 'buy', 'sell', 'hold'])
        
        result = {
            'question': question,
            'ticker': ticker,
            'data': {},
            'answer': None,
            'timestamp': datetime.now().isoformat()
        }
        
        if ticker:
            if needs_price or not (needs_history or needs_fundamentals or needs_analyst):
                result['data']['live_price'] = self.get_live_price(ticker)
            
            if needs_history:
                result['data']['historical'] = self.get_historical_prices(ticker, period='3mo')
            
            if needs_fundamentals:
                result['data']['fundamentals'] = self.get_company_info(ticker)
            
            if needs_analyst:
                result['data']['analyst'] = self.get_analyst_recommendations(ticker)
        
        # Generate simple answer
        if ticker and 'live_price' in result['data']:
            price_data = result['data']['live_price']
            if price_data.get('status') == 'success':
                change_str = f"{'+' if price_data['change'] >= 0 else ''}{price_data['change']} ({price_data['change_percent']}%)"
                result['answer'] = f"{ticker} is currently trading at ${price_data['price']}, {change_str} from previous close."
        
        result['status'] = 'success'
        return result


# Convenience functions
def get_stock_price(ticker: str) -> Dict:
    """Get current stock price."""
    agent = MarketDataAgent()
    return agent.get_live_price(ticker)

def get_stock_history(ticker: str, period: str = "1mo") -> Dict:
    """Get historical stock data."""
    agent = MarketDataAgent()
    return agent.get_historical_prices(ticker, period)

def get_stock_info(ticker: str) -> Dict:
    """Get company information."""
    agent = MarketDataAgent()
    return agent.get_company_info(ticker)


if __name__ == "__main__":
    # Test the agent
    agent = MarketDataAgent()
    
    print("=== Live Price ===")
    print(agent.get_live_price("AAPL"))
    
    print("\n=== Company Info ===")
    info = agent.get_company_info("AAPL")
    print(f"Name: {info['name']}")
    print(f"Market Cap: ${info['market_cap']:,.0f}" if info['market_cap'] else "N/A")
    print(f"P/E Ratio: {info['trailing_pe']}")
    print(f"Profit Margin: {info['profit_margin']}")
