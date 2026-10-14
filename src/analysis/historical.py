"""
Historical Financial Analysis for TemporalGuard-RAG

Provides historical analysis capabilities including:
- Finding maxima/minima (highest EPS, lowest margin, etc.)
- Trend analysis
- Year-over-year comparisons
- Period aggregations
- Statistical analysis
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HistoricalFinding:
    """Container for a historical analysis result."""
    query_type: str  # "maximum", "minimum", "trend", "average", etc.
    metric_name: str
    result_value: float
    result_date: str  # YYYY or YYYYMMDD
    context: Dict  # Additional context data
    analysis: str  # Human-readable analysis
    ticker: str = ""
    data_points: int = 0


class HistoricalAnalyzer:
    """
    Analyze historical financial data to answer questions about:
    - When was metric highest/lowest?
    - What's the historical trend?
    - Average over time periods
    - Volatility analysis
    - Correlation between metrics
    """
    
    def __init__(self, xbrl_dir: str = "data/raw/xbrl_structured"):
        """Initialize with path to XBRL data."""
        self.xbrl_dir = xbrl_dir
        logger.info(f"Initialized Historical Analyzer with XBRL dir: {xbrl_dir}")
    
    # ═══════════════════════════════════════════════════════════════════
    # EXTREME VALUE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    
    def find_maximum(self, 
                     data: List[Tuple[str, float]],
                     metric_name: str,
                     ticker: str = "") -> HistoricalFinding:
        """
        Find when a metric was at its highest value.
        
        Args:
            data: List of (date, value) tuples
            metric_name: Name of the metric (e.g., "EPS", "Revenue")
            ticker: Company ticker
            
        Returns:
            HistoricalFinding with max value and date
        """
        if not data:
            return HistoricalFinding(
                query_type="maximum",
                metric_name=metric_name,
                result_value=0,
                result_date="N/A",
                context={"error": "No data available"},
                analysis=f"No historical data available for {metric_name}",
                ticker=ticker,
                data_points=0
            )
        
        # Sort by value descending
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
        max_date, max_value = sorted_data[0]
        
        # Calculate context
        values = [d[1] for d in data]
        avg_value = np.mean(values)
        
        # Find runner-ups
        top_3 = sorted_data[:3]
        
        return HistoricalFinding(
            query_type="maximum",
            metric_name=metric_name,
            result_value=max_value,
            result_date=max_date,
            context={
                "average": round(avg_value, 2),
                "min_value": round(min(values), 2),
                "top_3": [(d, round(v, 2)) for d, v in top_3],
                "percentile": 100.0
            },
            analysis=f"Highest {metric_name} was {max_value:,.2f} recorded on {max_date}. "
                    f"This is {((max_value/avg_value)-1)*100:.1f}% above the historical average of {avg_value:,.2f}.",
            ticker=ticker,
            data_points=len(data)
        )
    
    def find_minimum(self,
                     data: List[Tuple[str, float]],
                     metric_name: str,
                     ticker: str = "") -> HistoricalFinding:
        """
        Find when a metric was at its lowest value.
        """
        if not data:
            return HistoricalFinding(
                query_type="minimum",
                metric_name=metric_name,
                result_value=0,
                result_date="N/A",
                context={"error": "No data available"},
                analysis=f"No historical data available for {metric_name}",
                ticker=ticker,
                data_points=0
            )
        
        # Sort by value ascending
        sorted_data = sorted(data, key=lambda x: x[1])
        min_date, min_value = sorted_data[0]
        
        values = [d[1] for d in data]
        avg_value = np.mean(values)
        
        bottom_3 = sorted_data[:3]
        
        return HistoricalFinding(
            query_type="minimum",
            metric_name=metric_name,
            result_value=min_value,
            result_date=min_date,
            context={
                "average": round(avg_value, 2),
                "max_value": round(max(values), 2),
                "bottom_3": [(d, round(v, 2)) for d, v in bottom_3],
                "percentile": 0.0
            },
            analysis=f"Lowest {metric_name} was {min_value:,.2f} recorded on {min_date}. "
                    f"This is {((avg_value-min_value)/avg_value)*100:.1f}% below the historical average of {avg_value:,.2f}.",
            ticker=ticker,
            data_points=len(data)
        )
    
    def find_nth_highest(self,
                        data: List[Tuple[str, float]],
                        n: int,
                        metric_name: str,
                        ticker: str = "") -> HistoricalFinding:
        """Find the nth highest value."""
        if not data or n > len(data):
            return HistoricalFinding(
                query_type=f"{n}th_highest",
                metric_name=metric_name,
                result_value=0,
                result_date="N/A",
                context={"error": f"Insufficient data for {n}th highest"},
                analysis=f"Cannot find {n}th highest - only {len(data)} data points available",
                ticker=ticker,
                data_points=len(data)
            )
        
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
        date_val = sorted_data[n-1]
        
        return HistoricalFinding(
            query_type=f"{n}th_highest",
            metric_name=metric_name,
            result_value=date_val[1],
            result_date=date_val[0],
            context={"rank": n, "total_periods": len(data)},
            analysis=f"The {n}th highest {metric_name} was {date_val[1]:,.2f} on {date_val[0]}",
            ticker=ticker,
            data_points=len(data)
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # TREND ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    
    def analyze_trend(self,
                      data: List[Tuple[str, float]],
                      metric_name: str,
                      ticker: str = "") -> HistoricalFinding:
        """
        Analyze the overall trend in historical data.
        
        Determines if metric is:
        - Growing, declining, or stable
        - Trend strength (strong, moderate, weak)
        - Volatility
        """
        if len(data) < 2:
            return HistoricalFinding(
                query_type="trend",
                metric_name=metric_name,
                result_value=0,
                result_date="N/A",
                context={"error": "Insufficient data for trend analysis"},
                analysis="Need at least 2 data points for trend analysis",
                ticker=ticker,
                data_points=len(data)
            )
        
        # Sort by date
        sorted_data = sorted(data, key=lambda x: x[0])
        values = np.array([d[1] for d in sorted_data])
        
        # Calculate growth rate
        first_value = values[0]
        last_value = values[-1]
        total_change = (last_value - first_value) / abs(first_value) * 100 if first_value != 0 else 0
        
        # Linear regression for trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Determine trend direction
        if total_change > 10:
            direction = "upward" if slope > 0 else "upward (volatile)"
        elif total_change < -10:
            direction = "downward"
        else:
            direction = "stable"
        
        # Trend strength based on R-squared
        correlation = np.corrcoef(x, values)[0, 1]
        r_squared = correlation ** 2
        
        strength = "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "weak"
        
        # Volatility (coefficient of variation)
        cv = np.std(values) / np.mean(values) * 100
        volatility = "high" if cv > 30 else "moderate" if cv > 15 else "low"
        
        # Calculate CAGR if applicable
        years = len(values) - 1
        if first_value > 0 and years > 0:
            cagr = ((last_value / first_value) ** (1 / years) - 1) * 100
        else:
            cagr = 0
        
        return HistoricalFinding(
            query_type="trend",
            metric_name=metric_name,
            result_value=round(cagr, 2),
            result_date=f"{sorted_data[0][0]} to {sorted_data[-1][0]}",
            context={
                "direction": direction,
                "strength": strength,
                "volatility": volatility,
                "r_squared": round(r_squared, 3),
                "cagr": round(cagr, 2),
                "total_change_pct": round(total_change, 2),
                "slope": round(slope, 2)
            },
            analysis=f"{metric_name} shows a {strength} {direction} trend with {volatility} volatility. "
                    f"CAGR of {cagr:.1f}% over {years} years. Total change: {total_change:.1f}%.",
            ticker=ticker,
            data_points=len(data)
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # PERIOD ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_average(self,
                         data: List[Tuple[str, float]],
                         metric_name: str,
                         period: str = "all",
                         ticker: str = "") -> HistoricalFinding:
        """
        Calculate average over specified period.
        
        Args:
            data: List of (date, value) tuples
            metric_name: Name of metric
            period: "all", "5yr", "3yr", "last_year"
        """
        if not data:
            return HistoricalFinding(
                query_type="average",
                metric_name=metric_name,
                result_value=0,
                result_date="N/A",
                context={"error": "No data available"},
                analysis="No data available for average calculation",
                ticker=ticker,
                data_points=0
            )
        
        sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
        
        # Filter by period
        if period == "5yr":
            filtered_data = sorted_data[:5]
        elif period == "3yr":
            filtered_data = sorted_data[:3]
        elif period == "last_year":
            filtered_data = sorted_data[:1]
        else:
            filtered_data = sorted_data
        
        values = [d[1] for d in filtered_data]
        avg = np.mean(values)
        std = np.std(values)
        
        return HistoricalFinding(
            query_type="average",
            metric_name=metric_name,
            result_value=round(avg, 2),
            result_date=f"{filtered_data[-1][0]} to {filtered_data[0][0]}",
            context={
                "period": period,
                "std_dev": round(std, 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "data_points": len(filtered_data)
            },
            analysis=f"Average {metric_name} over {period}: {avg:,.2f} "
                    f"(range: {min(values):,.2f} to {max(values):,.2f})",
            ticker=ticker,
            data_points=len(filtered_data)
        )
    
    def year_over_year_change(self,
                              data: List[Tuple[str, float]],
                              metric_name: str,
                              ticker: str = "") -> List[HistoricalFinding]:
        """Calculate year-over-year changes for all consecutive periods."""
        if len(data) < 2:
            return []
        
        sorted_data = sorted(data, key=lambda x: x[0])
        results = []
        
        for i in range(1, len(sorted_data)):
            prev_date, prev_value = sorted_data[i-1]
            curr_date, curr_value = sorted_data[i]
            
            if prev_value != 0:
                change = ((curr_value - prev_value) / abs(prev_value)) * 100
            else:
                change = 0
            
            results.append(HistoricalFinding(
                query_type="yoy_change",
                metric_name=metric_name,
                result_value=round(change, 2),
                result_date=curr_date,
                context={
                    "previous_date": prev_date,
                    "previous_value": prev_value,
                    "current_value": curr_value
                },
                analysis=f"{metric_name} changed {change:+.1f}% from {prev_date} to {curr_date}",
                ticker=ticker,
                data_points=2
            ))
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════
    # COMPARISON ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    
    def compare_to_benchmark(self,
                            company_data: List[Tuple[str, float]],
                            benchmark_data: List[Tuple[str, float]],
                            metric_name: str,
                            ticker: str = "",
                            benchmark_name: str = "Industry Average") -> HistoricalFinding:
        """Compare company metric to a benchmark over time."""
        if not company_data or not benchmark_data:
            return HistoricalFinding(
                query_type="benchmark_comparison",
                metric_name=metric_name,
                result_value=0,
                result_date="N/A",
                context={"error": "Missing data"},
                analysis="Cannot perform comparison - data missing",
                ticker=ticker,
                data_points=0
            )
        
        # Get latest values
        company_latest = sorted(company_data, key=lambda x: x[0], reverse=True)[0]
        benchmark_latest = sorted(benchmark_data, key=lambda x: x[0], reverse=True)[0]
        
        company_avg = np.mean([d[1] for d in company_data])
        benchmark_avg = np.mean([d[1] for d in benchmark_data])
        
        diff_current = ((company_latest[1] - benchmark_latest[1]) / abs(benchmark_latest[1])) * 100
        diff_historical = ((company_avg - benchmark_avg) / abs(benchmark_avg)) * 100
        
        position = "above" if diff_current > 0 else "below"
        
        return HistoricalFinding(
            query_type="benchmark_comparison",
            metric_name=metric_name,
            result_value=round(diff_current, 2),
            result_date=company_latest[0],
            context={
                "company_value": company_latest[1],
                "benchmark_value": benchmark_latest[1],
                "company_avg": round(company_avg, 2),
                "benchmark_avg": round(benchmark_avg, 2),
                "historical_diff": round(diff_historical, 2)
            },
            analysis=f"{ticker}'s {metric_name} is {abs(diff_current):.1f}% {position} {benchmark_name}. "
                    f"Historically, {ticker} has averaged {abs(diff_historical):.1f}% {'above' if diff_historical > 0 else 'below'} the benchmark.",
            ticker=ticker,
            data_points=len(company_data)
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_statistics(self,
                            data: List[Tuple[str, float]],
                            metric_name: str,
                            ticker: str = "") -> HistoricalFinding:
        """Calculate comprehensive statistics for a metric."""
        if not data:
            return HistoricalFinding(
                query_type="statistics",
                metric_name=metric_name,
                result_value=0,
                result_date="N/A",
                context={"error": "No data"},
                analysis="No data available",
                ticker=ticker,
                data_points=0
            )
        
        values = np.array([d[1] for d in data])
        
        stats = {
            "mean": round(np.mean(values), 2),
            "median": round(np.median(values), 2),
            "std_dev": round(np.std(values), 2),
            "min": round(np.min(values), 2),
            "max": round(np.max(values), 2),
            "q1": round(np.percentile(values, 25), 2),
            "q3": round(np.percentile(values, 75), 2),
            "iqr": round(np.percentile(values, 75) - np.percentile(values, 25), 2),
            "cv": round(np.std(values) / np.mean(values) * 100, 2) if np.mean(values) != 0 else 0,
            "skewness": round(self._calculate_skewness(values), 2),
            "n": len(values)
        }
        
        return HistoricalFinding(
            query_type="statistics",
            metric_name=metric_name,
            result_value=stats["mean"],
            result_date=f"{len(values)} periods",
            context=stats,
            analysis=f"Statistical summary for {metric_name}: "
                    f"Mean={stats['mean']:,.2f}, Median={stats['median']:,.2f}, "
                    f"Std Dev={stats['std_dev']:,.2f}, Range=[{stats['min']:,.2f}, {stats['max']:,.2f}]",
            ticker=ticker,
            data_points=len(data)
        )
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        n = len(values)
        if n < 3:
            return 0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0
        return (np.sum((values - mean) ** 3) / n) / (std ** 3)
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPER: Load data from XBRL
    # ═══════════════════════════════════════════════════════════════════
    
    def load_metric_history(self, 
                           ticker: str, 
                           metric: str,
                           start_year: int = None,
                           end_year: int = None) -> List[Tuple[str, float]]:
        """
        Load historical data for a metric from XBRL files.
        
        Args:
            ticker: Company ticker
            metric: Metric name (e.g., "Revenue", "NetIncome", "EPS")
            start_year: Optional start year filter
            end_year: Optional end year filter
            
        Returns:
            List of (date, value) tuples
        """
        import pandas as pd
        from pathlib import Path
        
        metrics_path = Path(self.xbrl_dir) / f"{ticker}_metrics.csv"
        
        if not metrics_path.exists():
            logger.warning(f"No metrics file found for {ticker}")
            return []
        
        try:
            df = pd.read_csv(metrics_path)

            metric_df = df[df['metric'] == metric].copy() if metric else df.copy()

            if metric_df.empty:
                logger.warning(f"No data found for metric '{metric}' in {ticker}")
                return []

            metric_df['year'] = pd.to_datetime(metric_df['end_date']).dt.year

            if start_year:
                metric_df = metric_df[metric_df['year'] >= start_year]
            if end_year:
                metric_df = metric_df[metric_df['year'] <= end_year]

            metric_df = metric_df.sort_values(['year', 'filed_date'], ascending=[True, False])
            metric_df = metric_df.drop_duplicates(subset='year', keep='first')

            result = [(str(year), value) for year, value in zip(metric_df['year'], metric_df['value'])]
            return sorted(result, key=lambda x: x[0])

        except Exception as e:
            logger.error(f"Error loading metric history: {e}")
            return []
    
    def format_finding(self, finding: HistoricalFinding) -> str:
        """Format a historical finding for display."""
        lines = [
            f"📊 Historical Analysis: {finding.metric_name}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Query Type: {finding.query_type.replace('_', ' ').title()}",
            f"Ticker: {finding.ticker}" if finding.ticker else "",
            f"Result: {finding.result_value:,.2f}",
            f"Date/Period: {finding.result_date}",
            f"Data Points: {finding.data_points}",
            "",
            f"📝 Analysis:",
            finding.analysis,
        ]
        
        if finding.context and "error" not in finding.context:
            lines.append("")
            lines.append("📋 Additional Context:")
            for key, value in finding.context.items():
                if isinstance(value, float):
                    lines.append(f"  • {key.replace('_', ' ').title()}: {value:,.2f}")
                elif isinstance(value, list):
                    lines.append(f"  • {key.replace('_', ' ').title()}: {value}")
                else:
                    lines.append(f"  • {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join([l for l in lines if l])
