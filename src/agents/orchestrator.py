"""
Multi-Agent Orchestrator for TemporalGuard-RAG

Coordinates multiple specialized agents to answer complex financial queries
with temporal consistency and fact verification.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging
import os
import time

from .llm_provider import get_llm
from .document_agent import DocumentRetrievalAgent
from .calculation_agent import CalculationAgent
from .verification_agent import VerificationAgent
from .temporal_agent import TemporalAgent
from .analysis_agent import FinancialAnalysisAgent
from ..rag_system.vector_store import TemporalVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Query type patterns
PROJECTION_PATTERNS = [
    'project', 'forecast', 'predict', 'expected', 'estimate',
    'will be', 'going to be', 'future'
]
HISTORICAL_EXTREME_PATTERNS = [
    'highest', 'lowest', 'maximum', 'minimum', 'max', 'min', 
    'peak', 'record', 'when was', 'best', 'worst'
]
RATIO_PATTERNS = [
    'roe', 'roa', 'roic', 'margin', 'ratio', 'multiple',
    'p/e', 'pe ratio', 'debt to equity', 'current ratio'
]


class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents for comprehensive financial analysis.
    
    Agents:
    - TemporalAgent: Validates temporal consistency (FIRST)
    - DocumentRetrievalAgent: Retrieves relevant documents
    - CalculationAgent: Performs financial calculations
    - VerificationAgent: Fact-checks results (LAST)
    
    Workflow:
    1. Temporal Validation → Ensure query is temporally consistent
    2. Document Retrieval → Get relevant documents
    3. Calculation → Compute financial metrics
    4. Verification → Fact-check claims
    5. Synthesis → Combine results into final answer
    """
    
    SYNTHESIS_PROMPT = """Answer the user's financial question directly and concisely.

Query: {query}
Analysis Date: {analysis_date}

AGENT DATA:
{agent_outputs}

INSTRUCTIONS:
1. Start with the DIRECT ANSWER to the question in 1-2 sentences
2. Then provide 2-3 key supporting points if relevant
3. Keep the response under 200 words
4. Use actual numbers ONLY if they appear in the agent data above
5. If data is insufficient, say: "The available data does not contain this information."

Format:
**Answer:** [Direct answer to the question]

**Key Points:**
- [Supporting point 1]
- [Supporting point 2]

Keep it simple and readable."""

    def __init__(self,
                 vector_store=None,
                 xbrl_dir: str = "data/raw/xbrl_structured",
                 model_name: str = None,
                 provider: str = None):
        """
        Initialize Multi-Agent Orchestrator.
        
        Args:
            vector_store: TemporalVectorStore for document retrieval
            xbrl_dir: Directory with XBRL structured data
            model_name: LLM model name (default: auto-select)
            provider: 'openai' or 'ollama' (default: auto-detect)
        """
        self.model_name = model_name
        self.provider = provider
        self.xbrl_dir = xbrl_dir

        # If a vector store isn't provided, use the local persistent one by default.
        # This avoids the system silently falling back to mock retrieval.
        if vector_store is None:
            vector_store = TemporalVectorStore()
        
        # Initialize LLM for synthesis using provider factory
        self.llm = get_llm(provider=provider, model_name=model_name, temperature=0)
        
        # Initialize agents
        logger.info("Initializing agent team...")
        
        self.temporal_agent = TemporalAgent(model_name=model_name, provider=provider)
        
        self.document_agent = DocumentRetrievalAgent(
            vector_store=vector_store,
            model_name=model_name,
            provider=provider
        )
        
        self.calculation_agent = CalculationAgent(
            xbrl_dir=xbrl_dir,
            model_name=model_name,
            provider=provider
        )
        
        self.verification_agent = VerificationAgent(
            vector_store=vector_store,
            xbrl_dir=xbrl_dir,
            model_name=model_name,
            provider=provider
        )
        
        # New: Analysis Agent for projections and advanced analysis
        self.analysis_agent = FinancialAnalysisAgent(
            xbrl_dir=xbrl_dir,
            model_name=model_name,
            provider=provider
        )
        
        logger.info("Multi-Agent Orchestrator initialized with 5 agents (including Analysis Agent)")

    def _classify_query(self, query: str) -> str:
        """
        Classify the query type to route to appropriate pipeline.
        
        Returns:
            'projection' - Forward-looking queries (forecasts, projections)
            'historical_extreme' - Historical max/min queries
            'ratio' - Financial ratio calculations
            'historical' - Standard historical RAG queries
        """
        query_lower = query.lower()
        
        # Check for projection/forecast queries
        if any(pattern in query_lower for pattern in PROJECTION_PATTERNS):
            return 'projection'
        
        # Check for historical extreme queries
        if any(pattern in query_lower for pattern in HISTORICAL_EXTREME_PATTERNS):
            return 'historical_extreme'
        
        # Check for ratio queries  
        if any(pattern in query_lower for pattern in RATIO_PATTERNS):
            return 'ratio'
        
        # Default to standard RAG pipeline
        return 'historical'
    
    def _infer_analysis_date(self, query: str) -> str:
        """
        Infer a reasonable analysis date from query context.
        
        Logic:
        - If query mentions a specific quarter (e.g., Q3 2023), use ~45 days after quarter end
        - If query mentions a year (e.g., 2023), use March of following year (after 10-K filing)
        - Otherwise default to today
        
        Args:
            query: User's financial query
            
        Returns:
            Analysis date in YYYYMMDD format
        """
        import re
        from datetime import timedelta
        
        query_upper = query.upper()
        now = datetime.now()
        
        # Quarter patterns: Q1 2023, Q2 2024, etc.
        quarter_match = re.search(r'Q([1-4])\s*(\d{4})', query_upper)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            year = int(quarter_match.group(2))
            # Quarter end dates + 45 days for filing lag
            quarter_ends = {
                1: (3, 31),   # Q1 ends March 31
                2: (6, 30),   # Q2 ends June 30
                3: (9, 30),   # Q3 ends September 30
                4: (12, 31)   # Q4 ends December 31
            }
            month, day = quarter_ends[quarter]
            # Add ~45 days for 10-Q filing lag
            quarter_end = datetime(year, month, day)
            analysis_dt = quarter_end + timedelta(days=45)
            return analysis_dt.strftime('%Y%m%d')
        
        # Fiscal year patterns: FY2023, fiscal 2023, 2023 annual
        year_match = re.search(r'(?:FY|FISCAL\s*)?(\d{4})(?:\s*(?:ANNUAL|YEAR|10-K))?', query_upper)
        if year_match:
            year = int(year_match.group(1))
            current_year = now.year
            # If it's a past year, use March of following year (after 10-K filing)
            if year < current_year:
                return f"{year + 1}0301"
            # If current year, use today
            elif year == current_year:
                return now.strftime('%Y%m%d')
        
        # Default to today
        return now.strftime('%Y%m%d')
        
    def process_query(self, 
                      query: str, 
                      ticker: str,
                      analysis_date: str = None,
                      verbose: bool = True,
                      progress_callback=None,
                      fast_mode: bool = False) -> Dict[str, Any]:
        """
        Process a financial query through the multi-agent pipeline.
        
        Args:
            query: User's financial query
            ticker: Company ticker symbol
            analysis_date: Point-in-time date (YYYYMMDD). Optional - defaults to today.
                          Use for backtesting: "What would I have known on this date?"
            verbose: Print progress updates
            progress_callback: Optional callback(stage_num, stage_name) for progress tracking
            fast_mode: Skip slow LLM verification for faster results
            
        Returns:
            Comprehensive analysis results
        """
        # Default analysis_date to today if not specified
        if analysis_date is None:
            analysis_date = self._infer_analysis_date(query)
        self._fast_mode = fast_mode  # Store for use in _run_verification
        
        def update_progress(stage_num, stage_name):
            if progress_callback:
                progress_callback(stage_num, stage_name)
            if verbose:
                print(f"\n{'='*40}")
                print(f"Stage {stage_num}: {stage_name}")
                print(f"{'='*40}")
        start_time = datetime.now()
        results = {
            'query': query,
            'ticker': ticker,
            'analysis_date': analysis_date,
            'stages': {},
            'final_answer': None,
            'metadata': {}
        }
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: Temporal Validation (Mandatory for all queries)
        # ═══════════════════════════════════════════════════════════════
        update_progress(1, "Temporal Validation")
            
        # Classify query and route appropriately
        query_type = self._classify_query(query)
        results['metadata']['query_type'] = query_type
        temporal_result = self._run_temporal_validation(
            query,
            analysis_date,
            allow_future_target=query_type == 'projection',
        )
        results['stages']['temporal'] = temporal_result

        if not temporal_result.get('is_valid', False):
            reason = temporal_result.get('bias_detection') or temporal_result.get('error')
            results['final_answer'] = (
                "Analysis blocked because temporal validation could not confirm that "
                f"the requested information was available as of {analysis_date}. {reason or ''}"
            ).strip()
            results['final'] = self._wrap_final_answer(
                query_type=query_type,
                analysis_date=analysis_date,
                final_answer=results['final_answer'],
                stages=results['stages'],
            )
            results['metadata'].update({
                'temporal_rejected': True,
                'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'model': self.model_name,
                'agents_used': ['temporal'],
            })
            return results
        
        # Route projection/forecast queries to Analysis Agent
        if query_type in ['projection', 'historical_extreme', 'ratio']:
            return self._process_analysis_query(
                query=query,
                ticker=ticker,
                query_type=query_type,
                analysis_date=analysis_date,
                verbose=verbose,
                progress_callback=progress_callback,
                start_time=start_time,
                temporal_result=temporal_result
            )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 MULTI-AGENT FINANCIAL ANALYSIS")
            print(f"{'='*60}")
            print(f"Query: {query}")
            print(f"Ticker: {ticker}")
            print(f"Analysis Date: {analysis_date}")
            print(f"{'='*60}\n")
            
        status = "✅ VALID" if temporal_result['is_valid'] else "❌ INVALID"
        if verbose:
            print(f"Status: {status}")
            if not temporal_result['is_valid']:
                print("⚠️ Temporal violations detected - proceeding with caution")
            print()
            
        # Get effective cutoff date
        cutoff_date = self.temporal_agent.get_cutoff_date(analysis_date)
        results['metadata']['effective_cutoff'] = cutoff_date
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: Document Retrieval
        # ═══════════════════════════════════════════════════════════════
        update_progress(2, "Document Retrieval")
            
        doc_result = self._run_document_retrieval(query, ticker, cutoff_date)
        results['stages']['document_retrieval'] = doc_result
        
        if verbose:
            doc_count = doc_result.get('document_count', 0)
            print(f"Retrieved {doc_count} relevant documents")
            print()
            
        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: Financial Calculations
        # ═══════════════════════════════════════════════════════════════
        update_progress(3, "Financial Calculations")
            
        calc_result = self._run_calculations(query, ticker, analysis_date)
        results['stages']['calculations'] = calc_result
        
        if verbose:
            print("Calculated relevant financial metrics")
            print()
            
        # ═══════════════════════════════════════════════════════════════
        # STAGE 4: Fact Verification
        # ═══════════════════════════════════════════════════════════════
        update_progress(4, "Fact Verification")
            
        verification_result = self._run_verification(query, ticker, cutoff_date)
        results['stages']['verification'] = verification_result
        
        if verbose:
            print("Claims verified against source documents")
            print()
            
        # ═══════════════════════════════════════════════════════════════
        # STAGE 5: Synthesis
        # ═══════════════════════════════════════════════════════════════
        update_progress(5, "Synthesizing Results")
            
        final_answer = self._synthesize_results(
            query=query,
            analysis_date=analysis_date,
            stages=results['stages']
        )
        results['final_answer'] = final_answer
        results['final'] = self._wrap_final_answer(
            query_type=query_type,
            analysis_date=analysis_date,
            final_answer=final_answer,
            stages=results.get("stages", {}),
        )
        
        # ═══════════════════════════════════════════════════════════════
        # Finalize
        # ═══════════════════════════════════════════════════════════════
        processing_time = (datetime.now() - start_time).total_seconds()
        results['metadata']['processing_time_seconds'] = processing_time
        results['metadata']['timestamp'] = datetime.now().isoformat()
        results['metadata']['model'] = self.model_name
        results['metadata']['agents_used'] = ['temporal', 'document', 'calculation', 'verification']
        
        if verbose:
            print()
            print(f"{'='*60}")
            print(f"📋 FINAL ANALYSIS")
            print(f"{'='*60}")
            print(final_answer)
            print(f"{'='*60}")
            print(f"\n⏱️ Total processing time: {processing_time:.2f}s")
            
        return results

    def _wrap_final_answer(self, query_type: str, analysis_date: str, final_answer: str, stages: Dict) -> Dict[str, Any]:
        """
        Standardize the final output envelope so the UI/report can reliably show:
        - what kind of answer this is (fact vs calculation vs estimate)
        - what date semantics were used
        - any important warnings
        """
        answer_type_map = {
            "historical": "reported_fact",
            "historical_extreme": "calculated_metric",
            "ratio": "calculated_metric",
            "projection": "model_estimate",
        }
        answer_type = answer_type_map.get(query_type, "unknown")

        warnings: List[str] = []
        temporal = stages.get("temporal", {}) if isinstance(stages, dict) else {}
        if temporal and not temporal.get("is_valid", True):
            warnings.append("Temporal warnings/violations detected; results may not be point-in-time safe.")

        doc = stages.get("document_retrieval", {}) if isinstance(stages, dict) else {}
        if doc and doc.get("document_count", 0) == 0:
            warnings.append("No documents retrieved for the cutoff; answer may be incomplete.")

        return {
            "answer_type": answer_type,
            "analysis_date": analysis_date,
            "answer": final_answer,
            "warnings": warnings,
        }
        
    def _run_temporal_validation(self, query: str, analysis_date: str,
                                 allow_future_target: bool = False) -> Dict:
        """Run temporal validation."""
        try:
            return self.temporal_agent.validate_query(
                query,
                analysis_date,
                allow_future_target=allow_future_target,
            )
        except Exception as e:
            logger.error(f"Temporal validation error: {e}")
            return {
                'is_valid': False,
                'has_violations': True,
                'error': str(e),
                'bias_detection': 'Temporal validation failed; analysis was blocked.',
            }
            
    def _run_document_retrieval(self, query: str, ticker: str, cutoff_date: str) -> Dict:
        """Run document retrieval."""
        try:
            # DocumentRetrievalAgent expects (query, cutoff_date, ticker)
            result = self.document_agent.retrieve(query, cutoff_date, ticker)
            
            # Count documents
            if 'output' in result:
                result['document_count'] = result['output'].count('Document')
            else:
                result['document_count'] = 0
            
            # Explicitly flag when no relevant documents found
            if result['document_count'] == 0:
                result['output'] = (
                    f"NO RELEVANT DOCUMENTS FOUND for query '{query}' with ticker={ticker} "
                    f"before cutoff date {cutoff_date}. The system cannot provide factual information "
                    "without supporting documents. Please verify the ticker symbol and date range."
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return {
                'output': f"Error retrieving documents: {e}",
                'document_count': 0,
                'error': str(e)
            }
            
    def _run_calculations(self, query: str, ticker: str, analysis_date: str) -> Dict:
        """Run financial calculations based on query."""
        try:
            # Determine what calculations are needed based on query
            query_lower = query.lower()
            
            calculations = {}
            
            # Check for specific metrics in query
            if any(term in query_lower for term in ['revenue', 'sales', 'growth']):
                calc = self.calculation_agent.calculate(
                    ticker=ticker,
                    metric="revenue_growth",
                    date=analysis_date
                )
                calculations['revenue_growth'] = calc
                
            if any(term in query_lower for term in ['profit', 'margin', 'earning']):
                calc = self.calculation_agent.calculate(
                    ticker=ticker,
                    metric="profit_margin",
                    date=analysis_date
                )
                calculations['profit_margin'] = calc
                
            if any(term in query_lower for term in ['roe', 'return on equity']):
                calc = self.calculation_agent.calculate(
                    ticker=ticker,
                    metric="ROE",
                    date=analysis_date
                )
                calculations['ROE'] = calc
                
            if any(term in query_lower for term in ['debt', 'leverage']):
                calc = self.calculation_agent.calculate(
                    ticker=ticker,
                    metric="debt_ratio",
                    date=analysis_date
                )
                calculations['debt_ratio'] = calc
                
            # If no specific metrics found, calculate common ones
            if not calculations:
                for metric_name in ['revenue_growth', 'profit_margin', 'ROE']:
                    calc = self.calculation_agent.calculate(
                        ticker=ticker,
                        metric=metric_name,
                        date=analysis_date
                    )
                    calculations[metric_name] = calc
                    
            return {
                'output': self._format_calculations(calculations),
                'calculations': calculations,
                'ticker': ticker,
                'analysis_date': analysis_date
            }
            
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return {
                'output': f"Error performing calculations: {e}",
                'error': str(e)
            }
            
    def _format_calculations(self, calculations: Dict) -> str:
        """Format calculation results."""
        lines = ["Financial Metrics Calculated:"]
        
        for metric, result in calculations.items():
            output = result.get('output', str(result))
            lines.append(f"\n{metric}:")
            lines.append(output[:500])  # Truncate for readability
            
        return "\n".join(lines)
        
    def _run_verification(self, query: str, ticker: str, cutoff_date: str) -> Dict:
        """Run fact verification."""
        try:
            fast_mode = getattr(self, '_fast_mode', False)
            return self.verification_agent.verify(query, ticker, cutoff_date, fast_mode=fast_mode)
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                'output': f"Error during verification: {e}",
                'error': str(e)
            }
            
    def _synthesize_results(self, query: str, analysis_date: str, stages: Dict) -> str:
        """Synthesize all agent outputs into final answer."""
        # Compile agent outputs
        agent_outputs = []
        
        # Temporal validation
        temporal = stages.get('temporal', {})
        agent_outputs.append(f"""TEMPORAL AGENT:
- Valid: {temporal.get('is_valid', 'Unknown')}
- Effective Cutoff Applied: Yes
{temporal.get('bias_detection', '')[:500]}
""")
        
        # Document retrieval
        doc = stages.get('document_retrieval', {})
        agent_outputs.append(f"""DOCUMENT AGENT:
- Documents Retrieved: {doc.get('document_count', 0)}
{doc.get('output', '')[:800]}
""")
        
        # Calculations
        calc = stages.get('calculations', {})
        agent_outputs.append(f"""CALCULATION AGENT:
{calc.get('output', '')[:800]}
""")
        
        # Verification
        verif = stages.get('verification', {})
        agent_outputs.append(f"""VERIFICATION AGENT:
{verif.get('output', '')[:500]}
""")
        
        # Use LLM for synthesis if available
        if self.llm:
            try:
                prompt = self.SYNTHESIS_PROMPT.format(
                    query=query,
                    analysis_date=analysis_date,
                    agent_outputs="\n\n".join(agent_outputs)
                )
                
                response = self.llm.invoke(prompt)
                return response.content
                
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
                
        # Fallback: manual synthesis
        return self._manual_synthesis(query, analysis_date, stages, agent_outputs)
        
    def _manual_synthesis(self, query: str, analysis_date: str, 
                           stages: Dict, agent_outputs: List[str]) -> str:
        """Manual synthesis when LLM unavailable - keeps it simple."""
        # Check for severe temporal violation first
        temporal = stages.get('temporal', {})
        if not temporal.get('is_valid', True) and "FUTURE" in temporal.get('bias_detection', '').upper():
            return f"⚠️ TEMPORAL VIOLATION DETECTED: {temporal.get('bias_detection', 'This query references data that would not be available at the analysis date.')}"

        lines = []
        
        # Direct answer attempt from calculations
        calc = stages.get('calculations', {})
        if calc.get('calculations'):
            lines.append("**Results:**")
            for metric, result in list(calc['calculations'].items())[:3]:  # Max 3 metrics
                output = result.get('output', '')
                # Extract just the key value
                for line in output.split('\n'):
                    if any(x in line for x in ['Result:', 'Value:', '=']):
                        lines.append(f"• {metric}: {line.split(':')[-1].strip()[:50]}")
                        break
            lines.append("")
        
        # Document context
        doc = stages.get('document_retrieval', {})
        doc_count = doc.get('document_count', 0)
        if doc_count > 0:
            lines.append(f"Based on {doc_count} SEC documents.")
        else:
            lines.append("Limited source documents available.")
        
        # Temporal note (brief)
        if not temporal.get('is_valid', True):
            lines.append("⚠️ Note: Query may reference future data.")
        
        return "\n".join(lines) if lines else "Analysis completed. See details below."

    def _process_analysis_query(self,
                                  query: str,
                                  ticker: str,
                                  query_type: str,
                                  analysis_date: str,
                                  verbose: bool = True,
                                  progress_callback=None,
                                  start_time=None,
                                  temporal_result: Dict = None) -> Dict[str, Any]:
        """
        Process projection/forecast/historical extreme queries through Analysis Agent.
        
        This is a specialized pipeline for forward-looking queries that don't fit
        the standard RAG pattern (which is point-in-time historical).
        """
        if start_time is None:
            start_time = datetime.now()
            
        def update_progress(stage_num, stage_name):
            if progress_callback:
                progress_callback(stage_num, stage_name)
            if verbose:
                print(f"\n{'='*40}")
                print(f"Stage {stage_num}: {stage_name}")
                print(f"{'='*40}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 FINANCIAL ANALYSIS (Analysis Agent Pipeline)")
            print(f"{'='*60}")
            print(f"Query: {query}")
            print(f"Ticker: {ticker}")
            print(f"Query Type: {query_type.upper()}")
            print(f"{'='*60}\n")
        
        results = {
            'query': query,
            'ticker': ticker,
            'analysis_date': analysis_date,
            'stages': {
                'temporal': temporal_result or {'is_valid': True}
            },
            'final_answer': None,
            'metadata': {
                'query_type': query_type,
                'pipeline': 'analysis_agent'
            }
        }
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: Data Loading
        # ═══════════════════════════════════════════════════════════════
        update_progress(1, "Loading Financial Data")
        
        try:
            self.analysis_agent._analysis_date = analysis_date
            data_summary = self.analysis_agent._load_financial_data(ticker)
            results['stages']['data_loading'] = {
                'output': data_summary,
                'status': 'success'
            }
            if verbose:
                print(f"Loaded financial data for {ticker}")
        except Exception as e:
            results['stages']['data_loading'] = {
                'output': str(e),
                'status': 'error'
            }
            
        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: Analysis Execution
        # ═══════════════════════════════════════════════════════════════
        update_progress(2, f"Executing {query_type.replace('_', ' ').title()} Analysis")
        
        try:
            analysis_result = self.analysis_agent.analyze(query, ticker, analysis_date=analysis_date)
            results['stages']['analysis'] = analysis_result
            
            if verbose:
                print("Analysis completed successfully")
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            results['stages']['analysis'] = {
                'output': f"Error during analysis: {e}",
                'status': 'error'
            }
            
        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: Market Context (for projections)
        # ═══════════════════════════════════════════════════════════════
        if query_type == 'projection' and analysis_date == datetime.now().strftime('%Y%m%d'):
            update_progress(3, "Adding Market Context")
            
            try:
                market_data = self.analysis_agent._get_market_data(ticker)
                results['stages']['market_context'] = {
                    'output': market_data,
                    'status': 'success'
                }
                if verbose:
                    print("Added current market data for context")
            except Exception as e:
                results['stages']['market_context'] = {
                    'output': str(e),
                    'status': 'error'
                }
        else:
            update_progress(3, "Trend Analysis")
            # For historical queries, add trend context
            try:
                # Try to extract the metric being asked about
                query_lower = query.lower()
                metric = "Revenue"  # default
                if "eps" in query_lower:
                    metric = "EPS_Diluted"
                elif "income" in query_lower or "profit" in query_lower:
                    metric = "NetIncome"
                elif "cash" in query_lower:
                    metric = "OperatingCashFlow"
                    
                trend_data = self.analysis_agent._analyze_trend(f"{ticker}|{metric}")
                results['stages']['trend_context'] = {
                    'output': trend_data,
                    'status': 'success'
                }
            except Exception as e:
                results['stages']['trend_context'] = {
                    'output': str(e),
                    'status': 'error'
                }
                
        # ═══════════════════════════════════════════════════════════════
        # STAGE 4: Compile Results
        # ═══════════════════════════════════════════════════════════════
        update_progress(4, "Compiling Results")
        
        # Extract the main analysis output
        analysis_stage = results['stages'].get('analysis', {})
        main_output = analysis_stage.get('output', '')
        
        # Build final answer
        final_lines = [
            f"📊 Analysis Results for {ticker.upper()}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Query Type: {query_type.replace('_', ' ').title()}",
            "",
            main_output,
        ]
        
        # Add temporal warning if invalid
        if temporal_result and not temporal_result.get('is_valid', True):
            final_lines.insert(3, f"⚠️ TEMPORAL WARNING: {temporal_result.get('bias_detection', 'Potential look-ahead bias detected.')}")
            final_lines.insert(4, "")
        
        # Add market context if available
        market_stage = results['stages'].get('market_context', {})
        if market_stage.get('status') == 'success':
            final_lines.extend([
                "",
                "─" * 40,
                "📈 Current Market Context:",
                market_stage.get('output', '')[:500]
            ])
        
        # Add trend context if available
        trend_stage = results['stages'].get('trend_context', {})
        if trend_stage.get('status') == 'success':
            final_lines.extend([
                "",
                "─" * 40,
                "📉 Historical Trend:",
                trend_stage.get('output', '')[:500]
            ])
        
        results['final_answer'] = "\n".join(final_lines)
        results['final'] = self._wrap_final_answer(
            query_type=query_type,
            analysis_date=analysis_date,
            final_answer=results["final_answer"],
            stages=results.get("stages", {}),
        )
        
        # ═══════════════════════════════════════════════════════════════
        # Finalize
        # ═══════════════════════════════════════════════════════════════
        processing_time = (datetime.now() - start_time).total_seconds()
        results['metadata']['processing_time_seconds'] = processing_time
        results['metadata']['timestamp'] = datetime.now().isoformat()
        results['metadata']['agents_used'] = ['analysis']
        
        if verbose:
            print()
            print(f"{'='*60}")
            print(f"📋 FINAL ANALYSIS")
            print(f"{'='*60}")
            print(results['final_answer'])
            print(f"{'='*60}")
            print(f"\n⏱️ Total processing time: {processing_time:.2f}s")
            
        return results
        
    def run_single_agent(self, agent_name: str, **kwargs) -> Dict:
        """
        Run a single agent for specific tasks.
        
        Args:
            agent_name: Name of agent ('temporal', 'document', 'calculation', 'verification')
            **kwargs: Agent-specific parameters
            
        Returns:
            Agent output
        """
        agents = {
            'temporal': lambda: self.temporal_agent.validate_query(
                kwargs.get('query', ''),
                kwargs.get('analysis_date', '')
            ),
            'document': lambda: self.document_agent.retrieve(
                kwargs.get('query', ''),
                kwargs.get('filing_date', ''),
                kwargs.get('ticker', '')
            ),
            'calculation': lambda: self.calculation_agent.calculate(
                kwargs.get('ticker', ''),
                kwargs.get('ratio', 'ROE'),
                kwargs.get('date', '')
            ),
            'verification': lambda: self.verification_agent.verify(
                kwargs.get('claim', ''),
                kwargs.get('ticker', ''),
                kwargs.get('filing_date', '')
            ),
            'analysis': lambda: self.analysis_agent.analyze(
                kwargs.get('query', ''),
                kwargs.get('ticker', '')
            )
        }
        
        if agent_name not in agents:
            return {'error': f'Unknown agent: {agent_name}'}
            
        try:
            return agents[agent_name]()
        except Exception as e:
            return {'error': str(e)}


class FinancialQuery:
    """Helper class for constructing financial queries."""
    
    def __init__(self, 
                 question: str,
                 ticker: str,
                 analysis_date: str,
                 metrics: Optional[List[str]] = None):
        self.question = question
        self.ticker = ticker
        self.analysis_date = analysis_date
        self.metrics = metrics or []
        
    def to_dict(self) -> Dict:
        return {
            'query': self.question,
            'ticker': self.ticker,
            'analysis_date': self.analysis_date,
            'metrics': self.metrics
        }


# Usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Example query
    result = orchestrator.process_query(
        query="What was Apple's revenue growth and profit margin in fiscal 2023?",
        ticker="AAPL",
        analysis_date="20231201",
        verbose=True
    )
    
    # Save results
    with open('analysis_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
