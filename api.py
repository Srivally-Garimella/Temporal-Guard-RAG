"""
FastAPI Application for TemporalGuard-RAG

REST API for financial analysis with temporal consistency.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TemporalGuard-RAG API",
    description="Multi-agent financial analysis with look-ahead bias prevention",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """Request model for financial query."""
    query: str = Field(..., description="Financial query to analyze")
    ticker: str = Field(..., description="Company ticker symbol", max_length=10)
    analysis_date: Optional[str] = Field(
        default=None, 
        description="Point-in-time date (YYYYMMDD). Optional - defaults to today. Use for backtesting: 'What would I have known on this date?'"
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional configuration"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What was Apple's revenue growth in fiscal 2023?",
                "ticker": "AAPL",
                "analysis_date": None,
                "options": {
                    "include_calculations": True,
                    "verbose": False
                }
            }
        }


class TemporalValidationRequest(BaseModel):
    """Request for temporal validation only."""
    query: str = Field(..., description="Query to validate")
    analysis_date: Optional[str] = Field(
        default=None,
        description="Point-in-time date (YYYYMMDD). Defaults to today if not specified."
    )


class CalculationRequest(BaseModel):
    """Request for financial calculation."""
    ticker: str = Field(..., description="Company ticker")
    ratio: str = Field(..., description="Ratio to calculate (ROE, ROA, debt_ratio, etc.)")
    date: str = Field(..., description="As-of date (YYYYMMDD)")


class VerificationRequest(BaseModel):
    """Request for claim verification."""
    claim: str = Field(..., description="Claim to verify")
    ticker: str = Field(..., description="Company ticker")
    filing_date: str = Field(..., description="Filing date cutoff")


class QueryResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    query: str
    ticker: str
    analysis_date: str
    result: Optional[Dict[str, Any]]
    processing_time_seconds: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str


# ═══════════════════════════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════════════════════════

orchestrator = None
vector_store = None
audit_logger = None


def get_orchestrator():
    """Get or initialize orchestrator."""
    global orchestrator, vector_store
    if orchestrator is None:
        try:
            # Use the persisted local vector store by default so retrieval works out of the box.
            if vector_store is None:
                from src.rag_system.vector_store import TemporalVectorStore
                vector_store = TemporalVectorStore()

            from src.agents.orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(vector_store=vector_store)
            logger.info("Orchestrator initialized")
        except Exception as e:
            logger.error(f"Could not initialize orchestrator: {e}")
    return orchestrator


def get_audit_logger():
    """Get or initialize audit logger."""
    global audit_logger
    if audit_logger is None:
        try:
            from src.security.audit_logger import AuditLogger
            audit_logger = AuditLogger()
            logger.info("Audit logger initialized")
        except Exception as e:
            logger.warning(f"Could not initialize audit logger: {e}")
    return audit_logger


# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/analyze", response_model=QueryResponse)
async def analyze_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Run full multi-agent financial analysis.
    
    This endpoint processes a financial query through the multi-agent pipeline:
    1. Temporal validation (prevent look-ahead bias)
    2. Document retrieval (SEC filings, XBRL data)
    3. Financial calculations
    4. Fact verification
    5. Result synthesis
    
    Note: analysis_date is optional. If not provided:
    - Defaults to today for most queries
    - Auto-infers from query context (e.g., "Q3 2023" → ~45 days after Q3 end)
    """
    start_time = datetime.now()
    
    # Use provided analysis_date or let orchestrator infer it
    effective_date = request.analysis_date or datetime.now().strftime('%Y%m%d')
    
    # Log query
    logger_instance = get_audit_logger()
    if logger_instance:
        logger_instance.log_query(
            query=request.query,
            analysis_date=effective_date,
            ticker=request.ticker
        )
    
    # Get orchestrator
    orch = get_orchestrator()
    if orch is None:
        raise HTTPException(
            status_code=503,
            detail="Analysis system not available"
        )
    
    try:
        # Run analysis (orchestrator handles None analysis_date)
        result = orch.process_query(
            query=request.query,
            ticker=request.ticker,
            analysis_date=request.analysis_date,  # Can be None
            verbose=request.options.get('verbose', False) if request.options else False
        )
        
        # Get actual analysis_date used (may have been inferred)
        actual_date = result.get('analysis_date', effective_date)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            success=True,
            query=request.query,
            ticker=request.ticker,
            analysis_date=actual_date,
            result=result,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/v1/validate-temporal")
async def validate_temporal(request: TemporalValidationRequest):
    """
    Validate query for temporal consistency.
    
    Checks if the query contains:
    - References to future dates
    - Forward-looking language
    - Potential look-ahead bias
    """
    try:
        from src.agents.temporal_agent import TemporalAgent
        agent = TemporalAgent()
        
        result = agent.validate_query(request.query, request.analysis_date)
        
        return {
            "success": True,
            "is_valid": result.get('is_valid', False),
            "has_warnings": result.get('has_warnings', False),
            "has_violations": result.get('has_violations', False),
            "analysis": result.get('bias_detection', ''),
            "cutoff_info": result.get('cutoff_analysis', '')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/calculate")
async def calculate_ratio(request: CalculationRequest):
    """
    Calculate financial ratio for a company.
    
    Available ratios:
    - ROE (Return on Equity)
    - ROA (Return on Assets)
    - debt_ratio
    - profit_margin
    - current_ratio
    - revenue_growth
    """
    try:
        from src.agents.calculation_agent import CalculationAgent
        agent = CalculationAgent()
        
        result = agent.calculate(
            ticker=request.ticker,
            ratio=request.ratio,
            date=request.date
        )
        
        return {
            "success": True,
            "ticker": request.ticker,
            "ratio": request.ratio,
            "date": request.date,
            "result": result.get('output', ''),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/verify")
async def verify_claim(request: VerificationRequest):
    """
    Verify a financial claim against source documents.
    
    Returns confidence score and supporting evidence.
    """
    try:
        from src.agents.verification_agent import VerificationAgent
        agent = VerificationAgent()
        
        result = agent.verify(
            claim=request.claim,
            ticker=request.ticker,
            filing_date=request.filing_date
        )
        
        return {
            "success": True,
            "claim": request.claim,
            "ticker": request.ticker,
            "filing_date": request.filing_date,
            "verification": result.get('output', ''),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tickers")
async def list_available_tickers():
    """List tickers with available data."""
    # Check data directories for available tickers
    data_dir = Path("data/raw")
    
    tickers = set()
    
    if data_dir.exists():
        # Check SEC filings
        sec_dir = data_dir / "sec_filings"
        if sec_dir.exists():
            for ticker_dir in sec_dir.iterdir():
                if ticker_dir.is_dir():
                    tickers.add(ticker_dir.name)
                    
        # Check XBRL data
        xbrl_dir = data_dir / "xbrl_structured"
        if xbrl_dir.exists():
            for file in xbrl_dir.glob("*_metrics.csv"):
                ticker = file.stem.replace("_metrics", "")
                tickers.add(ticker)
    
    return {
        "tickers": sorted(list(tickers)),
        "count": len(tickers)
    }


@app.get("/api/v1/audit/recent")
async def get_recent_audit_events(
    limit: int = Query(default=50, le=500),
    event_type: Optional[str] = None
):
    """Get recent audit events."""
    logger_instance = get_audit_logger()
    
    if logger_instance is None:
        return {"events": [], "count": 0}
        
    try:
        from src.security.audit_logger import EventType
        
        event_type_enum = None
        if event_type:
            try:
                event_type_enum = EventType(event_type)
            except ValueError:
                pass
                
        events = logger_instance.get_recent_events(
            limit=limit,
            event_type=event_type_enum
        )
        
        return {
            "events": [e.to_dict() for e in events],
            "count": len(events)
        }
        
    except Exception as e:
        logger.error(f"Could not fetch audit events: {e}")
        return {"events": [], "count": 0, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Error Handlers
# ═══════════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# ═══════════════════════════════════════════════════════════════
# Startup/Shutdown
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting TemporalGuard-RAG API...")
    
    # Pre-initialize components
    get_orchestrator()
    get_audit_logger()
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down TemporalGuard-RAG API...")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
