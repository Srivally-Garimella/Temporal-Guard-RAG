"""
FastAPI Backend for TemporalGuard-RAG
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import asyncio
import uuid
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="TemporalGuard-RAG API",
    description="Multi-Agent Financial Analysis with Look-Ahead Bias Prevention",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
orchestrator = None
vector_store = None
jobs: Dict[str, Dict] = {}


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z0-9.-]+$")
    analysis_date: str = Field(pattern=r"^\d{8}$")
    provider: str = "ollama"
    model_name: str = "llama3.2"
    fast_mode: bool = True  # Skip slow LLM verification by default

    @field_validator("analysis_date")
    @classmethod
    def validate_analysis_date(cls, value):
        datetime.strptime(value, "%Y%m%d")
        return value


class QueryResponse(BaseModel):
    job_id: str
    status: str
    message: str


class AnalysisResult(BaseModel):
    job_id: str
    status: str
    query: str
    ticker: str
    analysis_date: str
    current_stage: Optional[int] = 0
    stage_name: Optional[str] = None
    final_answer: Optional[str] = None
    stages: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def get_orchestrator(provider: str = "ollama", model_name: str = "llama3.2"):
    """Initialize orchestrator lazily."""
    global orchestrator, vector_store
    
    if vector_store is None:
        from src.rag_system.vector_store import TemporalVectorStore
        vector_store = TemporalVectorStore()
    
    if orchestrator is None:
        from src.agents.orchestrator import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator(
            vector_store=vector_store,
            provider=provider,
            model_name=model_name
        )
    
    return orchestrator


async def run_analysis(job_id: str, query: str, ticker: str, analysis_date: str,
                       provider: str, model_name: str, fast_mode: bool = True):
    """Run analysis in background."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.now().isoformat()
        jobs[job_id]["current_stage"] = 0
        jobs[job_id]["stage_name"] = "Initializing"
        
        orch = get_orchestrator(provider, model_name)
        
        # Progress callback to update job status
        def progress_callback(stage_num, stage_name):
            jobs[job_id]["current_stage"] = stage_num
            jobs[job_id]["stage_name"] = stage_name
        
        # Run in thread pool (blocking operation)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: orch.process_query(
                query=query,
                ticker=ticker,
                analysis_date=analysis_date,
                verbose=False,
                progress_callback=progress_callback,
                fast_mode=fast_mode
            )
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        logger.exception("Analysis job %s failed", job_id)
        jobs[job_id]["error"] = "Analysis failed due to an internal error."


@app.get("/")
async def root():
    return {
        "name": "TemporalGuard-RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/tickers")
async def get_tickers():
    """Get available company tickers."""
    return {
        "tickers": [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "GS", "name": "Goldman Sachs Group Inc."},
            {"symbol": "XOM", "name": "Exxon Mobil Corporation"},
            {"symbol": "CVX", "name": "Chevron Corporation"}
        ]
    }


@app.get("/api/models")
async def get_models():
    """Get available LLM models."""
    try:
        from src.agents.llm_provider import list_ollama_models
        ollama_models = list_ollama_models()
    except:
        ollama_models = ["llama3.2"]
    
    return {
        "providers": [
            {
                "id": "ollama",
                "name": "Ollama (Local)",
                "models": ollama_models
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "requires_key": True
            }
        ]
    }


@app.get("/api/stats")
async def get_stats():
    """Get vector store statistics."""
    try:
        if vector_store is None:
            from src.rag_system.vector_store import TemporalVectorStore
            vs = TemporalVectorStore()
            count = vs.collection.count()
        else:
            count = vector_store.collection.count()
        
        return {
            "document_count": count,
            "status": "ready"
        }
    except Exception as e:
        logger.exception("Could not read vector store statistics")
        return {
            "document_count": 0,
            "status": "error",
            "error": "Vector store statistics are unavailable."
        }


@app.post("/api/analyze", response_model=QueryResponse)
async def analyze(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit analysis query (async)."""
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "queued",
        "query": request.query,
        "ticker": request.ticker,
        "analysis_date": request.analysis_date,
        "created_at": datetime.now().isoformat()
    }
    
    background_tasks.add_task(
        run_analysis,
        job_id,
        request.query,
        request.ticker,
        request.analysis_date,
        request.provider,
        request.model_name,
        request.fast_mode
    )
    
    return QueryResponse(
        job_id=job_id,
        status="queued",
        message="Analysis started"
    )


@app.get("/api/jobs/{job_id}", response_model=AnalysisResult)
async def get_job(job_id: str):
    """Get job status and results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return AnalysisResult(
        job_id=job_id,
        status=job["status"],
        query=job["query"],
        ticker=job["ticker"],
        analysis_date=job["analysis_date"],
        current_stage=job.get("current_stage", 0),
        stage_name=job.get("stage_name"),
        final_answer=job.get("result", {}).get("final_answer"),
        stages=job.get("result", {}).get("stages"),
        metadata=job.get("result", {}).get("metadata"),
        error=job.get("error")
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return {
        "jobs": [
            {
                "job_id": jid,
                "status": j["status"],
                "ticker": j["ticker"],
                "created_at": j["created_at"]
            }
            for jid, j in jobs.items()
        ]
    }


# ═══════════════════════════════════════════════════════════════
# Real-Time Market Data Endpoints
# ═══════════════════════════════════════════════════════════════

market_data_agent = None

def get_market_agent():
    """Lazy load market data agent."""
    global market_data_agent
    if market_data_agent is None:
        from src.agents.market_data_agent import MarketDataAgent
        market_data_agent = MarketDataAgent()
    return market_data_agent


@app.get("/api/market/price/{ticker}")
async def get_live_price(ticker: str):
    """Get real-time stock price."""
    agent = get_market_agent()
    return agent.get_live_price(ticker.upper())


@app.get("/api/market/history/{ticker}")
async def get_price_history(ticker: str, period: str = "1mo", interval: str = "1d"):
    """Get historical price data."""
    agent = get_market_agent()
    return agent.get_historical_prices(ticker.upper(), period, interval)


@app.get("/api/market/info/{ticker}")
async def get_company_info(ticker: str):
    """Get company information and fundamentals."""
    agent = get_market_agent()
    return agent.get_company_info(ticker.upper())


@app.get("/api/market/financials/{ticker}")
async def get_financials(ticker: str):
    """Get financial statements."""
    agent = get_market_agent()
    return agent.get_financials(ticker.upper())


@app.get("/api/market/analyst/{ticker}")
async def get_analyst_data(ticker: str):
    """Get analyst recommendations and price targets."""
    agent = get_market_agent()
    return agent.get_analyst_recommendations(ticker.upper())


@app.get("/api/market/earnings/{ticker}")
async def get_earnings_estimates(ticker: str):
    """Get earnings estimates and projections."""
    agent = get_market_agent()
    return agent.get_earnings_estimates(ticker.upper())


@app.get("/api/market/forecast/{ticker}")
async def get_forward_guidance(ticker: str):
    """Get comprehensive forward-looking projections and analyst targets."""
    agent = get_market_agent()
    return agent.get_forward_guidance(ticker.upper())


@app.get("/api/market/compare")
async def compare_stocks(tickers: str):
    """Compare multiple stocks. Tickers should be comma-separated."""
    agent = get_market_agent()
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    return agent.compare_stocks(ticker_list)


@app.post("/api/market/question")
async def answer_market_question(request: dict):
    """Answer a market-related question with real-time data."""
    agent = get_market_agent()
    return agent.answer_market_question(
        question=request.get("question", ""),
        ticker=request.get("ticker")
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
