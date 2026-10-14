import { useState, useEffect, useCallback } from 'react'
import {
  Search, Clock, FileText, Calculator, ShieldCheck, Loader2,
  TrendingUp, Building2, Calendar, CheckCircle2, AlertCircle,
  XCircle, ChevronDown, Activity, Database, Cpu, ArrowUpRight, ArrowDownRight, DollarSign
} from 'lucide-react'
import clsx from 'clsx'

// Types
interface Ticker {
  symbol: string
  name: string
}

interface LivePrice {
  ticker: string
  price: number
  change: number
  change_percent: number
  market_cap: number
  volume: number
  timestamp: string
  status: string
}

interface AnalysisResult {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  query: string
  ticker: string
  analysis_date: string
  current_stage?: number
  stage_name?: string
  final_answer?: string
  stages?: {
    temporal?: { is_valid: boolean }
    document_retrieval?: { document_count: number; output?: string }
    calculations?: { output?: string }
    verification?: { output?: string }
  }
  metadata?: {
    processing_time_seconds?: number
    model?: string
  }
  error?: string
}

interface Stats {
  document_count: number
  status: string
}

// Components
function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx('bg-white rounded-2xl shadow-lg border border-gray-100', className)}>
      {children}
    </div>
  )
}

function Button({
  children,
  onClick,
  disabled,
  variant = 'primary',
  className = '',
  loading = false
}: {
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  variant?: 'primary' | 'secondary'
  className?: string
  loading?: boolean
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={clsx(
        'px-6 py-3 rounded-xl font-semibold transition-all duration-200 flex items-center justify-center gap-2',
        variant === 'primary' && 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 shadow-lg shadow-blue-500/30',
        variant === 'secondary' && 'bg-gray-100 text-gray-700 hover:bg-gray-200',
        (disabled || loading) && 'opacity-50 cursor-not-allowed',
        className
      )}
    >
      {loading && <Loader2 className="w-5 h-5 animate-spin" />}
      {children}
    </button>
  )
}

function StageIndicator({
  stage,
  label,
  icon: Icon,
  status
}: {
  stage: number
  label: string
  icon: React.ElementType
  status: 'pending' | 'running' | 'done' | 'error'
}) {
  return (
    <div className="flex items-center gap-3">
      <div className={clsx(
        'w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300',
        status === 'pending' && 'bg-gray-100 text-gray-400',
        status === 'running' && 'bg-blue-100 text-blue-600 animate-pulse',
        status === 'done' && 'bg-green-100 text-green-600',
        status === 'error' && 'bg-red-100 text-red-600'
      )}>
        {status === 'running' ? (
          <Loader2 className="w-5 h-5 animate-spin" />
        ) : status === 'done' ? (
          <CheckCircle2 className="w-5 h-5" />
        ) : status === 'error' ? (
          <XCircle className="w-5 h-5" />
        ) : (
          <Icon className="w-5 h-5" />
        )}
      </div>
      <div>
        <p className={clsx(
          'text-sm font-medium',
          status === 'pending' && 'text-gray-400',
          status === 'running' && 'text-blue-600',
          status === 'done' && 'text-green-600',
          status === 'error' && 'text-red-600'
        )}>
          Stage {stage}
        </p>
        <p className="text-xs text-gray-500">{label}</p>
      </div>
    </div>
  )
}

function ResultCard({ result }: { result: AnalysisResult }) {
  const [expanded, setExpanded] = useState(false)
  
  return (
    <Card className="animate-fadeIn overflow-hidden">
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-semibold">
                {result.ticker}
              </span>
              <span className="text-sm text-gray-500">
                {result.analysis_date.slice(0, 4)}-{result.analysis_date.slice(4, 6)}-{result.analysis_date.slice(6, 8)}
              </span>
            </div>
            <p className="text-gray-600 text-sm">{result.query}</p>
          </div>
          {result.metadata?.processing_time_seconds && (
            <span className="text-xs text-gray-400">
              {result.metadata.processing_time_seconds.toFixed(1)}s
            </span>
          )}
        </div>
      </div>
      
      <div className="p-6 bg-gradient-to-br from-gray-50 to-white">
        <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">
          Analysis Result
        </h4>
        <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-wrap">
          {result.final_answer || 'No answer available'}
        </div>
      </div>
      
      {result.stages && (
        <div className="border-t border-gray-100">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full px-6 py-3 flex items-center justify-between text-sm text-gray-500 hover:bg-gray-50 transition-colors"
          >
            <span>View Stage Details</span>
            <ChevronDown className={clsx('w-4 h-4 transition-transform', expanded && 'rotate-180')} />
          </button>
          
          {expanded && (
            <div className="px-6 pb-6 space-y-4 animate-fadeIn">
              {result.stages.temporal && (
                <div className="p-4 bg-gray-50 rounded-xl">
                  <h5 className="font-medium text-gray-700 mb-2 flex items-center gap-2">
                    <Clock className="w-4 h-4" /> Temporal Validation
                  </h5>
                  <p className={clsx(
                    'text-sm',
                    result.stages.temporal.is_valid ? 'text-green-600' : 'text-red-600'
                  )}>
                    {result.stages.temporal.is_valid ? '✓ Valid - No look-ahead bias' : '⚠ Potential issues detected'}
                  </p>
                </div>
              )}
              
              {result.stages.document_retrieval && (
                <div className="p-4 bg-gray-50 rounded-xl">
                  <h5 className="font-medium text-gray-700 mb-2 flex items-center gap-2">
                    <FileText className="w-4 h-4" /> Document Retrieval
                  </h5>
                  <p className="text-sm text-gray-600">
                    Retrieved {result.stages.document_retrieval.document_count} documents
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

function LivePriceWidget({ price, loading = false }: { price: LivePrice | null; loading?: boolean }) {
  
  if (loading) {
    return (
      <div className="bg-gradient-to-r from-gray-100 to-gray-50 rounded-2xl p-4 animate-pulse">
        <div className="h-6 w-24 bg-gray-200 rounded mb-2" />
        <div className="h-8 w-32 bg-gray-200 rounded" />
      </div>
    )
  }

  if (!price || price.status !== 'success') {
    return (
      <div className="bg-gradient-to-r from-gray-100 to-gray-50 rounded-2xl p-4 animate-pulse">
        <div className="h-6 w-20 bg-gray-200 rounded mb-2" />
        <div className="h-8 w-32 bg-gray-200 rounded" />
      </div>
    )
  }

  const isPositive = price.change >= 0
  
  return (
    <div className={clsx(
      'rounded-2xl p-4 transition-all duration-300',
      isPositive 
        ? 'bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200' 
        : 'bg-gradient-to-br from-red-50 to-rose-50 border border-red-200'
    )}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <DollarSign className={clsx('w-4 h-4', isPositive ? 'text-green-600' : 'text-red-600')} />
          <span className="text-sm font-semibold text-gray-600">Live Price</span>
        </div>
        <span className="text-xs text-gray-400">
          {new Date(price.timestamp).toLocaleTimeString()}
        </span>
      </div>
      
      <div className="flex items-end gap-3">
        <span className="text-3xl font-bold text-gray-800">
          ${price.price?.toLocaleString()}
        </span>
        <div className={clsx(
          'flex items-center gap-1 px-2 py-1 rounded-lg text-sm font-semibold',
          isPositive ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
        )}>
          {isPositive ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
          {isPositive ? '+' : ''}{price.change_percent}%
        </div>
      </div>
      
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-gray-500">
        <div>
          <span className="block text-gray-400">Market Cap</span>
          <span className="font-medium">${price.market_cap ? (price.market_cap / 1e9).toFixed(1) + 'B' : 'N/A'}</span>
        </div>
        <div>
          <span className="block text-gray-400">Volume</span>
          <span className="font-medium">{price.volume ? (price.volume / 1e6).toFixed(1) + 'M' : 'N/A'}</span>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [ticker, setTicker] = useState('AAPL')
  const [analysisDate, setAnalysisDate] = useState('2025-10-01')
  const [tickers, setTickers] = useState<Ticker[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [currentStage, setCurrentStage] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [livePrice, setLivePrice] = useState<LivePrice | null>(null)

  // Fetch live price when ticker changes
  const fetchLivePrice = useCallback(async (t: string) => {
    setLoading(true)
    try {
      const res = await fetch(`/api/market/price/${t}`)
      const data = await res.json()
      setLivePrice(data)
    } catch (err) {
      console.error('Failed to fetch live price:', err)
    }
    setLoading(false)
  }, [])

  // Fetch price on ticker change
  useEffect(() => {
    if (ticker) {
      fetchLivePrice(ticker)
    }
  }, [ticker, fetchLivePrice])

  // Auto-refresh price every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (ticker && !loading) {
        fetchLivePrice(ticker)
      }
    }, 30000)
    return () => clearInterval(interval)
  }, [ticker, loading, fetchLivePrice])

  // Fetch initial data
  useEffect(() => {
    fetch('/api/tickers')
      .then(res => res.json())
      .then(data => setTickers(data.tickers))
      .catch(console.error)
    
    fetch('/api/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(console.error)
  }, [])

  // Poll for job status
  const pollJob = useCallback(async (jobId: string) => {
    if (!jobId) {
      setError('Analysis request did not return a job ID')
      setLoading(false)
      return
    }

    let attempts = 0
    const maxAttempts = 240 // 8 minutes max (local LLM is slow)
    
    while (attempts < maxAttempts) {
      try {
        const res = await fetch(`/api/jobs/${jobId}`)
        const data: AnalysisResult = await res.json()
        
        // Update stage based on status
        if (data.status === 'running') {
          setCurrentStage(data.current_stage || 1)
        }
        
        if (data.status === 'completed') {
          setResult(data)
          setLoading(false)
          setCurrentStage(5)
          return
        }
        
        if (data.status === 'failed') {
          setError(data.error || 'Analysis failed')
          setLoading(false)
          return
        }
        
        await new Promise(resolve => setTimeout(resolve, 2000))
        attempts++
      } catch (err) {
        console.error('Poll error:', err)
      }
    }
    
    setError('Analysis timed out')
    setLoading(false)
  }, [])

  const handleSubmit = async () => {
    if (!query.trim()) return
    
    setLoading(true)
    setError(null)
    setResult(null)
    setCurrentStage(1)
    
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          ticker,
          analysis_date: analysisDate.replace(/-/g, ''),
          provider: 'ollama',
          model_name: 'llama3.2'
        })
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data?.error || 'Failed to submit query')
      }

      if (!data?.job_id) {
        throw new Error('Backend did not return a job ID')
      }

      pollJob(data.job_id)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to submit query'
      setError(message)
      setLoading(false)
    }
  }

  const stages = [
    { label: 'Temporal', icon: Clock },
    { label: 'Documents', icon: FileText },
    { label: 'Calculate', icon: Calculator },
    { label: 'Verify', icon: ShieldCheck },
    { label: 'Synthesize', icon: TrendingUp }
  ]

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="glass sticky top-0 z-50 border-b border-gray-200/50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">TemporalGuard-RAG</h1>
                <p className="text-xs text-gray-500">Financial Analysis without Look-Ahead Bias</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {stats && (
                <div className="flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-xl">
                  <Database className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-medium">{stats.document_count.toLocaleString()} docs</span>
                </div>
              )}
              <div className="flex items-center gap-2 px-4 py-2 bg-green-100 rounded-xl">
                <Cpu className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-green-700">Ollama</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Query Input */}
        <Card className="mb-8 card-hover">
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Financial Query
                </label>
                <div className="relative">
                  <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    placeholder="e.g., What was Apple's revenue growth in 2023?"
                    className="w-full pl-12 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                    onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Building2 className="inline w-4 h-4 mr-1" /> Company
                </label>
                <select
                  value={ticker}
                  onChange={e => setTicker(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {tickers.map(t => (
                    <option key={t.symbol} value={t.symbol}>
                      {t.symbol} - {t.name}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Calendar className="inline w-4 h-4 mr-1" /> Analysis Date
                </label>
                <input
                  type="date"
                  value={analysisDate}
                  onChange={e => setAnalysisDate(e.target.value)}
                  max={new Date().toISOString().split('T')[0]}
                  className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500">
                <Clock className="inline w-4 h-4 mr-1" />
                Only data available before your analysis date will be used
              </p>
              <Button onClick={handleSubmit} disabled={!query.trim()} loading={loading}>
                {loading ? 'Analyzing...' : 'Run Analysis'}
              </Button>
            </div>
          </div>
        </Card>

        {/* Live Price Widget */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="md:col-span-1">
            <LivePriceWidget price={livePrice} loading={loading} />
          </div>
          <div className="md:col-span-2">
            <Card className="h-full">
              <div className="p-4">
                <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" /> Quick Actions
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  <button 
                    onClick={() => setQuery(`What was ${ticker}'s revenue last year?`)}
                    className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-xl text-sm text-gray-700 transition-colors"
                  >
                    📊 Revenue Analysis
                  </button>
                  <button 
                    onClick={() => setQuery(`What is ${ticker}'s profit margin trend?`)}
                    className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-xl text-sm text-gray-700 transition-colors"
                  >
                    💰 Profit Margins
                  </button>
                  <button 
                    onClick={() => setQuery(`What are the key risks for ${ticker}?`)}
                    className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-xl text-sm text-gray-700 transition-colors"
                  >
                    ⚠️ Risk Factors
                  </button>
                  <button 
                    onClick={() => setQuery(`How does ${ticker}'s ROE compare to industry?`)}
                    className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-xl text-sm text-gray-700 transition-colors"
                  >
                    📈 ROE Analysis
                  </button>
                </div>
              </div>
            </Card>
          </div>
        </div>

        {/* Progress Stages */}
        {loading && (
          <Card className="mb-8 animate-fadeIn">
            <div className="p-6">
              <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-600" />
                Multi-Agent Analysis in Progress
              </h3>
              <div className="flex items-center justify-between">
                {stages.map((stage, i) => (
                  <div key={i} className="flex-1 flex items-center">
                    <StageIndicator
                      stage={i + 1}
                      label={stage.label}
                      icon={stage.icon}
                      status={
                        currentStage > i + 1 ? 'done' :
                        currentStage === i + 1 ? 'running' : 'pending'
                      }
                    />
                    {i < stages.length - 1 && (
                      <div className={clsx(
                        'flex-1 h-0.5 mx-4 rounded',
                        currentStage > i + 1 ? 'bg-green-300' : 'bg-gray-200'
                      )} />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </Card>
        )}

        {/* Error */}
        {error && (
          <Card className="mb-8 border-red-200 bg-red-50 animate-fadeIn">
            <div className="p-6 flex items-center gap-3">
              <AlertCircle className="w-6 h-6 text-red-500" />
              <div>
                <h4 className="font-semibold text-red-700">Analysis Error</h4>
                <p className="text-sm text-red-600">{error}</p>
              </div>
            </div>
          </Card>
        )}

        {/* Results */}
        {result && <ResultCard result={result} />}

        {/* Empty State */}
        {!loading && !result && !error && (
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center">
              <Search className="w-10 h-10 text-blue-500" />
            </div>
            <h3 className="text-xl font-semibold text-gray-700 mb-2">Ready for Analysis</h3>
            <p className="text-gray-500 max-w-md mx-auto">
              Enter a financial query above to analyze SEC filings with temporal consistency.
              Our multi-agent system prevents look-ahead bias by only using data available at your specified date.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white/50 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between text-sm text-gray-500">
          <span>TemporalGuard-RAG v1.0.0</span>
          <span>Powered by Ollama + LLaMA 3.2</span>
        </div>
      </footer>
    </div>
  )
}
