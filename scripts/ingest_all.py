"""
Unified Data Ingestion CLI for TemporalGuard-RAG

This script provides a single command to collect as much data as possible using
the repo's existing collectors:
- SEC filings (10-K/10-Q/8-K)
- SEC XBRL company facts
- Stock prices (Yahoo)
- Yahoo Finance fundamentals / statements
- Investor relations documents (semi-automated)
- Earnings transcripts (optional, requires API key or local Kaggle dataset)

Note: Some sources require network access and/or API keys.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from src.data_collection import (
    SECDataCollector,
    XBRLCollector,
    StockDataCollector,
    TranscriptScraper,
    IRDocumentCollector,
)
from src.data_collection.yahoo_finance import YahooFinanceCollector

load_dotenv()


def _load_tickers(args_tickers: List[str], company_list_path: str) -> List[str]:
    if args_tickers:
        return [t.strip().upper() for t in args_tickers if t.strip()]

    path = Path(company_list_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ticker list not found at {company_list_path}. "
            f"Provide tickers via --tickers or create data/company_list.json."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tickers: List[str] = []
    if isinstance(data, dict):
        for _, values in data.items():
            tickers.extend(values)
    elif isinstance(data, list):
        tickers = data

    return [t.strip().upper() for t in tickers if t.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="TemporalGuard-RAG data ingestion")
    parser.add_argument("--tickers", nargs="*", default=[], help="Tickers like AAPL MSFT JPM")
    parser.add_argument("--company-list", default="data/company_list.json", help="Path to ticker list JSON")

    parser.add_argument("--sec", action="store_true", help="Download SEC filings")
    parser.add_argument("--sec-years", type=int, default=5, help="Years/limits for SEC downloader")
    parser.add_argument("--sec-types", nargs="*", default=["10-K", "10-Q"], help="SEC filing types to download")
    parser.add_argument("--sec-email", default=os.getenv("SEC_EMAIL", ""), help="SEC email for User-Agent")

    parser.add_argument("--xbrl", action="store_true", help="Download SEC XBRL company facts")
    parser.add_argument("--yahoo", action="store_true", help="Fetch Yahoo Finance fundamentals/statements")
    parser.add_argument("--prices", action="store_true", help="Download historical stock prices (CSV)")

    parser.add_argument("--ir", action="store_true", help="Discover official IR documents (semi-automated)")
    parser.add_argument("--ir-download", action="store_true", help="Download IR documents where possible")
    parser.add_argument("--ir-max-docs", type=int, default=30, help="Max IR documents per company")

    parser.add_argument("--transcripts", action="store_true", help="Download earnings transcripts (FMP)")
    parser.add_argument("--transcript-years", nargs="*", type=int, default=[], help="Years for transcripts, e.g. 2023 2024")
    parser.add_argument("--kaggle-transcripts", default="", help="Load transcripts from a local Kaggle folder instead")

    parser.add_argument("--all", action="store_true", help="Run SEC, XBRL, Yahoo, prices, and IR discovery")
    args = parser.parse_args()

    tickers = _load_tickers(args.tickers, args.company_list)

    if args.all:
        args.sec = True
        args.xbrl = True
        args.yahoo = True
        args.prices = True
        args.ir = True

    # Basic safety: avoid accidental huge runs.
    if len(tickers) > 250 and (args.ir or args.transcripts):
        raise SystemExit(
            "Refusing to run IR/transcript collection on >250 tickers by default. "
            "Start smaller, then scale."
        )

    if args.sec:
        if not args.sec_email:
            raise SystemExit("SEC download requires --sec-email or SEC_EMAIL env var.")
        sec = SECDataCollector(company_email=args.sec_email)
        for filing_type in args.sec_types:
            sec.download_filings(tickers, filing_type=filing_type, years=args.sec_years)

    if args.xbrl:
        xbrl = XBRLCollector(user_agent=f"TemporalGuardRAG {args.sec_email or 'your.email@example.com'}")
        xbrl.download_all_companies(tickers)

    if args.prices:
        prices = StockDataCollector()
        prices.download_multiple_stocks(tickers)

    if args.yahoo:
        yf = YahooFinanceCollector()
        yf.fetch_multiple(tickers)

    if args.ir:
        ir = IRDocumentCollector()
        ir.collect_multiple(
            tickers,
            download=args.ir_download,
            max_documents=args.ir_max_docs,
        )

    if args.transcripts or args.kaggle_transcripts:
        scraper = TranscriptScraper()
        if args.kaggle_transcripts:
            scraper.load_kaggle_transcripts(args.kaggle_transcripts)
        else:
            years = args.transcript_years or []
            scraper.download_all_transcripts(tickers, years=years or None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

