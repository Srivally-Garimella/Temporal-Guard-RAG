"""
Temporal Vector Store for TemporalGuard-RAG

ChromaDB-based vector store with temporal filtering for look-ahead bias prevention.
Supports point-in-time queries and temporal metadata filtering.
"""

import chromadb
from chromadb.config import Settings
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import os
import re
import hashlib
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _DeterministicHashEmbedder:
    """
    Offline-safe embedder fallback.

    Produces deterministic, normalized vectors using feature hashing.
    This is not comparable in quality to real embeddings, but keeps the system
    functional when the sentence-transformers model can't be loaded (e.g. no
    network access and no local cache).
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = int(dimensions)

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
        if not tokens:
            return vec

        for token in tokens[:500]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = -1.0 if (digest[4] & 1) else 1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


class TemporalVectorStore:
    """
    Vector store with temporal awareness for financial documents.
    
    Key Features:
    - Point-in-time (PiT) queries
    - Temporal metadata filtering
    - Look-ahead bias prevention
    - Company and filing type filters
    """
    
    def __init__(self, 
                 persist_dir: str = "data/processed/vector_db",
                 collection_name: str = "temporal_financial_docs",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Temporal Vector Store.
        
        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            embedding_model: Model name for query embeddings
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model for queries.
        # Prefer local cache when running in restricted/offline environments.
        self.embedder = None
        self.embedding_model = embedding_model
        self.embedding_dim = int(os.getenv("TEMPORAL_GUARD_EMBED_DIM", "384"))
        self._init_embedder()
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Get or create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.collection_name = collection_name
        
        logger.info(f"Initialized Temporal Vector Store")
        logger.info(f"Persist dir: {self.persist_dir}")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Current count: {self.collection.count()}")

    def _init_embedder(self):
        allow_download = os.getenv("TEMPORAL_GUARD_ALLOW_MODEL_DOWNLOAD", "").strip() in {"1", "true", "yes"}
        logger.info(f"Loading embedding model: {self.embedding_model}")
        try:
            # Default to local-only to avoid long network hangs; allow override via env var.
            self.embedder = SentenceTransformer(
                self.embedding_model,
                local_files_only=not allow_download,
            )
            return
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer model ({self.embedding_model}): {e}")

        # Fallback embedder keeps the system running offline.
        logger.warning("Falling back to deterministic hash embeddings (offline-safe, lower quality retrieval).")
        self.embedder = _DeterministicHashEmbedder(dimensions=self.embedding_dim)
        
    def add_chunks_with_temporal_metadata(self,
                                         chunks_path: str = "data/processed/chunks/temporal_chunks.jsonl",
                                         embeddings_path: str = "data/processed/embeddings/embeddings.npy",
                                         batch_size: int = 500):
        """
        Add chunks to vector store with temporal filtering support.
        
        Args:
            chunks_path: Path to temporal chunks JSONL
            embeddings_path: Path to embeddings numpy file
            batch_size: Batch size for adding to ChromaDB
        """
        chunks_path = Path(chunks_path)
        embeddings_path = Path(embeddings_path)
        
        # Load chunks
        logger.info(f"Loading chunks from {chunks_path}...")

        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)

        with open(chunks_path, 'r', encoding='utf-8') as chunk_counter:
            chunk_count = sum(1 for _ in chunk_counter)

        if len(embeddings) != chunk_count:
            raise ValueError(f"Mismatch: {chunk_count} chunks, {len(embeddings)} embeddings")

        logger.info(f"Adding {len(embeddings)} chunks to vector store...")
        
        # Avoid retaining the full memory footprint of the chunk corpus when
        # ingesting to ChromaDB in batches. We only need a streaming window.
        with open(chunks_path, 'r', encoding='utf-8') as chunks_file:
            ids = []
            documents = []
            metadatas = []
            embedding_list = []
            processed_count = 0

            for chunk_line in chunks_file:
                chunk = json.loads(chunk_line)
                processed_count += 1

                if processed_count > len(embeddings):
                    raise ValueError(f"Chunk count exceeds embedding count: {processed_count} > {len(embeddings)}")

                ids.append(chunk['chunk_id'])
                documents.append(chunk['text'])

                filing_date_str = str(chunk.get('filing_date', ''))
                try:
                    filing_date_int = int(filing_date_str) if filing_date_str else 0
                except ValueError:
                    filing_date_int = 0

                metadata = {
                    'ticker': str(chunk.get('ticker', '')),
                    'filing_type': str(chunk.get('filing_type', '')),
                    'filing_date': filing_date_int,
                    'filing_date_str': filing_date_str,
                    'fiscal_year': str(chunk.get('fiscal_year', '')),
                    'fiscal_period': str(chunk.get('fiscal_period', '')),
                    'source_path': str(chunk.get('source_path', '')),
                    'chunk_index': int(chunk.get('chunk_index', 0)),
                    'chunk_hash': str(chunk.get('chunk_hash', ''))
                }
                metadatas.append(metadata)
                embedding_list.append(embeddings[processed_count - 1].tolist())

                if len(ids) >= batch_size:
                    self.collection.add(
                        ids=ids,
                        embeddings=embedding_list,
                        documents=documents,
                        metadatas=metadatas
                    )
                    logger.info(f"Added batch of {len(ids)} chunks")
                    ids = []
                    documents = []
                    metadatas = []
                    embedding_list = []

            if ids:
                self.collection.add(
                    ids=ids,
                    embeddings=embedding_list,
                    documents=documents,
                    metadatas=metadatas
                )
                
        logger.info(f"✅ Vector store created with {self.collection.count()} chunks")
        
        # Save configuration
        config = {
            'collection_name': self.collection_name,
            'persist_dir': str(self.persist_dir),
            'total_chunks': chunk_count,
            'created_at': datetime.now().isoformat(),
            'chunks_source': str(chunks_path),
            'embeddings_source': str(embeddings_path)
        }
        
        config_path = self.persist_dir / "vector_store_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    def temporal_search(self,
                       query: str,
                       cutoff_date: str = None,
                       ticker: str = None,
                       filing_type: str = None,
                       n_results: int = 5,
                       query_embeddings: List[float] = None) -> Dict:
        """
        Search with temporal constraints.
        
        CRITICAL for look-ahead bias prevention!
        
        Args:
            query: Search query text
            cutoff_date: Only return documents filed BEFORE this date (YYYYMMDD format)
            ticker: Filter by specific company ticker
            filing_type: Filter by filing type (10-K, 10-Q, etc.)
            n_results: Number of results to return
            query_embeddings: Pre-computed query embeddings (optional)
            
        Returns:
            Dictionary containing search results with metadata
        """
        # Build where filter
        where_filter = None
        where_conditions = []
        
        # Apply temporal filter (CRITICAL!)
        if cutoff_date:
            # Convert to integer for comparison
            cutoff_int = int(cutoff_date) if isinstance(cutoff_date, str) else cutoff_date
            where_conditions.append({"filing_date": {"$lt": cutoff_int}})
            logger.debug(f"Applying temporal filter: filing_date < {cutoff_int}")
            
        # Apply ticker filter
        if ticker:
            where_conditions.append({"ticker": ticker})
            
        # Apply filing type filter
        if filing_type:
            where_conditions.append({"filing_type": filing_type})
            
        # Combine conditions with AND
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}
            
        # Execute search
        try:
            if query_embeddings:
                embeddings_to_use = query_embeddings
            else:
                # Use the configured embedder for query embedding (may be fallback embedder).
                if self.embedder is None:
                    raise RuntimeError("Embedder not initialized")
                embeddings_to_use = self.embedder.encode(query).tolist()
            
            results = self.collection.query(
                query_embeddings=[embeddings_to_use],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
                
            # Add search metadata
            results['search_metadata'] = {
                'query': query,
                'cutoff_date': cutoff_date,
                'ticker': ticker,
                'filing_type': filing_type,
                'n_results_requested': n_results,
                'n_results_returned': len(results['documents'][0]) if results['documents'] else 0,
                'search_time': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]],
                'error': str(e)
            }
            
    def search_by_ticker_and_period(self,
                                    query: str,
                                    ticker: str,
                                    fiscal_year: int,
                                    fiscal_period: str = None,
                                    n_results: int = 5) -> Dict:
        """
        Search for specific company and fiscal period.
        
        Args:
            query: Search query
            ticker: Company ticker
            fiscal_year: Fiscal year
            fiscal_period: Fiscal period (Q1, Q2, Q3, Q4, FY)
            n_results: Number of results
            
        Returns:
            Search results
        """
        where_conditions = [
            {"ticker": ticker},
            {"fiscal_year": str(fiscal_year)}
        ]
        
        if fiscal_period:
            where_conditions.append({"fiscal_period": fiscal_period})
            
        where_filter = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
        
    def get_document_by_id(self, chunk_id: str) -> Dict:
        """
        Get a specific document by its chunk ID.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            Document with metadata
        """
        result = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if result['documents']:
            return {
                'chunk_id': chunk_id,
                'document': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
            
        return None
        
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get sample to analyze metadata
        sample = self.collection.peek(min(100, count))
        
        tickers = set()
        filing_types = set()
        filing_dates = []
        
        for meta in sample.get('metadatas', []):
            if meta.get('ticker'):
                tickers.add(meta['ticker'])
            if meta.get('filing_type'):
                filing_types.add(meta['filing_type'])
            if meta.get('filing_date'):
                filing_dates.append(meta['filing_date'])
                
        stats = {
            'total_documents': count,
            'collection_name': self.collection_name,
            'sample_size': len(sample.get('metadatas', [])),
            'unique_tickers_in_sample': list(tickers),
            'filing_types_in_sample': list(filing_types),
            'date_range_in_sample': {
                'earliest': min(filing_dates) if filing_dates else None,
                'latest': max(filing_dates) if filing_dates else None
            }
        }
        
        return stats
        
    def delete_collection(self):
        """Delete the collection (use with caution!)."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")
        
    def validate_temporal_integrity(self, cutoff_date: str) -> Dict:
        """
        Validate no documents after cutoff date are retrievable.
        
        Used for testing look-ahead bias prevention.
        
        Args:
            cutoff_date: Date to test (YYYYMMDD format)
            
        Returns:
            Validation results
        """
        # Try to retrieve documents after cutoff
        results = self.collection.query(
            query_texts=["financial report"],
            n_results=100,
            where={"filing_date": {"$gte": cutoff_date}},
            include=["metadatas"]
        )
        
        violations = []
        for meta in results.get('metadatas', [[]])[0]:
            if meta.get('filing_date', '') >= cutoff_date:
                violations.append({
                    'filing_date': meta['filing_date'],
                    'ticker': meta.get('ticker'),
                    'filing_type': meta.get('filing_type')
                })
                
        return {
            'cutoff_date': cutoff_date,
            'violations_found': len(violations),
            'temporal_integrity_valid': len(violations) == 0,
            'violations': violations[:10]  # Show first 10
        }


# Usage
if __name__ == "__main__":
    # Initialize vector store
    vector_store = TemporalVectorStore()
    
    # Check if data exists
    chunks_path = Path("data/processed/chunks/temporal_chunks.jsonl")
    embeddings_path = Path("data/processed/embeddings/embeddings.npy")
    
    if chunks_path.exists() and embeddings_path.exists():
        # Add chunks if collection is empty
        if vector_store.collection.count() == 0:
            vector_store.add_chunks_with_temporal_metadata()
            
        # Print stats
        stats = vector_store.get_collection_stats()
        print("\nVector Store Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Test temporal search
        print("\nTesting temporal search...")
        results = vector_store.temporal_search(
            query="What are the major risk factors?",
            cutoff_date="20230701",
            ticker="AAPL",
            n_results=3
        )
        
        print(f"\nFound {results['search_metadata']['n_results_returned']} results")
        for i, doc in enumerate(results['documents'][0][:2]):
            print(f"\n--- Result {i+1} ---")
            print(f"Metadata: {results['metadatas'][0][i]}")
            print(f"Content: {doc[:200]}...")
    else:
        print("Chunks and/or embeddings not found.")
        print("Please run the preprocessing pipeline first:")
        print("  1. python src/preprocessing/temporal_chunker.py")
        print("  2. python src/preprocessing/embedder.py")
