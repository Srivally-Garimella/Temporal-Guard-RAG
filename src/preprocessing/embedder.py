"""
Temporal Embedder for TemporalGuard-RAG

Generates embeddings for document chunks with temporal metadata indexing.
Supports multiple embedding models for financial text.
"""

from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalEmbedder:
    """
    Embedding generator with temporal metadata preservation.
    
    Supports multiple embedding models optimized for financial text:
    - all-mpnet-base-v2: Best quality (768 dimensions)
    - all-MiniLM-L6-v2: Faster option (384 dimensions)
    - BAAI/bge-large-en: High quality for retrieval
    """
    
    # Model configurations
    MODELS = {
        'mpnet': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'dimensions': 768,
            'description': 'Best quality general-purpose embeddings'
        },
        'minilm': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimensions': 384,
            'description': 'Fast and efficient'
        },
        'bge-large': {
            'name': 'BAAI/bge-large-en',
            'dimensions': 1024,
            'description': 'High quality for retrieval tasks'
        },
        'finance': {
            'name': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
            'dimensions': 384,
            'description': 'Good for paraphrase and semantic similarity'
        }
    }
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = None):
        """
        Initialize Temporal Embedder.
        
        Args:
            model_name: Name of the embedding model or shortcut (e.g., 'mpnet')
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Resolve model shortcut if provided
        if model_name in self.MODELS:
            model_config = self.MODELS[model_name]
            model_name = model_config['name']
            logger.info(f"Using model: {model_name} ({model_config['description']})")
            
        logger.info(f"Loading embedding model: {model_name}...")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(text, convert_to_numpy=True)
        
    def embed_texts(self, texts: List[str], batch_size: int = 32,
                   show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
        
    def create_embeddings(self, 
                         chunks_path: str = "data/processed/chunks/temporal_chunks.jsonl",
                         output_dir: str = "data/processed/embeddings",
                         batch_size: int = 32) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate embeddings for all chunks with temporal metadata.
        
        Args:
            chunks_path: Path to temporal chunks JSONL file
            output_dir: Directory to save embeddings and metadata
            batch_size: Batch size for embedding generation
            
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        chunks_path = Path(chunks_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load chunks
        logger.info(f"Loading chunks from {chunks_path}...")
        chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
                
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings (batch_size={batch_size})...")
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        
        # Save embeddings as numpy array
        embeddings_file = output_path / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        logger.info(f"Saved embeddings to {embeddings_file}")
        
        # Create and save metadata index
        metadata = []
        for i, chunk in enumerate(chunks):
            meta = {
                'index': i,
                'chunk_id': chunk['chunk_id'],
                'ticker': chunk.get('ticker'),
                'filing_date': chunk.get('filing_date'),
                'filing_type': chunk.get('filing_type'),
                'fiscal_year': chunk.get('fiscal_year'),
                'fiscal_period': chunk.get('fiscal_period'),
                'source_path': chunk.get('source_path'),
                'chunk_hash': chunk.get('chunk_hash')
            }
            metadata.append(meta)
            
        metadata_file = output_path / "embedding_metadata.jsonl"
        with open(metadata_file, 'w') as f:
            for meta in metadata:
                f.write(json.dumps(meta) + '\n')
                
        logger.info(f"Saved metadata to {metadata_file}")
        
        # Save embedding configuration
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'total_embeddings': len(embeddings),
            'batch_size': batch_size,
            'created_at': datetime.now().isoformat(),
            'chunks_source': str(chunks_path),
            'embeddings_file': str(embeddings_file),
            'metadata_file': str(metadata_file),
            'memory_size_mb': embeddings.nbytes / (1024 ** 2)
        }
        
        config_file = output_path / "embedding_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"✅ Embedding generation complete!")
        logger.info(f"   Total embeddings: {len(embeddings)}")
        logger.info(f"   Dimension: {self.embedding_dim}")
        logger.info(f"   Size: {embeddings.nbytes / (1024**2):.2f} MB")
        
        return embeddings, metadata
        
    def build_temporal_index(self,
                            metadata_path: str = "data/processed/embeddings/embedding_metadata.jsonl",
                            output_path: str = "data/processed/temporal_index"):
        """
        Build temporal index for efficient point-in-time queries.
        
        Creates indices for:
        - Filing date ranges
        - Ticker lookups
        - Filing type filters
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        metadata = []
        with open(metadata_path, 'r') as f:
            for line in f:
                metadata.append(json.loads(line))
                
        # Build indices
        by_ticker = {}
        by_filing_date = {}
        by_filing_type = {}
        
        for meta in metadata:
            idx = meta['index']
            ticker = meta.get('ticker', 'unknown')
            filing_date = meta.get('filing_date', 'unknown')
            filing_type = meta.get('filing_type', 'unknown')
            
            # Index by ticker
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append(idx)
            
            # Index by filing date
            if filing_date not in by_filing_date:
                by_filing_date[filing_date] = []
            by_filing_date[filing_date].append(idx)
            
            # Index by filing type
            if filing_type not in by_filing_type:
                by_filing_type[filing_type] = []
            by_filing_type[filing_type].append(idx)
            
        # Save indices
        with open(output_path / "by_ticker.json", 'w') as f:
            json.dump(by_ticker, f, indent=2)
            
        with open(output_path / "by_filing_date.json", 'w') as f:
            json.dump(by_filing_date, f, indent=2)
            
        with open(output_path / "by_filing_type.json", 'w') as f:
            json.dump(by_filing_type, f, indent=2)
            
        # Create sorted date list for efficient range queries
        sorted_dates = sorted([d for d in by_filing_date.keys() if d != 'unknown'])
        with open(output_path / "sorted_dates.json", 'w') as f:
            json.dump(sorted_dates, f, indent=2)
            
        # Save index statistics
        stats = {
            'total_embeddings': len(metadata),
            'unique_tickers': len(by_ticker),
            'unique_dates': len(by_filing_date),
            'unique_filing_types': len(by_filing_type),
            'date_range': {
                'earliest': sorted_dates[0] if sorted_dates else None,
                'latest': sorted_dates[-1] if sorted_dates else None
            },
            'tickers': list(by_ticker.keys()),
            'filing_types': list(by_filing_type.keys()),
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_path / "index_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"✅ Temporal index built!")
        logger.info(f"   Unique tickers: {stats['unique_tickers']}")
        logger.info(f"   Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        
        return stats
        
    def load_embeddings(self, 
                       embeddings_path: str = "data/processed/embeddings/embeddings.npy",
                       metadata_path: str = "data/processed/embeddings/embedding_metadata.jsonl"
                       ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Load pre-computed embeddings and metadata.
        
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        embeddings = np.load(embeddings_path)
        
        metadata = []
        with open(metadata_path, 'r') as f:
            for line in f:
                metadata.append(json.loads(line))
                
        logger.info(f"Loaded {len(embeddings)} embeddings")
        
        return embeddings, metadata
        
    def get_embeddings_by_cutoff(self, 
                                 embeddings: np.ndarray,
                                 metadata: List[Dict],
                                 cutoff_date: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get embeddings for documents filed before cutoff date.
        
        CRITICAL for point-in-time analysis!
        
        Args:
            embeddings: Full embeddings array
            metadata: Full metadata list
            cutoff_date: Date string (YYYYMMDD format)
            
        Returns:
            Filtered embeddings and metadata
        """
        valid_indices = []
        valid_metadata = []
        
        for meta in metadata:
            filing_date = meta.get('filing_date')
            
            if filing_date and filing_date <= cutoff_date:
                valid_indices.append(meta['index'])
                valid_metadata.append(meta)
                
        valid_embeddings = embeddings[valid_indices]
        
        logger.info(f"Filtered to {len(valid_indices)} embeddings (cutoff: {cutoff_date})")
        
        return valid_embeddings, valid_metadata


# Usage
if __name__ == "__main__":
    # Initialize embedder
    embedder = TemporalEmbedder(model_name='minilm')
    
    # Check if chunks exist
    chunks_path = Path("data/processed/chunks/temporal_chunks.jsonl")
    
    if chunks_path.exists():
        # Generate embeddings
        embeddings, metadata = embedder.create_embeddings()
        
        # Build temporal index
        embedder.build_temporal_index()
    else:
        print("Chunks file not found. Please run temporal_chunker.py first.")
        print(f"Expected path: {chunks_path}")
