"""
FAISS + SQLite Retriever
Combines vector search (FAISS) with document storage (SQLite)
Simplified from original medical_RAG_system code
"""

import faiss
import sqlite3
import numpy as np
import requests
import json
from typing import List, Dict, Optional
from pathlib import Path
from .embedder import MedCPTEmbedder


class FAISSRetriever:
    """
    Retrieves relevant documents using FAISS vector search + SQLite storage.
    Supports optional cross-encoder reranking.
    """
    
    def __init__(
        self,
        faiss_url: str = "http://localhost:5000/search",
        sqlite_path: str = "data/pubmed.db",
        rerank: bool = True,
        embedder: Optional[MedCPTEmbedder] = None
    ):
        """
        Initialize retriever.
        
        Args:
            faiss_url: URL of FAISS Docker API server
            sqlite_path: Path to SQLite database
            rerank: Whether to use cross-encoder reranking
            embedder: Pre-initialized MedCPTEmbedder (creates new if None)
        """
        self.faiss_url = faiss_url
        self.rerank = rerank
        
        # Initialize embedder
        if embedder is None:
            self.embedder = MedCPTEmbedder()
        else:
            self.embedder = embedder
        
        # Connect to SQLite
        self.db_path = Path(sqlite_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
        
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        print(f"✓ Connected to SQLite: {sqlite_path}")
    
    def retrieve(self, query: str, k: int = 5, retrieval_k: int = 100) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Question text
            k: Number of final documents to return
            retrieval_k: Number of candidates to retrieve before reranking
            
        Returns:
            List of dicts with keys: pmid, title, content, score (if reranked)
        """
        # Step 1: Encode query
        query_vec = self.embedder.encode_query(query)
        
        # Step 2: FAISS search via Docker API
        pmids = self._faiss_search(query_vec, k=retrieval_k)
        
        # Step 3: Fetch documents from SQLite
        docs = self._fetch_documents(pmids)
        
        if len(docs) == 0:
            return []
        
        # Step 4: Rerank if enabled
        if self.rerank:
            return self._rerank_documents(query, docs, k)
        
        # Return top-k without reranking
        return docs[:k]
    
    def _faiss_search(self, query_vec: np.ndarray, k: int) -> List[int]:
        """
        Query FAISS Docker API for similar document PMIDs.
        
        Args:
            query_vec: Query embedding (1, 768)
            k: Number of results
            
        Returns:
            List of PMIDs
        """
        try:
            data = {
                'queries': [query_vec.tolist()[0]],
                'k': k
            }
            response = requests.post(
                self.faiss_url,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(data),
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # FAISS server returns: {'PMIDs': [[pmid1, pmid2, ...]], ...}
            pmids = result.get('PMIDs', [[]])[0]
            return [int(pmid) for pmid in pmids]
            
        except requests.RequestException as e:
            print(f"⚠ FAISS API error: {e}")
            return []
    
    def _fetch_documents(self, pmids: List[int]) -> List[Dict]:
        """
        Fetch document content from SQLite by PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of document dicts
        """
        if not pmids:
            return []
        
        placeholders = ','.join('?' * len(pmids))
        query = f"""
            SELECT pmid, title, abstract
            FROM documents
            WHERE pmid IN ({placeholders})
        """
        
        cursor = self.db.execute(query, pmids)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        docs = [
            {
                'pmid': row['pmid'],
                'title': row['title'],
                'content': row['abstract']
            }
            for row in rows
        ]
        
        return docs
    
    def _rerank_documents(self, query: str, docs: List[Dict], k: int) -> List[Dict]:
        """
        Rerank documents using cross-encoder and return top-k.
        
        Args:
            query: Question text
            docs: List of candidate documents
            k: Number of documents to return
            
        Returns:
            List of top-k documents with scores
        """
        # Get reranking scores
        contents = [doc['content'] for doc in docs]
        scores = self.embedder.rerank(query, contents)
        
        # Combine docs with scores
        scored_docs = [
            {**doc, 'score': float(score)}
            for doc, score in zip(docs, scores)
        ]
        
        # Sort by score (descending) and filter score > 0
        scored_docs = sorted(
            scored_docs,
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Keep only positive scores
        scored_docs = [doc for doc in scored_docs if doc['score'] > 0]
        
        return scored_docs[:k]
    
    def __del__(self):
        """Close SQLite connection on cleanup."""
        if hasattr(self, 'db'):
            self.db.close()


# Example usage
if __name__ == "__main__":
    retriever = FAISSRetriever(
        faiss_url="http://localhost:5000/search",
        sqlite_path="data/pubmed.db",
        rerank=True
    )
    
    query = "What is the role of TP53 in cancer?"
    docs = retriever.retrieve(query, k=5)
    
    print(f"Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. [PMID: {doc['pmid']}] {doc['title']}")
        if 'score' in doc:
            print(f"   Score: {doc['score']:.4f}")
        print(f"   {doc['content'][:150]}...\n")