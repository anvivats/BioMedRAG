"""
FAISS + SQLite Retriever (DIRECT FAISS)
Research-grade, Colab-safe implementation
"""

import faiss
import sqlite3
import numpy as np
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
        index_path: str = "data/faiss_index.bin",
        pmid_map_path: str = "data/pubmed.db",  # Actually using it as sqlite_path
        top_k: int = 5,
        rerank: bool = True,
        embedder: Optional[MedCPTEmbedder] = None
    ):
        """
        Initialize retriever.
        
        Args:
            index_path: Path to FAISS index file
            pmid_map_path: Path to SQLite database (backwards compatible name)
            top_k: Default number of documents to retrieve
            rerank: Whether to use cross-encoder reranking
            embedder: Optional pre-initialized embedder
        """
        self.top_k = top_k
        self.rerank = rerank

        # Load embedder
        self.embedder = embedder or MedCPTEmbedder()

        # Load FAISS index
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(str(self.index_path))
        print(f"✓ Loaded FAISS index: {index_path}")
        print(f"  Total vectors: {self.index.ntotal}")

        # Load SQLite DB (using pmid_map_path as db path for compatibility)
        self.db_path = Path(pmid_map_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {pmid_map_path}")

        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        print(f"✓ Connected to SQLite: {pmid_map_path}")

    def retrieve(self, query: str, k: int = None, retrieval_k: int = 50) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of final documents to return (uses self.top_k if None)
            retrieval_k: Number of candidates to fetch before reranking
            
        Returns:
            List of document dicts with pmid, title, content, and optional score
        """
        if k is None:
            k = self.top_k

        # Encode query
        query_vec = self.embedder.encode_query(query).astype("float32")

        # FAISS search
        scores, indices = self.index.search(query_vec, retrieval_k)

        if indices.size == 0:
            print("⚠️ No documents found in FAISS search")
            return []

        # Extract valid PMIDs (FAISS uses row indices that map to PMIDs)
        pmids = [int(idx) for idx in indices[0] if idx != -1]

        if not pmids:
            print("⚠️ No valid document indices returned")
            return []

        # Fetch documents from SQLite
        docs = self._fetch_documents(pmids)

        if not docs:
            print("⚠️ No documents found in database")
            return []

        # Optional reranking
        if self.rerank:
            return self._rerank_documents(query, docs, k)

        return docs[:k]

    def _fetch_documents(self, pmids: List[int]) -> List[Dict]:
        """Fetch documents from SQLite by PMID."""
        if not pmids:
            return []

        placeholders = ",".join("?" * len(pmids))
        query = f"""
            SELECT pmid, title, abstract
            FROM documents
            WHERE pmid IN ({placeholders})
        """

        cursor = self.db.execute(query, pmids)
        rows = cursor.fetchall()

        return [
            {
                "pmid": row["pmid"],
                "title": row["title"],
                "content": row["abstract"],
            }
            for row in rows
        ]

    def _rerank_documents(self, query: str, docs: List[Dict], k: int) -> List[Dict]:
        """Rerank documents using cross-encoder."""
        contents = [doc["content"] for doc in docs]
        scores = self.embedder.rerank(query, contents)

        scored_docs = [
            {**doc, "score": float(score)}
            for doc, score in zip(docs, scores)
        ]

        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        return scored_docs[:k]

    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, "db"):
            self.db.close()


# Quick test
if __name__ == "__main__":
    retriever = FAISSRetriever(
        index_path="data/faiss_index.bin",
        pmid_map_path="data/pubmed.db",
        top_k=5,
        rerank=True
    )

    query = "What is the role of TP53 in cancer?"
    docs = retriever.retrieve(query, k=5)

    print(f"\n✅ Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. PMID {doc['pmid']} — {doc['title']}")
        if "score" in doc:
            print(f"   Score: {doc['score']:.4f}")
        print(f"   {doc['content'][:150]}...\n")