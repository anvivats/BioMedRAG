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
        faiss_index_path: str = "data/faiss_index.bin",
        sqlite_path: str = "data/pubmed.db",
        rerank: bool = True,
        embedder: Optional[MedCPTEmbedder] = None
    ):
        self.rerank = rerank

        # Load embedder
        self.embedder = embedder or MedCPTEmbedder()

        # Load FAISS index
        self.faiss_index_path = Path(faiss_index_path)
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")

        self.index = faiss.read_index(str(self.faiss_index_path))
        print(f"✓ Loaded FAISS index: {faiss_index_path}")

        # Load SQLite DB
        self.db_path = Path(sqlite_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        print(f"✓ Connected to SQLite: {sqlite_path}")

    def retrieve(self, query: str, k: int = 5, retrieval_k: int = 50) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        """
        # Encode query
        query_vec = self.embedder.encode_query(query).astype("float32")

        # FAISS search
        scores, indices = self.index.search(query_vec, retrieval_k)

        if indices.size == 0:
            return []

        pmids = [int(idx) for idx in indices[0] if idx != -1]

        # Fetch documents
        docs = self._fetch_documents(pmids)

        if not docs:
            return []

        # Optional reranking
        if self.rerank:
            return self._rerank_documents(query, docs, k)

        return docs[:k]

    def _fetch_documents(self, pmids: List[int]) -> List[Dict]:
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
        contents = [doc["content"] for doc in docs]
        scores = self.embedder.rerank(query, contents)

        scored_docs = [
            {**doc, "score": float(score)}
            for doc, score in zip(docs, scores)
        ]

        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        return scored_docs[:k]

    def __del__(self):
        if hasattr(self, "db"):
            self.db.close()


# Quick test
if __name__ == "__main__":
    retriever = FAISSRetriever(
        faiss_index_path="data/faiss_index.bin",
        sqlite_path="data/pubmed.db",
        rerank=True
    )

    query = "What is the role of TP53 in cancer?"
    docs = retriever.retrieve(query, k=5)

    print(f"\nRetrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. PMID {doc['pmid']} — {doc['title']}")
        if "score" in doc:
            print(f"   Score: {doc['score']:.4f}")
        print(doc["content"][:150], "...\n")
