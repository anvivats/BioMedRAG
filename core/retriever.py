"""
FAISS + SQLite Retriever
"""

import sys
from pathlib import Path
import sqlite3
import faiss
import numpy as np
from typing import List, Dict, Optional

# Force project root import (FIXED - no relative imports)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedder import MedCPTEmbedder


class FAISSRetriever:
    """
    Retrieves documents using FAISS vector search + SQLite storage.
    """

    def __init__(
        self,
        index_path: str = "data/faiss.index",
        db_path: str = "data/pubmed.db",
        top_k: int = 5,
        embedder: Optional[MedCPTEmbedder] = None,
    ):
        self.top_k = top_k

        # Load embedder
        self.embedder = embedder or MedCPTEmbedder()

        # Load FAISS index
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(str(self.index_path))
        print(f"✓ Loaded FAISS index ({self.index.ntotal} vectors)")

        # Connect SQLite
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite DB not found: {db_path}")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        print("✓ Connected to SQLite database")

    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        if k is None:
            k = self.top_k

        # Encode query
        query_vec = self.embedder.encode_query(query).astype("float32")
        faiss.normalize_L2(query_vec)

        # FAISS search
        scores, indices = self.index.search(query_vec, k)

        if indices.size == 0:
            return []

        rowids = [int(i) for i in indices[0] if i != -1]
        if not rowids:
            return []

        return self._fetch_documents(rowids)

    def _fetch_documents(self, rowids: List[int]) -> List[Dict]:
        placeholders = ",".join("?" for _ in rowids)

        query = f"""
            SELECT rowid, pmid, title, abstract
            FROM documents
            WHERE rowid IN ({placeholders})
        """

        rows = self.conn.execute(query, rowids).fetchall()

        return [
            {
                "rowid": row["rowid"],
                "pmid": row["pmid"],
                "title": row["title"],
                "content": row["abstract"],
            }
            for row in rows
        ]

    def close(self):
        self.conn.close()


# Test code
if __name__ == "__main__":
    retriever = FAISSRetriever(
        index_path="data/faiss.index",
        db_path="data/pubmed.db",
        top_k=5,
    )

    query = "What is the role of TP53 in cancer?"
    docs = retriever.retrieve(query)

    print(f"\n✅ Retrieved {len(docs)} documents\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. PMID {d['pmid']} — {d['title']}")
        print(f"   {d['content'][:150]}...\n")

    retriever.close()