"""
FAISS + SQLite Retriever (with PMID mapping)
"""

import sys
from pathlib import Path
import sqlite3
import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional

# Force project root import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedder import MedCPTEmbedder


class FAISSRetriever:
    """
    Retrieves documents using FAISS vector search + SQLite storage.
    Uses PMID mapping to correctly link FAISS indices to documents.
    """

    def __init__(
        self,
        index_path: str = "data/faiss_index.bin",
        db_path: str = "data/pubmed.db",
        pmid_map_path: str = "data/pmid_map.pkl",
        top_k: int = 5,
        embedder: Optional[MedCPTEmbedder] = None,
    ):
        self.top_k = top_k

        # Load embedder (or use shared one)
        if embedder is None:
            print("Loading MedCPT embedder...")
            self.embedder = MedCPTEmbedder()
        else:
            print("Using shared embedder")
            self.embedder = embedder

        # Load FAISS index
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(str(self.index_path))
        print(f"✓ Loaded FAISS index ({self.index.ntotal} vectors)")

        # Load PMID mapping (CRITICAL!)
        self.pmid_map_path = Path(pmid_map_path)
        if not self.pmid_map_path.exists():
            raise FileNotFoundError(f"PMID mapping not found: {pmid_map_path}")
        
        with open(self.pmid_map_path, 'rb') as f:
            self.pmid_map = pickle.load(f)
        print(f"✓ Loaded PMID mapping ({len(self.pmid_map)} entries)")

        # Connect SQLite
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite DB not found: {db_path}")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        print("✓ Connected to SQLite database")

    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of documents with pmid, title, content
        """
        if k is None:
            k = self.top_k

        # Encode query
        query_vec = self.embedder.encode_query(query).astype("float32")
        faiss.normalize_L2(query_vec)

        # FAISS search
        scores, indices = self.index.search(query_vec, k)

        if indices.size == 0:
            return []

        # Map FAISS indices to PMIDs (FIXED!)
        pmids = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.pmid_map):
                pmids.append(self.pmid_map[idx])
        
        if not pmids:
            return []

        return self._fetch_documents(pmids)

    def _fetch_documents(self, pmids: List[int]) -> List[Dict]:
        """Fetch documents from SQLite by PMID."""
        placeholders = ",".join("?" for _ in pmids)

        query = f"""
            SELECT pmid, title, abstract
            FROM documents
            WHERE pmid IN ({placeholders})
        """

        rows = self.conn.execute(query, pmids).fetchall()

        return [
            {
                "pmid": row["pmid"],
                "title": row["title"],
                "content": row["abstract"],
            }
            for row in rows
        ]

    def close(self):
        """Close database connection."""
        self.conn.close()


# Test code
if __name__ == "__main__":
    retriever = FAISSRetriever(
        index_path="data/faiss_index.bin",
        db_path="data/pubmed.db",
        pmid_map_path="data/pmid_map.pkl",
        top_k=5,
    )

    query = "What is the role of TP53 in cancer?"
    docs = retriever.retrieve(query)

    print(f"\n✅ Retrieved {len(docs)} documents\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. PMID {d['pmid']} — {d['title']}")
        print(f"   {d['content'][:150]}...\n")

    retriever.close()