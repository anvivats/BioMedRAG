"""
Build FAISS index from SQLite PubMed database with PMID mapping
"""

import sys
from pathlib import Path
import sqlite3
import numpy as np
import faiss
import pickle
from tqdm import tqdm

# Project root import fix
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedder import MedCPTEmbedder


DB_PATH = "data/pubmed.db"
FAISS_INDEX_PATH = "data/faiss_index.bin"
PMID_MAP_PATH = "data/pmid_map.pkl"


def load_documents(db_path: str):
    """Load documents from SQLite, ordered by PMID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pmid, abstract
        FROM documents
        WHERE abstract IS NOT NULL
        ORDER BY pmid
    """)

    rows = cursor.fetchall()
    conn.close()

    pmids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    return pmids, texts


def main():
    print("=" * 60)
    print("FAISS Index Builder with PMID Mapping")
    print("=" * 60)

    # Load documents
    print("\n📄 Loading documents from database...")
    pmids, texts = load_documents(DB_PATH)

    if not texts:
        print("❌ No documents found. Run load_pubmedqa.py first.")
        return

    print(f"✓ Loaded {len(texts)} documents")
    print(f"  First PMID: {pmids[0]}")
    print(f"  Last PMID: {pmids[-1]}")

    # Load embedder
    print("\n🧠 Loading MedCPT embedder...")
    embedder = MedCPTEmbedder()

    # Encode documents (BATCHED)
    print("\n🔢 Encoding documents...")
    embeddings = embedder.encode_documents(texts, batch_size=32)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Build FAISS index
    print("\n📐 Building FAISS index...")
    
    dim = embeddings.shape[1]
    
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save FAISS index
    print(f"\n💾 Saving FAISS index → {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save PMID mapping (CRITICAL!)
    print(f"💾 Saving PMID mapping → {PMID_MAP_PATH}")
    with open(PMID_MAP_PATH, 'wb') as f:
        pickle.dump(pmids, f)

    print("\n✅ FAISS INDEX BUILD COMPLETE")
    print(f"Vectors indexed: {index.ntotal}")
    print(f"Dimensions: {dim}")
    print(f"\nMapping: FAISS index 0 → PMID {pmids[0]}")
    print(f"         FAISS index {len(pmids)-1} → PMID {pmids[-1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()