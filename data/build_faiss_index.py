"""
Build FAISS index from SQLite PubMed database
"""

import sys
from pathlib import Path
import sqlite3
import numpy as np
import faiss
from tqdm import tqdm

# Project root import fix
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedder import MedCPTEmbedder


DB_PATH = "data/pubmed.db"
FAISS_INDEX_PATH = "data/faiss.index"


def load_documents(db_path: str):
    """Load rowid + abstract text from SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT rowid, abstract
        FROM documents
        WHERE abstract IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    return ids, texts


def main():
    print("=" * 60)
    print("FAISS Index Builder")
    print("=" * 60)

    # Load documents
    print("📄 Loading documents from database...")
    ids, texts = load_documents(DB_PATH)

    if not texts:
        print("❌ No documents found. Run load_pubmedqa.py first.")
        return

    print(f"✓ Loaded {len(texts)} documents")

    # Load embedder
    print("🧠 Loading MedCPT embedder...")
    embedder = MedCPTEmbedder()

    # Encode documents (BATCHED)
    print("🔢 Encoding documents...")
    embeddings = embedder.encode_documents(texts, batch_size=32)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Build FAISS index
    print("📐 Building FAISS index...")
    
    # Get dimension from actual embeddings (FIXED - no embedding_dim attribute)
    dim = embeddings.shape[1]
    
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index
    print(f"💾 Saving FAISS index → {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("\n✅ FAISS INDEX BUILD COMPLETE")
    print(f"Vectors indexed: {index.ntotal}")
    print(f"Dimensions: {dim}")
    print("=" * 60)


if __name__ == "__main__":
    main()