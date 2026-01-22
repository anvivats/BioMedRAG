"""
Build FAISS index from SQLite documents
Encodes all abstracts and creates searchable vector index
"""

import faiss
import numpy as np
import sqlite3
from pathlib import Path
from tqdm import tqdm
import pickle
import sys

# Handle imports
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from core.embedder import MedCPTEmbedder
    from data.create_database import DatabaseManager
except ImportError:
    print("⚠ Import error - make sure you're running from project root")
    from embedder import MedCPTEmbedder


class FAISSIndexBuilder:
    """Builds FAISS index from documents in SQLite database."""
    
    def __init__(
        self,
        db_path: str = "data/pubmed.db",
        index_path: str = "data/faiss_index.bin",
        pmid_map_path: str = "data/pmid_map.pkl"
    ):
        """
        Initialize index builder.
        
        Args:
            db_path: Path to SQLite database
            index_path: Path to save FAISS index
            pmid_map_path: Path to save PMID mapping (index_id -> pmid)
        """
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.pmid_map_path = Path(pmid_map_path)
        
        # Create output directory
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedder
        print("🔧 Initializing MedCPT embedder...")
        self.embedder = MedCPTEmbedder()
    
    def load_documents_from_db(self):
        """
        Load all documents from database.
        
        Returns:
            Tuple of (pmids, abstracts)
        """
        print(f"📚 Loading documents from {self.db_path}...")
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT pmid, abstract FROM documents ORDER BY pmid")
        rows = cursor.fetchall()
        conn.close()
        
        pmids = [row[0] for row in rows]
        abstracts = [row[1] for row in rows]
        
        print(f"✓ Loaded {len(pmids)} documents")
        return pmids, abstracts
    
    def encode_documents(self, abstracts: list, batch_size: int = 32):
        """
        Encode all abstracts to vectors.
        
        Args:
            abstracts: List of abstract texts
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (num_docs, embedding_dim)
        """
        print(f"🔢 Encoding {len(abstracts)} abstracts...")
        print(f"  Batch size: {batch_size}")
        print(f"  This may take several minutes...\n")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(abstracts), batch_size), desc="Encoding"):
            batch = abstracts[i:i + batch_size]
            embeddings = self.embedder.encode_documents(batch, batch_size=len(batch))
            all_embeddings.append(embeddings)
        
        # Combine all batches
        embeddings_matrix = np.vstack(all_embeddings)
        
        print(f"\n✓ Encoding complete")
        print(f"  Shape: {embeddings_matrix.shape}")
        print(f"  Dtype: {embeddings_matrix.dtype}")
        
        return embeddings_matrix
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of embeddings
            
        Returns:
            FAISS index
        """
        print("🔨 Building FAISS index...")
        
        # Get dimensions
        num_vectors, dim = embeddings.shape
        print(f"  Vectors: {num_vectors}")
        print(f"  Dimensions: {dim}")
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index (IndexFlatIP for cosine similarity)
        # For larger datasets, consider IndexIVFFlat for faster search
        if num_vectors > 100000:
            print("  Using IVF index for large dataset...")
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, min(100, num_vectors // 1000))
            index.train(embeddings)
        else:
            print("  Using flat index...")
            index = faiss.IndexFlatIP(dim)
        
        # Add vectors to index
        index.add(embeddings)
        
        print(f"✓ FAISS index built")
        print(f"  Total vectors: {index.ntotal}")
        
        return index
    
    def save_index(self, index, pmids: list):
        """
        Save FAISS index and PMID mapping.
        
        Args:
            index: FAISS index
            pmids: List of PMIDs (in same order as index)
        """
        print(f"💾 Saving index to {self.index_path}...")
        faiss.write_index(index, str(self.index_path))
        print(f"✓ FAISS index saved")
        
        print(f"💾 Saving PMID mapping to {self.pmid_map_path}...")
        with open(self.pmid_map_path, 'wb') as f:
            pickle.dump(pmids, f)
        print(f"✓ PMID mapping saved")
    
    def build_full_pipeline(self, batch_size: int = 32):
        """
        Complete pipeline: load, encode, build, save.
        
        Args:
            batch_size: Batch size for encoding
        """
        print("=" * 60)
        print("FAISS Index Builder")
        print("=" * 60)
        print()
        
        # Step 1: Load documents
        pmids, abstracts = self.load_documents_from_db()
        
        if len(abstracts) == 0:
            print("❌ No documents found in database!")
            print("   Run 'python data/load_pubmedqa.py' first")
            return
        
        # Step 2: Encode abstracts
        print()
        embeddings = self.encode_documents(abstracts, batch_size)
        
        # Step 3: Build FAISS index
        print()
        index = self.build_faiss_index(embeddings)
        
        # Step 4: Save index and mapping
        print()
        self.save_index(index, pmids)
        
        print("\n" + "=" * 60)
        print("✅ FAISS INDEX BUILT SUCCESSFULLY")
        print("=" * 60)
        print(f"\nIndex location: {self.index_path.absolute()}")
        print(f"PMID map location: {self.pmid_map_path.absolute()}")
        print(f"\nNext steps:")
        print("  1. Start FAISS server: docker-compose up -d")
        print("  2. Test retrieval: python experiments/test_retrieval.py")


def test_index(index_path: str = "data/faiss_index.bin", pmid_map_path: str = "data/pmid_map.pkl"):
    """
    Test the built index with a sample query.
    
    Args:
        index_path: Path to FAISS index
        pmid_map_path: Path to PMID mapping
    """
    print("\n" + "=" * 60)
    print("Testing FAISS Index")
    print("=" * 60)
    
    # Load index
    print("Loading index...")
    index = faiss.read_index(index_path)
    
    with open(pmid_map_path, 'rb') as f:
        pmids = pickle.load(f)
    
    print(f"✓ Index loaded: {index.ntotal} vectors")
    
    # Load embedder
    embedder = MedCPTEmbedder()
    
    # Test query
    query = "What is the role of TP53 in cancer?"
    print(f"\nTest query: {query}")
    
    # Encode query
    query_vec = embedder.encode_query(query)
    faiss.normalize_L2(query_vec)
    
    # Search
    k = 5
    distances, indices = index.search(query_vec, k)
    
    print(f"\nTop {k} results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        pmid = pmids[idx]
        print(f"  {i}. PMID: {pmid}, Score: {dist:.4f}")
    
    print("\n✓ Index is working correctly!")


def main():
    """Main execution."""
    # Build index
    builder = FAISSIndexBuilder()
    builder.build_full_pipeline(batch_size=32)
    
    # Test index
    test_index()


if __name__ == "__main__":
    main()