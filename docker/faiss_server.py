"""
FAISS Server - Serves FAISS index via HTTP for retrieval
"""

from flask import Flask, request, jsonify
import faiss
import numpy as np
import pickle
import os
from pathlib import Path

app = Flask(__name__)

# Global variables
index = None
pmid_map = None


def load_index():
    """Load FAISS index and PMID mapping on startup."""
    global index, pmid_map
    
    index_path = os.getenv('INDEX_PATH', '../data/faiss_index.bin')
    pmid_path = os.getenv('PMID_MAP_PATH', '../data/pmid_map.pkl')
    
    print(f"Loading FAISS index from {index_path}...")
    
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    
    if not Path(pmid_path).exists():
        raise FileNotFoundError(f"PMID map not found: {pmid_path}")
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    print(f"✓ FAISS index loaded: {index.ntotal} vectors")
    
    # Load PMID mapping
    with open(pmid_path, 'rb') as f:
        pmid_map = pickle.load(f)
    print(f"✓ PMID mapping loaded: {len(pmid_map)} entries")
    
    print("✓ FAISS server ready")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'index_loaded': index is not None,
        'total_vectors': index.ntotal if index else 0
    })


@app.route('/search', methods=['POST'])
def search():
    """
    Search for similar vectors.
    
    Request JSON:
    {
        "queries": [[0.1, 0.2, ..., 0.768], ...],  # List of query vectors
        "k": 10  # Number of results per query
    }
    
    Response JSON:
    {
        "PMIDs": [[pmid1, pmid2, ...], ...],  # PMIDs for each query
        "distances": [[dist1, dist2, ...], ...],  # Similarity scores
        "indices": [[idx1, idx2, ...], ...]  # Index positions
    }
    """
    if index is None or pmid_map is None:
        return jsonify({'error': 'Index not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'queries' not in data:
            return jsonify({'error': 'Missing queries field'}), 400
        
        queries = np.array(data['queries'], dtype=np.float32)
        k = data.get('k', 10)
        
        # Ensure 2D array
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        
        # Normalize queries (for cosine similarity)
        faiss.normalize_L2(queries)
        
        # Search
        distances, indices = index.search(queries, k)
        
        # Convert indices to PMIDs
        pmids = []
        for query_indices in indices:
            query_pmids = [int(pmid_map[idx]) for idx in query_indices if idx < len(pmid_map)]
            pmids.append(query_pmids)
        
        return jsonify({
            'PMIDs': pmids,
            'distances': distances.tolist(),
            'indices': indices.tolist(),
            'num_queries': len(queries),
            'k': k
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get index statistics."""
    if index is None:
        return jsonify({'error': 'Index not loaded'}), 500
    
    return jsonify({
        'total_vectors': index.ntotal,
        'dimension': index.d,
        'is_trained': index.is_trained,
        'metric_type': 'METRIC_INNER_PRODUCT',  # Cosine similarity
        'num_pmids': len(pmid_map) if pmid_map else 0
    })


@app.route('/batch_search', methods=['POST'])
def batch_search():
    """
    Batch search endpoint for large query sets.
    Same as /search but optimized for batches.
    """
    return search()


if __name__ == '__main__':
    # Load index on startup
    load_index()
    
    # Start server
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"\n🚀 Starting FAISS server on {host}:{port}")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=False)