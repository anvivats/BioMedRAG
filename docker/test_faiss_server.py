"""
Test script for FAISS Docker server
Verifies that the server is working correctly
"""

import requests
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.embedder import MedCPTEmbedder


def test_health_check(base_url: str = "http://localhost:5000"):
    """Test health check endpoint."""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"✓ Server is healthy")
        print(f"  Status: {data['status']}")
        print(f"  Index loaded: {data['index_loaded']}")
        print(f"  Total vectors: {data['total_vectors']}")
        
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_stats_endpoint(base_url: str = "http://localhost:5000"):
    """Test statistics endpoint."""
    print("\n" + "=" * 60)
    print("TEST 2: Statistics Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"✓ Stats retrieved successfully")
        print(f"  Total vectors: {data['total_vectors']}")
        print(f"  Dimension: {data['dimension']}")
        print(f"  Trained: {data['is_trained']}")
        print(f"  Metric: {data['metric_type']}")
        print(f"  PMIDs: {data['num_pmids']}")
        
        return True
    except Exception as e:
        print(f"✗ Stats endpoint failed: {e}")
        return False


def test_search_endpoint(base_url: str = "http://localhost:5000"):
    """Test search endpoint with real query."""
    print("\n" + "=" * 60)
    print("TEST 3: Search Endpoint")
    print("=" * 60)
    
    try:
        # Initialize embedder
        print("Loading MedCPT embedder...")
        embedder = MedCPTEmbedder()
        
        # Create query
        query = "What is the role of TP53 in cancer?"
        print(f"\nQuery: {query}")
        
        # Encode query
        print("Encoding query...")
        query_vec = embedder.encode_query(query)
        
        # Search
        print("Searching FAISS index...")
        data = {
            'queries': [query_vec.tolist()[0]],
            'k': 5
        }
        
        response = requests.post(
            f"{base_url}/search",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(data),
            timeout=10
        )
        response.raise_for_status()
        
        result = response.json()
        
        print(f"\n✓ Search completed successfully")
        print(f"  Number of queries: {result['num_queries']}")
        print(f"  Results per query: {result['k']}")
        
        print(f"\n  Top 5 PMIDs:")
        for i, (pmid, dist) in enumerate(zip(result['PMIDs'][0], result['distances'][0]), 1):
            print(f"    {i}. PMID: {pmid}, Score: {dist:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Search endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_search(base_url: str = "http://localhost:5000"):
    """Test batch search with multiple queries."""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Search")
    print("=" * 60)
    
    try:
        embedder = MedCPTEmbedder()
        
        queries = [
            "What is the role of TP53 in cancer?",
            "How does insulin resistance lead to diabetes?",
            "What are the treatments for hypertension?"
        ]
        
        print(f"Testing with {len(queries)} queries...")
        
        # Encode all queries
        query_vecs = [embedder.encode_query(q).tolist()[0] for q in queries]
        
        # Batch search
        data = {
            'queries': query_vecs,
            'k': 3
        }
        
        response = requests.post(
            f"{base_url}/batch_search",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(data),
            timeout=15
        )
        response.raise_for_status()
        
        result = response.json()
        
        print(f"\n✓ Batch search completed")
        print(f"  Queries processed: {result['num_queries']}")
        
        for i, (query, pmids) in enumerate(zip(queries, result['PMIDs']), 1):
            print(f"\n  Query {i}: {query[:50]}...")
            print(f"    Top PMIDs: {pmids}")
        
        return True
    except Exception as e:
        print(f"✗ Batch search failed: {e}")
        return False


def main():
    """Run all tests."""
    base_url = "http://localhost:5000"
    
    print("\n" + "🧪" * 30)
    print("FAISS DOCKER SERVER TESTS")
    print("🧪" * 30)
    print(f"\nServer URL: {base_url}\n")
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Statistics", test_stats_endpoint),
        ("Search", test_search_endpoint),
        ("Batch Search", test_batch_search)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func(base_url)
            results[test_name] = success
        except KeyboardInterrupt:
            print("\n\n⚠ Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<10} {test_name}")
    
    print("=" * 60)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)