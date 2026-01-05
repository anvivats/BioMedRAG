"""
Debug script to test the RAG pipeline directly
Run this to identify the exact error before testing the API
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import traceback
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rag_pipeline():
    """Test the RAG pipeline step by step."""
    
    print("\n" + "="*60)
    print("🔍 DEBUG: Testing BiomedRAG Pipeline")
    print("="*60 + "\n")
    
    try:
        # Step 1: Import
        print("Step 1: Importing BiomedRAG...")
        from core.rag_pipeline import BiomedRAG
        print("✓ Import successful\n")
        
        # Step 2: Initialize
        print("Step 2: Initializing BiomedRAG...")
        rag = BiomedRAG(
            model_name="phi3",
            n_docs=5,
            rerank=True,
            use_rag=True
        )
        print("✓ Initialization successful\n")
        
        # Step 3: Test question
        print("Step 3: Testing question answering...")
        test_question = "What is the role of TP53 in cancer?"
        print(f"Question: {test_question}\n")
        
        result = rag.answer_question(
            question=test_question,
            temperature=0.7,
            max_new_tokens=256,
            return_context=False
        )
        
        print("✓ Question answered successfully\n")
        
        # Step 4: Display results
        print("="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nModel: {result['model']}")
        print(f"Use RAG: {result['use_rag']}")
        print(f"Total Time: {result['total_time']:.2f}s")
        print(f"Retrieval Time: {result['retrieval_time']:.2f}s")
        print(f"Generation Time: {result['generation_time']:.2f}s")
        print(f"Documents Retrieved: {result['num_docs_retrieved']}")
        print(f"PMIDs: {result['retrieved_pmids'][:5]}")
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR DETECTED")
        print("="*60)
        print(f"\nError Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\nFull Traceback:")
        print("-"*60)
        traceback.print_exc()
        print("-"*60 + "\n")
        
        # Common issues and solutions
        print("COMMON ISSUES & SOLUTIONS:")
        print("-"*60)
        print("1. FAISS Server Not Running:")
        print("   → Make sure FAISS server is running on port 5000")
        print("   → Check with: curl http://localhost:5000/health\n")
        
        print("2. SQLite Database Missing:")
        print("   → Ensure data/pubmed.db exists")
        print("   → Check path in core/rag_pipeline.py\n")
        
        print("3. Model Loading Issues:")
        print("   → Models are loaded on CPU, this takes time")
        print("   → Check if models are downloaded in cache\n")
        
        print("4. Memory Issues:")
        print("   → Running on CPU requires significant RAM")
        print("   → Try closing other applications\n")
        
        print("5. Missing Dependencies:")
        print("   → Run: pip install -r requirements.txt")
        print("-"*60 + "\n")
        
        return False


def test_faiss_connection():
    """Test FAISS server connection."""
    print("\n" + "="*60)
    print("🔍 DEBUG: Testing FAISS Connection")
    print("="*60 + "\n")
    
    try:
        import requests
        
        print("Testing FAISS server at http://localhost:5000...")
        response = requests.get("http://localhost:5000/health", timeout=5)
        
        if response.status_code == 200:
            print("✓ FAISS server is running and healthy\n")
            return True
        else:
            print(f"⚠ FAISS server returned status code: {response.status_code}\n")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to FAISS server")
        print("→ Make sure FAISS server is running on port 5000")
        print("→ Start it with: python docker/faiss_server.py\n")
        return False
        
    except Exception as e:
        print(f"❌ Error testing FAISS: {e}\n")
        return False


def test_database_connection():
    """Test SQLite database connection."""
    print("\n" + "="*60)
    print("🔍 DEBUG: Testing Database Connection")
    print("="*60 + "\n")
    
    try:
        import sqlite3
        from pathlib import Path
        
        db_path = Path("data/pubmed.db")
        
        if not db_path.exists():
            print(f"❌ Database not found at: {db_path.absolute()}")
            print("→ Check the path in your configuration\n")
            return False
        
        print(f"Connecting to: {db_path.absolute()}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"✓ Found tables: {[t[0] for t in tables]}")
        
        # Test query - FIXED: use 'documents' table instead of 'articles'
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        
        print(f"✓ Database connected successfully")
        print(f"→ Total documents: {count:,}\n")
        
        conn.close()
        return True
        
    except sqlite3.OperationalError as e:
        print(f"❌ Database error: {e}")
        if "no such table" in str(e):
            print("\n→ Your database exists but might be empty")
            print("→ Try running: python data/load_pubmedqa.py")
            print("→ This will populate the database with PubMedQA data\n")
        return False
        
    except Exception as e:
        print(f"❌ Database error: {e}\n")
        return False


def main():
    """Run all debug tests."""
    print("\n" + "="*60)
    print("🚀 BioMed-RAG Debug Tool")
    print("="*60)
    
    # Test 1: FAISS
    faiss_ok = test_faiss_connection()
    
    # Test 2: Database
    db_ok = test_database_connection()
    
    # Test 3: Full Pipeline
    if faiss_ok and db_ok:
        pipeline_ok = test_rag_pipeline()
    else:
        print("\n⚠ Skipping pipeline test due to failed prerequisites")
        pipeline_ok = False
    
    # Summary
    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)
    print(f"FAISS Server:    {'✓ OK' if faiss_ok else '❌ FAILED'}")
    print(f"Database:        {'✓ OK' if db_ok else '❌ FAILED'}")
    print(f"RAG Pipeline:    {'✓ OK' if pipeline_ok else '❌ FAILED'}")
    print("="*60 + "\n")
    
    if faiss_ok and db_ok and pipeline_ok:
        print("✅ All tests passed! You can now run the API server.")
        print("→ Run: uvicorn api.app:app --host 0.0.0.0 --port 8000\n")
    else:
        print("❌ Some tests failed. Please fix the issues above.\n")


if __name__ == "__main__":
    main()