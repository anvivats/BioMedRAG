"""
Simple test script for RAG pipeline
Currently testing Phi-3 only (memory-safe on 16GB Mac)
"""

from core.rag_pipeline import RAGPipeline
import json


def test_single_model(model_name: str, question: str):
    """Test a single model with RAG."""
    print(f"\n{'=' * 60}")
    print(f"Testing {model_name.upper()}")
    print('=' * 60)
    
    # Initialize RAG
    rag = RAGPipeline(
        model_name=model_name,
        top_k=5,
        use_rag=True,
        use_llm=True,
    )
    
    # Answer question
    result = rag.answer(question, return_context=True)
    
    # Print results
    print(f"\n✓ Answer generated")
    print(f"  Total time: {result['total_time']:.2f}s")
    print(f"  Retrieval: {result['retrieval_time']:.2f}s")
    print(f"  Generation: {result['generation_time']:.2f}s")
    print(f"  Docs retrieved: {result['num_docs']}")
    print(f"  PMIDs: {result['pmids']}")
    
    print(f"\n💬 Answer:")
    print(f"  {result['answer']}")
    
    return result


def main():
    """Main test execution."""
    print("\n" + "=" * 60)
    print("BIOMED-RAG TESTING SUITE (Phi-3 only)")
    print("=" * 60)
    
    # Test questions
    questions = [
        "What is the role of TP53 in cancer?",
        "How does insulin resistance lead to diabetes?",
    ]
    
    # Test: Single model test (Phi-3 only, RAG pipeline)
    print("\n\n### TEST: Single Model Test (Phi-3) ###")
    phi3_result = test_single_model("phi3", questions[0])
    
    # NOTE: llama and biomistral will be run as separate processes later
    # to stay within 16GB memory. Same for RAG-vs-no-RAG ablation.
    
    # Save results
    print("\n\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    output = {
        'phi3_rag': phi3_result,
    }
    
    with open('test_results_phi3.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✓ Results saved to test_results_phi3.json")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()