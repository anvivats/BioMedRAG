"""
Simple test script for RAG pipeline
Tests all three models with and without RAG
"""

from core.rag_pipeline import BiomedRAG
from models import MODEL_INFO
import json


def test_single_model(model_name: str, question: str):
    """Test a single model with RAG."""
    print(f"\n{'=' * 60}")
    print(f"Testing {model_name.upper()}")
    print('=' * 60)
    
    # Initialize RAG
    rag = BiomedRAG(
        model_name=model_name,
        n_docs=5,
        rerank=True,
        use_rag=True
    )
    
    # Answer question
    result = rag.answer_question(question, return_context=True)
    
    # Print results
    print(f"\n✓ Answer generated")
    print(f"  Total time: {result['total_time']:.2f}s")
    print(f"  Retrieval: {result['retrieval_time']:.2f}s")
    print(f"  Generation: {result['generation_time']:.2f}s")
    print(f"  Docs retrieved: {result['num_docs_retrieved']}")
    print(f"  PMIDs: {result['retrieved_pmids']}")
    
    print(f"\n💬 Answer:")
    print(f"  {result['answer']}")
    
    return result


def test_all_models(question: str):
    """Test all three models on the same question."""
    print("\n" + "🔬" * 30)
    print("TESTING ALL MODELS")
    print("🔬" * 30)
    print(f"\nQuestion: {question}")
    
    results = {}
    
    for model_name in ['phi3', 'llama', 'biomistral']:
        try:
            result = test_single_model(model_name, question)
            results[model_name] = result
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}")
            results[model_name] = None
    
    # Summary comparison
    print("\n\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Model':<15} {'Time (s)':<12} {'Docs':<8} {'Answer Length'}")
    print("-" * 60)
    
    for model_name, result in results.items():
        if result:
            answer_len = len(result['answer'].split())
            print(
                f"{model_name.upper():<15} "
                f"{result['total_time']:<12.2f} "
                f"{result['num_docs_retrieved']:<8} "
                f"{answer_len} words"
            )
    
    return results


def test_rag_ablation(model_name: str = "phi3", question: str = None):
    """Test RAG vs No-RAG ablation."""
    if question is None:
        question = "What is the role of TP53 in cancer?"
    
    print("\n" + "🧪" * 30)
    print("RAG ABLATION STUDY")
    print("🧪" * 30)
    print(f"\nModel: {model_name.upper()}")
    print(f"Question: {question}")
    
    # Initialize RAG
    rag = BiomedRAG(model_name=model_name)
    
    # Compare
    comparison = rag.compare_rag_vs_no_rag(question)
    
    print(f"\n{'=' * 60}")
    print("WITH RAG")
    print('=' * 60)
    print(f"Time: {comparison['with_rag']['total_time']:.2f}s")
    print(f"Docs: {comparison['with_rag']['num_docs_retrieved']}")
    print(f"Answer: {comparison['with_rag']['answer']}")
    
    print(f"\n{'=' * 60}")
    print("WITHOUT RAG")
    print('=' * 60)
    print(f"Time: {comparison['without_rag']['total_time']:.2f}s")
    print(f"Answer: {comparison['without_rag']['answer']}")
    
    return comparison


def main():
    """Main test execution."""
    print("\n" + "=" * 60)
    print("BIOMED-RAG TESTING SUITE")
    print("=" * 60)
    
    # Test questions
    questions = [
        "What is the role of TP53 in cancer?",
        "How does insulin resistance lead to diabetes?",
    ]
    
    # Test 1: Single model test
    print("\n\n### TEST 1: Single Model Test ###")
    test_single_model("phi3", questions[0])
    
    # Test 2: All models comparison
    print("\n\n### TEST 2: All Models Comparison ###")
    all_results = test_all_models(questions[0])
    
    # Test 3: RAG ablation
    print("\n\n### TEST 3: RAG Ablation Study ###")
    ablation_results = test_rag_ablation("phi3", questions[0])
    
    # Save results
    print("\n\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    output = {
        'all_models': all_results,
        'ablation': ablation_results
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✓ Results saved to test_results.json")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
    