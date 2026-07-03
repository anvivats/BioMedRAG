"""
Multi-model test script for RAG pipeline
Tests Phi-3, Llama, and BioMistral sequentially (memory-safe)
"""

from core.rag_pipeline import RAGPipeline
import json
import gc
import torch


def test_single_model(model_name: str, question: str):
    """Test a single model with RAG."""
    print(f"\n{'=' * 60}")
    print(f"Testing {model_name.upper()}")
    print('=' * 60)

    rag = None
    try:
        rag = RAGPipeline(
            model_name=model_name,
            top_k=5,
            use_rag=True,
            use_llm=True,
        )

        result = rag.answer(question, return_context=True)

        print(f"\n✓ Answer generated")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Retrieval: {result['retrieval_time']:.2f}s")
        print(f"  Generation: {result['generation_time']:.2f}s")
        print(f"  Docs retrieved: {result['num_docs']}")
        print(f"  PMIDs: {result['pmids']}")

        print(f"\n💬 Answer:")
        print(f"  {result['answer']}")

        return result

    finally:
        # Free GPU memory before loading the next model
        if rag is not None:
            del rag
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main test execution."""
    print("\n" + "=" * 60)
    print("BIOMED-RAG TESTING SUITE (Phi-3, Llama, BioMistral)")
    print("=" * 60)

    question = "What is the role of TP53 in cancer?"

    models_to_test = ["phi3", "llama", "biomistral"]
    results = {}

    for model_name in models_to_test:
        try:
            results[f"{model_name}_rag"] = test_single_model(model_name, question)
        except Exception as e:
            print(f"\n❌ {model_name} failed: {e}")
            results[f"{model_name}_rag"] = {"error": str(e)}

    # Save results
    print("\n\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    with open('test_results_all_models.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Results saved to test_results_all_models.json")

    # Comparison summary
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    for model_name in models_to_test:
        r = results.get(f"{model_name}_rag", {})
        if "error" in r:
            print(f"  {model_name.upper():12s} FAILED — {r['error']}")
        else:
            print(
                f"  {model_name.upper():12s} "
                f"gen={r['generation_time']:.2f}s  "
                f"total={r['total_time']:.2f}s  "
                f"docs={r['num_docs']}"
            )

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()