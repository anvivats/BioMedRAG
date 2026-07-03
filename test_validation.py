"""
Validation suite for BioMedRAG paper
- Runs a small question set across multiple models (RAG mode)
- Runs a RAG-vs-no-RAG ablation on one model
Produces logs suitable for citing in the paper.
"""

from core.rag_pipeline import RAGPipeline
import json
import gc
import torch


QUESTIONS = [
    "What is the role of TP53 in cancer?",
    "How does insulin resistance lead to type 2 diabetes?",
    "What is the mechanism of action of statins?",
    "How do checkpoint inhibitors work in cancer immunotherapy?",
    "What is the significance of BRCA1 mutations in breast cancer risk?",
    "How does chronic inflammation contribute to cardiovascular disease?",
]

MODELS_FOR_COMPARISON = ["phi3", "tinyllama"]
MODEL_FOR_ABLATION = "phi3"


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model_over_questions(model_name: str, questions, use_rag: bool = True):
    """Run one model over all questions, return list of results."""
    print(f"\n{'=' * 60}")
    print(f"Running {model_name.upper()}  |  RAG={'ON' if use_rag else 'OFF'}")
    print('=' * 60)

    results = []
    rag = None
    try:
        rag = RAGPipeline(
            model_name=model_name,
            top_k=5,
            use_rag=use_rag,
            use_llm=True,
        )

        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {q}")
            try:
                result = rag.answer(q, return_context=True)
                print(f"  ✓ gen={result['generation_time']:.2f}s  "
                      f"total={result['total_time']:.2f}s  "
                      f"docs={result['num_docs']}")
                results.append(result)
            except Exception as e:
                print(f"  ❌ failed: {e}")
                results.append({"question": q, "error": str(e)})
    finally:
        if rag is not None:
            del rag
        cleanup()

    return results


def summarize(label, results):
    """Print a quick summary table for a set of results."""
    ok = [r for r in results if "error" not in r]
    failed = len(results) - len(ok)
    if not ok:
        print(f"  {label:20s} ALL FAILED ({failed} errors)")
        return
    avg_gen = sum(r["generation_time"] for r in ok) / len(ok)
    avg_total = sum(r["total_time"] for r in ok) / len(ok)
    avg_retrieval = sum(r["retrieval_time"] for r in ok) / len(ok)
    print(
        f"  {label:20s} n={len(ok)}/{len(results)}  "
        f"avg_retrieval={avg_retrieval:.3f}s  "
        f"avg_gen={avg_gen:.2f}s  "
        f"avg_total={avg_total:.2f}s"
    )


def main():
    print("\n" + "=" * 60)
    print("BIOMEDRAG VALIDATION SUITE")
    print("=" * 60)
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Models: {MODELS_FOR_COMPARISON}")

    all_results = {}

    # ---- Part 1: multi-model comparison (RAG mode) ----
    for model_name in MODELS_FOR_COMPARISON:
        key = f"{model_name}_rag"
        all_results[key] = run_model_over_questions(
            model_name, QUESTIONS, use_rag=True
        )

    # ---- Part 2: RAG vs non-RAG ablation (single model) ----
    all_results[f"{MODEL_FOR_ABLATION}_no_rag"] = run_model_over_questions(
        MODEL_FOR_ABLATION, QUESTIONS, use_rag=False
    )

    # ---- Save ----
    with open("validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n✓ Results saved to validation_results.json")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    for model_name in MODELS_FOR_COMPARISON:
        summarize(f"{model_name} (RAG)", all_results[f"{model_name}_rag"])
    summarize(
        f"{MODEL_FOR_ABLATION} (no RAG)",
        all_results[f"{MODEL_FOR_ABLATION}_no_rag"],
    )

    print("\n" + "=" * 60)
    print("✅ VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()