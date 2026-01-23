"""
RAG Pipeline
FAISS + SQLite retrieval with LLM generation
"""

import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Optional

# Force project root import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.retriever import FAISSRetriever
from core.embedder import MedCPTEmbedder

# Try to import models (optional - for LLM integration)
try:
    from models import get_model
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("⚠️ models module not found - RAG will work without LLM generation")


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline
    Supports retrieval-only mode and full RAG with LLM
    """

    def __init__(
        self,
        model_name: str = "phi3",
        index_path: str = "data/faiss_index.bin",
        db_path: str = "data/pubmed.db",
        pmid_map_path: str = "data/pmid_map.pkl",
        top_k: int = 5,
        use_rag: bool = True,
        use_llm: bool = True,
        share_embedder: bool = True,
    ):
        """
        Initialize RAG pipeline.

        Args:
            model_name: LLM backend name (phi3, llama, etc.)
            index_path: Path to FAISS index
            db_path: Path to SQLite database
            pmid_map_path: Path to PMID mapping file
            top_k: Number of documents to retrieve
            use_rag: Whether to use retrieval (if False, pure LLM)
            use_llm: Whether to use LLM for generation
            share_embedder: Share embedder between components (saves GPU memory)
        """
        self.model_name = model_name
        self.top_k = top_k
        self.use_rag = use_rag
        self.use_llm = use_llm

        print("\n🚀 Initializing RAG Pipeline")
        print("=" * 60)

        # Load shared embedder first (if sharing)
        shared_embedder = None
        if self.use_rag and share_embedder:
            print("🧠 Loading shared embedder...")
            shared_embedder = MedCPTEmbedder()

        # Load LLM (if available and requested)
        if self.use_llm and HAS_LLM:
            print("\n1️⃣ Loading language model...")
            try:
                self.model = get_model(model_name)
            except Exception as e:
                print(f"⚠️ Failed to load LLM: {e}")
                print("   Continuing with retrieval-only mode")
                self.model = None
                self.use_llm = False
        else:
            self.model = None
            print("1️⃣ LLM disabled (retrieval-only mode)")

        # Load retriever
        if self.use_rag:
            print("\n2️⃣ Initializing retriever...")
            print(f"   FAISS index : {index_path}")
            print(f"   SQLite DB   : {db_path}")
            print(f"   PMID map    : {pmid_map_path}")
            print(f"   Top-K docs  : {top_k}")

            try:
                self.retriever = FAISSRetriever(
                    index_path=index_path,
                    db_path=db_path,
                    pmid_map_path=pmid_map_path,
                    top_k=top_k,
                    embedder=shared_embedder,  # Share embedder!
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize retriever: {e}")
                raise
        else:
            self.retriever = None
            print("\n2️⃣ RAG disabled (direct generation only)")

        print("\n✅ RAG Pipeline ready")
        print("=" * 60 + "\n")

    def retrieve(self, question: str, k: int = None) -> List[Dict]:
        """
        Retrieve relevant documents for a question.

        Args:
            question: Query string
            k: Number of documents (defaults to self.top_k)

        Returns:
            List of retrieved documents with pmid, title, content
        """
        if not self.use_rag or not self.retriever:
            return []

        return self.retriever.retrieve(question, k=k)

    def answer(
        self,
        question: str,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        return_context: bool = False,
    ) -> Dict:
        """
        Answer a question using RAG pipeline.

        Args:
            question: Question to answer
            temperature: Sampling temperature for LLM
            max_new_tokens: Maximum tokens to generate
            return_context: Whether to include retrieved documents in result

        Returns:
            Dictionary with answer, metadata, and timing info
        """
        result = {
            "question": question,
            "model": self.model_name,
            "use_rag": self.use_rag,
            "use_llm": self.use_llm,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # ---- Step 1: Retrieval ----
        docs = []
        if self.use_rag and self.retriever:
            t0 = time.time()
            docs = self.retriever.retrieve(question, k=self.top_k)
            retrieval_time = time.time() - t0

            result["retrieval_time"] = retrieval_time
            result["num_docs"] = len(docs)
            result["pmids"] = [d["pmid"] for d in docs]

            if return_context:
                result["documents"] = docs
        else:
            result["retrieval_time"] = 0.0
            result["num_docs"] = 0
            result["pmids"] = []

        # ---- Step 2: Generation ----
        if self.use_llm and self.model:
            t0 = time.time()
            try:
                generation = self.model.generate(
                    question=question,
                    context=docs if docs else None,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                result["answer"] = generation.get("answer", "")
            except Exception as e:
                result["answer"] = f"Error during generation: {e}"
                print(f"⚠️ Generation error: {e}")

            generation_time = time.time() - t0
            result["generation_time"] = generation_time
        else:
            # Retrieval-only mode: return summary of retrieved docs
            generation_time = 0.0
            result["generation_time"] = generation_time

            if docs:
                top_doc = docs[0]
                result["answer"] = (
                    f"[Retrieval-only mode]\n\n"
                    f"Retrieved {len(docs)} relevant documents.\n\n"
                    f"Top result (PMID {top_doc['pmid']}): {top_doc['title']}\n"
                    f"{top_doc['content'][:300]}..."
                )
            else:
                result["answer"] = "No relevant documents found."

        result["total_time"] = result["retrieval_time"] + result["generation_time"]

        return result

    def batch_answer(
        self,
        questions: List[str],
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Answer multiple questions in batch.

        Args:
            questions: List of questions
            temperature: Sampling temperature
            max_new_tokens: Max tokens per answer
            show_progress: Whether to print progress

        Returns:
            List of result dictionaries
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                questions = tqdm(questions, desc="Answering questions")
            except ImportError:
                pass

        for q in questions:
            results.append(
                self.answer(
                    q,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            )

        return results

    def save_results(self, results: List[Dict], path: str):
        """
        Save results to JSON file.

        Args:
            results: List of result dictionaries
            path: Output file path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {output_path}")


# ============================================================
# Test / Example Usage
# ============================================================
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline(
        model_name="phi3",
        index_path="data/faiss_index.bin",
        db_path="data/pubmed.db",
        pmid_map_path="data/pmid_map.pkl",
        top_k=5,
        use_rag=True,
        use_llm=True,
        share_embedder=True,
    )

    # Test question
    question = "What is the role of TP53 in cancer?"

    print("\n" + "=" * 60)
    print("TEST QUERY")
    print("=" * 60)
    print(f"\n📌 Question: {question}\n")

    # Get answer
    result = pipeline.answer(question, return_context=True)

    # Display results
    print("💬 Answer:")
    print("-" * 60)
    print(result["answer"])
    print()

    print(f"📊 Retrieval Stats:")
    print(f"   Documents: {result['num_docs']}")
    print(f"   PMIDs: {result['pmids']}")
    print()

    print("⏱️ Timing:")
    print(f"   Retrieval: {result['retrieval_time']:.3f}s")
    print(f"   Generation: {result['generation_time']:.3f}s")
    print(f"   Total: {result['total_time']:.3f}s")
    print()

    # Display retrieved documents (if requested)
    if "documents" in result and result["documents"]:
        print("📚 Retrieved Documents:")
        print("-" * 60)
        for i, doc in enumerate(result["documents"], 1):
            print(f"\n{i}. PMID {doc['pmid']} — {doc['title']}")
            print(f"   {doc['content'][:200]}...")

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)