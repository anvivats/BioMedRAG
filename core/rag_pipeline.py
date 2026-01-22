"""
RAG Pipeline
FAISS + SQLite retrieval with LLM generation
Research-grade, Colab-safe
"""

import time
import json
from typing import Dict, List

# ---- Robust imports (script / package safe) ----
try:
    from .retriever import FAISSRetriever
    from .embedder import MedCPTEmbedder
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.retriever import FAISSRetriever
    from core.embedder import MedCPTEmbedder

try:
    from models import get_model
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import get_model


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline
    """

    def __init__(
        self,
        model_name: str = "phi3",
        index_path: str = "data/faiss_index.bin",
        sqlite_path: str = "data/pubmed.db",
        top_k: int = 5,
        use_rag: bool = True,
    ):
        """
        Initialize pipeline.

        Args:
            model_name: LLM backend name
            index_path: FAISS index path
            sqlite_path: SQLite database path
            top_k: Number of retrieved documents
            use_rag: Disable to use pure LLM
        """
        self.model_name = model_name
        self.top_k = top_k
        self.use_rag = use_rag

        print("\n🚀 Initializing RAG Pipeline")
        print("=" * 60)

        # Load LLM
        print("1️⃣ Loading language model...")
        self.model = get_model(model_name)

        # Load retriever
        if self.use_rag:
            print("\n2️⃣ Initializing retriever...")
            print(f"   FAISS index : {index_path}")
            print(f"   SQLite DB  : {sqlite_path}")
            print(f"   Top-K docs : {top_k}")

            self.retriever = FAISSRetriever(
                index_path=index_path,
                pmid_map_path=sqlite_path,
                top_k=top_k,
            )
        else:
            self.retriever = None
            print("\n2️⃣ RAG disabled (direct generation only)")

        print("\n✅ RAG Pipeline ready")
        print("=" * 60 + "\n")

    def answer(
        self,
        question: str,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        return_context: bool = False,
    ) -> Dict:
        """
        Answer a question using RAG.
        """
        result = {
            "question": question,
            "model": self.model_name,
            "use_rag": self.use_rag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # ---- Retrieval ----
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
            docs = None
            result["retrieval_time"] = 0.0
            result["num_docs"] = 0

        # ---- Generation ----
        t0 = time.time()
        generation = self.model.generate(
            question=question,
            context=docs,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        generation_time = time.time() - t0

        result["answer"] = generation["answer"]
        result["generation_time"] = generation_time
        result["total_time"] = result["retrieval_time"] + generation_time

        return result

    def batch_answer(
        self,
        questions: List[str],
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> List[Dict]:
        """Answer multiple questions."""
        results = []
        for q in questions:
            results.append(
                self.answer(
                    q,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            )
        return results

    def save(self, results: List[Dict], path: str):
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {path}")


# ---- Example run ----
if __name__ == "__main__":
    pipeline = RAGPipeline(
        model_name="phi3",
        index_path="data/faiss_index.bin",
        sqlite_path="data/pubmed.db",
        top_k=5,
        use_rag=True,
    )

    question = "What is the role of TP53 in cancer?"

    result = pipeline.answer(question, return_context=True)

    print("\n📌 Question:")
    print(question)

    print("\n💬 Answer:")
    print(result["answer"][:400])

    print("\n📄 Retrieved PMIDs:")
    print(result["pmids"])
