"""
Main RAG Pipeline
Orchestrates retrieval → generation workflow
"""

import time
from typing import Dict, List, Optional
import json

# Handle imports
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
    from models import get_model, BaseLLM
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import get_model, BaseLLM


class BiomedRAG:
    """
    Main RAG pipeline for biomedical question answering.
    Combines retrieval (FAISS + SQLite) with generation (LLM).
    """
    
    def __init__(
        self,
        model_name: str = "phi3",
        faiss_url: str = "http://localhost:5000/search",
        sqlite_path: str = "data/pubmed.db",
        n_docs: int = 5,
        rerank: bool = True,
        use_rag: bool = True
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            model_name: LLM to use ('phi3', 'llama', or 'biomistral')
            faiss_url: URL of FAISS Docker API server
            sqlite_path: Path to SQLite database
            n_docs: Number of documents to retrieve
            rerank: Whether to use cross-encoder reranking
            use_rag: Whether to use retrieval (False = direct LLM)
        """
        self.model_name = model_name
        self.n_docs = n_docs
        self.use_rag = use_rag
        
        # Initialize components
        print(f"🚀 Initializing BiomedRAG with {model_name.upper()}")
        print("=" * 60)
        
        # Load LLM
        print(f"\n1️⃣ Loading language model...")
        self.model = get_model(model_name)
        
        # Load retriever (only if using RAG)
        if use_rag:
            print(f"\n2️⃣ Initializing retriever...")
            print(f"   FAISS URL: {faiss_url}")
            print(f"   SQLite DB: {sqlite_path}")
            print(f"   Reranking: {'Enabled' if rerank else 'Disabled'}")
            
            self.retriever = FAISSRetriever(
                faiss_url=faiss_url,
                sqlite_path=sqlite_path,
                rerank=rerank
            )
        else:
            print(f"\n2️⃣ RAG disabled - using direct LLM generation")
            self.retriever = None
        
        print("\n" + "=" * 60)
        print("✅ BiomedRAG initialized successfully\n")
    
    def answer_question(
        self,
        question: str,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        return_context: bool = False
    ) -> Dict:
        """
        Answer a biomedical question using RAG.
        
        Args:
            question: User question
            temperature: Sampling temperature for generation
            max_new_tokens: Maximum tokens to generate
            return_context: Whether to include retrieved docs in response
            
        Returns:
            Dict with answer, timing, and optional context
        """
        result = {
            'question': question,
            'model': self.model_name,
            'use_rag': self.use_rag,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Step 1: Retrieve documents (if using RAG)
        if self.use_rag and self.retriever:
            start_time = time.time()
            retrieved_docs = self.retriever.retrieve(question, k=self.n_docs)
            retrieval_time = time.time() - start_time
            
            result['retrieval_time'] = retrieval_time
            result['num_docs_retrieved'] = len(retrieved_docs)
            result['retrieved_pmids'] = [doc['pmid'] for doc in retrieved_docs]
            
            if return_context:
                result['retrieved_docs'] = retrieved_docs
        else:
            retrieved_docs = None
            result['retrieval_time'] = 0.0
            result['num_docs_retrieved'] = 0
        
        # Step 2: Generate answer
        start_time = time.time()
        generation_result = self.model.generate(
            question,
            context=retrieved_docs,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        generation_time = time.time() - start_time
        
        # Combine results
        result['answer'] = generation_result['answer']
        result['generation_time'] = generation_time
        result['total_time'] = result.get('retrieval_time', 0) + generation_time
        
        return result
    
    def batch_answer(
        self,
        questions: List[str],
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions
            temperature: Sampling temperature
            max_new_tokens: Max tokens per answer
            show_progress: Whether to show progress bar
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            questions = tqdm(questions, desc="Answering questions")
        
        for question in questions:
            result = self.answer_question(
                question,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                return_context=False
            )
            results.append(result)
        
        return results
    
    def compare_rag_vs_no_rag(
        self,
        question: str,
        temperature: float = 0.7,
        max_new_tokens: int = 256
    ) -> Dict:
        """
        Compare RAG vs No-RAG for the same question.
        
        Args:
            question: Question to answer
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            
        Returns:
            Dict with both results for comparison
        """
        # Answer with RAG
        original_use_rag = self.use_rag
        
        self.use_rag = True
        result_with_rag = self.answer_question(
            question,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        self.use_rag = False
        result_without_rag = self.answer_question(
            question,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        # Restore original setting
        self.use_rag = original_use_rag
        
        return {
            'question': question,
            'model': self.model_name,
            'with_rag': result_with_rag,
            'without_rag': result_without_rag
        }
    
    def get_system_info(self) -> Dict:
        """Get information about the RAG system."""
        info = {
            'model': self.model_name,
            'model_info': self.model.get_model_info(),
            'use_rag': self.use_rag,
            'n_docs': self.n_docs,
            'retriever_active': self.retriever is not None
        }
        return info
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save results to JSON file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save JSON
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {output_path}")


def main():
    """Example usage of BiomedRAG."""
    print("\n" + "=" * 60)
    print("BiomedRAG Pipeline - Example Usage")
    print("=" * 60 + "\n")
    
    # Initialize pipeline with Phi-3
    rag = BiomedRAG(
        model_name="phi3",
        n_docs=5,
        rerank=True,
        use_rag=True
    )
    
    # Example questions
    questions = [
        "What is the role of TP53 in cancer?",
        "How does insulin resistance lead to diabetes?",
        "What are the main treatments for hypertension?"
    ]
    
    # Answer questions
    print("📝 Answering sample questions...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Question {i}: {question}")
        print('=' * 60)
        
        result = rag.answer_question(question, return_context=True)
        
        print(f"\n✓ Answer generated in {result['total_time']:.2f}s")
        print(f"  Retrieval: {result['retrieval_time']:.2f}s")
        print(f"  Generation: {result['generation_time']:.2f}s")
        print(f"  Documents used: {result['num_docs_retrieved']}")
        print(f"  PMIDs: {result['retrieved_pmids'][:3]}...")
        
        print(f"\n💬 Answer:")
        print(f"  {result['answer'][:200]}...")
    
    # Compare RAG vs No-RAG
    print(f"\n\n{'=' * 60}")
    print("RAG vs No-RAG Comparison")
    print('=' * 60)
    
    comparison = rag.compare_rag_vs_no_rag(questions[0])
    
    print(f"\nQuestion: {comparison['question']}")
    print(f"\n📚 WITH RAG:")
    print(f"  Time: {comparison['with_rag']['total_time']:.2f}s")
    print(f"  Answer: {comparison['with_rag']['answer'][:150]}...")
    
    print(f"\n🚫 WITHOUT RAG:")
    print(f"  Time: {comparison['without_rag']['total_time']:.2f}s")
    print(f"  Answer: {comparison['without_rag']['answer'][:150]}...")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()