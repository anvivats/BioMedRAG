"""
Llama 3.2 3B Model Implementation
Meta's lightweight instruction-tuned model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

# Handle both package and direct imports
try:
    from .base_model import BaseLLM
except ImportError:
    from base_model import BaseLLM


class LlamaModel(BaseLLM):
    """
    Llama 3.2 3B Instruct implementation.
    Meta's smallest instruction-tuned Llama model.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_length: int = 8192
    ):
        """
        Initialize Llama 3.2 model.
        
        Args:
            model_name: HuggingFace model ID
            max_length: Maximum context length (8k for Llama 3.2)
        """
        super().__init__(model_name, max_length)
        self.load_model()
    
    def load_model(self):
        """Load Llama 3.2 model and tokenizer."""
        print(f"Loading {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Llama models use EOS as pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ {self.model_name} loaded successfully")
        print(f"  Parameters: {self._count_parameters() / 1e9:.2f}B")
        print(f"  Device: {self.device}")
    
    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        """
        Format prompt using Llama 3.2's chat template.
        
        Llama 3.2 uses this format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        [system prompt]<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        [user message]<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
        Args:
            question: User question
            context: Retrieved documents
            
        Returns:
            Formatted prompt
        """
        system_prompt = "You are a biomedical expert assistant. Provide accurate, evidence-based answers to medical and scientific questions."
        
        if context:
            # RAG mode: include context
            context_text = self._format_context(context)
            
            user_message = f"""Answer the following question based on the provided scientific context.

Context from PubMed:
{context_text}

Question: {question}

Provide a clear, accurate answer based on the context above. Cite relevant PMIDs when possible. If the context doesn't contain enough information, acknowledge the limitations."""
        else:
            # No-RAG mode: direct question
            user_message = f"""Answer the following biomedical question:

{question}

Provide a clear, accurate answer based on your knowledge."""
        
        # Format using Llama's chat template
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt


# Example usage
if __name__ == "__main__":
    # Test model loading
    model = LlamaModel()
    
    question = "What is the role of TP53 in cancer?"
    
    print("\n=== Testing No-RAG Mode ===")
    result_no_rag = model.generate(question, context=None, max_new_tokens=150)
    print(f"Answer: {result_no_rag['answer']}")
    print(f"Time: {result_no_rag['generation_time']:.2f}s")
    
    # Test with RAG (mock context)
    print("\n=== Testing RAG Mode ===")
    mock_context = [
        {
            'pmid': 12345,
            'title': 'TP53 mutations in human cancers',
            'content': 'TP53 is a tumor suppressor gene that encodes p53 protein. Mutations in TP53 are found in over 50% of human cancers and lead to loss of cell cycle control.'
        }
    ]
    
    result_rag = model.generate(question, context=mock_context, max_new_tokens=150)
    print(f"Answer: {result_rag['answer']}")
    print(f"Time: {result_rag['generation_time']:.2f}s")
    
    # Model info
    print("\n=== Model Information ===")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")