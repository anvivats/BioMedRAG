"""
BioMistral 7B Model Implementation
Domain-specific model pre-trained on biomedical literature
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

# Handle both package and direct imports
try:
    from .base_model import BaseLLM
except ImportError:
    from base_model import BaseLLM


class BioMistralModel(BaseLLM):
    """
    BioMistral 7B implementation.
    Mistral 7B fine-tuned on PubMed abstracts and medical literature.
    Expected to perform best due to domain-specific pre-training.
    """
    
    def __init__(
        self,
        model_name: str = "BioMistral/BioMistral-7B",
        max_length: int = 8192
    ):
        """
        Initialize BioMistral model.
        
        Args:
            model_name: HuggingFace model ID
            max_length: Maximum context length (8k for Mistral)
        """
        super().__init__(model_name, max_length)
        self.load_model()
    
    def load_model(self):
        """Load BioMistral model and tokenizer."""
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
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ {self.model_name} loaded successfully")
        print(f"  Parameters: {self._count_parameters() / 1e9:.2f}B")
        print(f"  Device: {self.device}")
    
    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        """
        Format prompt using Mistral's instruction template.
        
        Mistral uses this format:
        <s>[INST] {instruction} [/INST] {response}</s>
        
        Args:
            question: User question
            context: Retrieved documents
            
        Returns:
            Formatted prompt
        """
        if context:
            # RAG mode: include context
            context_text = self._format_context(context)
            
            instruction = f"""You are a biomedical expert. Answer the following question using the provided scientific literature.

Context from PubMed:
{context_text}

Question: {question}

Instructions:
- Base your answer on the context provided
- Cite relevant PMIDs when making claims
- Be precise and evidence-based
- If the context is insufficient, state this clearly"""
        else:
            # No-RAG mode: direct question
            instruction = f"""You are a biomedical expert. Answer the following question:

{question}

Provide a clear, accurate, and evidence-based answer."""
        
        # Format using Mistral's instruction template
        prompt = f"<s>[INST] {instruction} [/INST]"
        
        return prompt


# Example usage
if __name__ == "__main__":
    # Test model loading
    model = BioMistralModel()
    
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
        },
        {
            'pmid': 67890,
            'title': 'p53 pathway in cancer therapy',
            'content': 'The p53 protein acts as a guardian of the genome by inducing cell cycle arrest or apoptosis in response to DNA damage. Loss of p53 function is a hallmark of cancer.'
        }
    ]
    
    result_rag = model.generate(question, context=mock_context, max_new_tokens=200)
    print(f"Answer: {result_rag['answer']}")
    print(f"Time: {result_rag['generation_time']:.2f}s")
    
    # Model info
    print("\n=== Model Information ===")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")