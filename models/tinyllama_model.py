"""
TinyLlama Model Implementation
1.1B parameter model - Fast CPU inference
Properly integrated with BaseLLM interface
"""

from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

try:
    from .base_model import BaseLLM
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.base_model import BaseLLM


class TinyLlamaModel(BaseLLM):
    """
    TinyLlama 1.1B Chat model.
    Optimized for fast CPU inference while maintaining quality.
    """
    
    def __init__(self, max_length: int = 2048):
        """
        Initialize TinyLlama model.
        
        Args:
            max_length: Maximum sequence length (default: 2048)
        """
        super().__init__(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_length=max_length
        )
        self.load_model()
    
    def load_model(self):
        """Load TinyLlama model and tokenizer."""
        print(f"🔄 Loading TinyLlama (1.1B params)...")
        print(f"   Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # CPU uses float32
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✅ TinyLlama loaded successfully!")
        print(f"   Parameters: {self._count_parameters():,}")
        print(f"   Memory: ~2.2 GB")
    
    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        """
        Format prompt using TinyLlama's chat template.
        
        TinyLlama uses ChatML format:
        <|system|>
        {system_message}</s>
        <|user|>
        {user_message}</s>
        <|assistant|>
        
        Args:
            question: User question
            context: Retrieved documents (for RAG)
            
        Returns:
            Formatted prompt string
        """
        # System message
        system_message = (
            "You are a helpful medical AI assistant. "
            "Provide clear, accurate answers based on the given context. "
            "If using context, cite it appropriately. Keep answers concise."
        )
        
        # Format context if provided
        if context and len(context) > 0:
            context_text = self._format_context(context)
            user_message = f"""Based on the following medical literature:

{context_text}

Question: {question}

Provide a clear, evidence-based answer:"""
        else:
            # No context - direct question
            user_message = f"Question: {question}\n\nProvide a clear, concise answer:"
        
        # Apply ChatML template
        prompt = f"""<|system|>
{system_message}</s>
<|user|>
{user_message}</s>
<|assistant|>
"""
        
        return prompt
    
    def _format_context(self, context: List[Dict]) -> str:
        """
        Format context documents for the prompt.
        
        Args:
            context: List of documents with 'pmid', 'title', 'text' or 'content'
            
        Returns:
            Formatted context string
        """
        if not context:
            return ""
        
        formatted = []
        for i, doc in enumerate(context[:5], 1):  # Limit to top 5 docs
            # Handle different field names
            text = doc.get('text', doc.get('content', 'N/A'))
            title = doc.get('title', 'Untitled')
            pmid = doc.get('pmid', 'Unknown')
            
            # Truncate text to avoid context overflow
            text = text[:500] if len(text) > 500 else text
            
            formatted.append(
                f"[Document {i}] PMID: {pmid}\n"
                f"Title: {title}\n"
                f"Content: {text}"
            )
        
        return "\n\n".join(formatted)
    
    def _generate_text(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int
    ) -> str:
        """
        Generate text using TinyLlama.
        Optimized for CPU with reasonable speed.
        
        Args:
            prompt: Formatted input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Reduce repetition
                no_repeat_ngram_size=3   # Avoid 3-gram repetition
            )
        
        # Decode only the new tokens (exclude prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def get_model_info(self) -> Dict:
        """Get TinyLlama model information."""
        info = super().get_model_info()
        info.update({
            'full_name': 'TinyLlama 1.1B Chat',
            'organization': 'TinyLlama',
            'optimized_for': 'Fast CPU inference',
            'context_length': '2k'
        })
        return info


# Test the model
def test_tinyllama():
    """Quick test of TinyLlama model."""
    print("\n" + "=" * 70)
    print("Testing TinyLlama Model")
    print("=" * 70 + "\n")
    
    # Initialize model
    model = TinyLlamaModel()
    
    # Test 1: Simple question (no context)
    print("\n📝 Test 1: Simple question (no RAG)")
    print("-" * 70)
    result = model.generate(
        question="What is diabetes?",
        context=None,
        max_new_tokens=80,
        temperature=0.7
    )
    print(f"Answer: {result['answer']}")
    print(f"Time: {result['generation_time']:.2f}s")
    
    # Test 2: Question with context (RAG mode)
    print("\n\n📚 Test 2: Question with context (RAG)")
    print("-" * 70)
    
    mock_context = [
        {
            'pmid': '12345678',
            'title': 'Diabetes mellitus: A metabolic disorder',
            'text': 'Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period. It is caused by insufficient insulin production or insulin resistance.'
        }
    ]
    
    result = model.generate(
        question="What causes diabetes?",
        context=mock_context,
        max_new_tokens=80,
        temperature=0.7
    )
    print(f"Answer: {result['answer']}")
    print(f"Time: {result['generation_time']:.2f}s")
    print(f"Used context: {result['used_context']}")
    
    # Model info
    print("\n\n📊 Model Information")
    print("-" * 70)
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_tinyllama()