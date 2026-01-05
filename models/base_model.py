"""
Base Model Interface
Abstract class for all LLM implementations (Phi-3, Llama, BioMistral)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import time
import torch


class BaseLLM(ABC):
    """
    Abstract base class for language models.
    All model wrappers (Phi-3, Llama, BioMistral) inherit from this.
    """
    
    def __init__(self, model_name: str, max_length: int = 512):
        """
        Initialize base model.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum generation length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        """
        Format the prompt for the specific model.
        Different models have different prompt templates.
        
        Args:
            question: User question
            context: List of retrieved documents
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def generate(
        self,
        question: str,
        context: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256
    ) -> Dict:
        """
        Generate answer to question (with optional context for RAG).
        
        Args:
            question: User question
            context: List of retrieved documents (for RAG mode)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with keys: answer, generation_time, used_context
        """
        if self.model is None:
            self.load_model()
        
        # Format prompt
        if context:
            prompt = self._format_prompt(question, context)
            used_context = True
        else:
            # No-RAG mode: just ask the question
            prompt = self._format_prompt(question, [])
            used_context = False
        
        # Generate answer with timing
        start_time = time.time()
        answer = self._generate_text(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
        generation_time = time.time() - start_time
        
        return {
            'answer': answer,
            'generation_time': generation_time,
            'used_context': used_context,
            'model': self.model_name
        }
    
    def _generate_text(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int
    ) -> str:
        """
        Internal method to generate text using the model.
        
        Args:
            prompt: Formatted input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling
            max_new_tokens: Max tokens to generate
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (exclude prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def get_model_info(self) -> Dict:
        """
        Get model metadata.
        
        Returns:
            Dict with model information
        """
        return {
            'name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'parameters': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count trainable parameters in the model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _format_context(self, context: List[Dict]) -> str:
        """
        Helper method to format context documents into text.
        
        Args:
            context: List of documents with pmid, title, content
            
        Returns:
            Formatted context string
        """
        if not context:
            return ""
        
        formatted = []
        for i, doc in enumerate(context, 1):
            formatted.append(
                f"Document {i} [PMID: {doc['pmid']}]:\n"
                f"Title: {doc['title']}\n"
                f"Content: {doc['content']}\n"
            )
        
        return "\n".join(formatted)


# Example usage pattern (will be implemented in subclasses)
if __name__ == "__main__":
    # This is abstract, see phi3_model.py for concrete implementation
    print("Base model class - use Phi3Model, LlamaModel, or BioMistralModel instead")