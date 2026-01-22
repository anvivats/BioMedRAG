"""
BioMistral 7B Model
Colab-safe with CPU/GPU offloading support
"""

from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .base_model import BaseLLM
except ImportError:
    from base_model import BaseLLM


class BioMistralModel(BaseLLM):
    """
    BioMistral 7B - biomedical domain-specific model
    Supports automatic device mapping with disk offloading
    """

    def __init__(
        self,
        model_name: str = "BioMistral/BioMistral-7B",
        max_length: int = 2048
    ):
        super().__init__(model_name, max_length)
        self.load_model()

    def load_model(self):
        print(f"🔄 Loading BioMistral on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with disk offloading support
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            offload_folder="./offload",
            offload_buffers=True
        )

        self.model.eval()

        print("✅ BioMistral loaded successfully")
        print(f"   Device: {self.device}")
        print(f"   Offloading: Enabled (CPU + Disk)")

    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        """
        Format prompt using Mistral's instruction template.
        
        Args:
            question: User's question
            context: List of context documents
            
        Returns:
            Formatted prompt string
        """
        if context:
            context_text = self._format_context(context)
            instruction = f"""Use the following biomedical literature to answer the question.

Context:
{context_text}

Question: {question}

Provide a clear, evidence-based answer:"""
        else:
            instruction = question

        return f"<s>[INST] {instruction} [/INST]"