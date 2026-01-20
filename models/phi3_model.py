"""
Phi-3 Mini Model Implementation
GPU + CPU safe (NO pipeline)
"""

from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .base_model import BaseLLM
except ImportError:
    from base_model import BaseLLM


class Phi3Model(BaseLLM):
    """
    Phi-3 Mini Instruct model
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        max_length: int = 4096
    ):
        super().__init__(model_name, max_length)
        self.load_model()

    def load_model(self):
        print(f"🔄 Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        self.model.eval()

        print(f"✅ Phi-3 loaded successfully")
        print(f"   Parameters: {self._count_parameters() / 1e9:.2f}B")
        print(f"   Device: {self.device}")

    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        system_prompt = (
            "You are a biomedical expert assistant. "
            "Provide accurate, evidence-based answers."
        )

        if context:
            context_text = self._format_context(context)
            user_prompt = f"""
Context:
{context_text}

Question: {question}
"""
        else:
            user_prompt = question

        return f"""<|system|>
{system_prompt}
<|user|>
{user_prompt}
<|assistant|>
"""
