"""
TinyLlama 1.1B Chat Model
GPU + CPU compatible
"""

from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .base_model import BaseLLM
except ImportError:
    from base_model import BaseLLM


class TinyLlamaModel(BaseLLM):

    def __init__(self, max_length: int = 2048):
        super().__init__(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_length=max_length
        )
        self.load_model()

    def load_model(self):
        print(f"🔄 Loading TinyLlama on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"✅ TinyLlama loaded")
        print(f"   Device: {self.device}")

    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        system = "You are a helpful biomedical assistant."

        if context:
            context_text = self._format_context(context)
            user = f"{context_text}\n\nQuestion: {question}"
        else:
            user = question

        return f"""<|system|>
{system}
<|user|>
{user}
<|assistant|>
"""