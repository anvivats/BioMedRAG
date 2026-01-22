"""
LLaMA 3.2 3B Instruct
GPU-native
"""

from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .base_model import BaseLLM
except ImportError:
    from base_model import BaseLLM


class LlamaModel(BaseLLM):

    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        max_length=8192
    ):
        super().__init__(model_name, max_length)
        self.load_model()

    def load_model(self):
        print(f"🔄 Loading {self.model_name} on {self.device}...")

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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        print(f"✅ LLaMA 3.2 loaded")

    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        system = (
            "You are a biomedical expert assistant. "
            "Use evidence-based reasoning."
        )

        if context:
            context_text = self._format_context(context)
            user = f"{context_text}\n\nQuestion: {question}"
        else:
            user = question

        return f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
