"""
BioMistral 7B
GPU-ready (supports quantization)
"""

from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .base_model import BaseLLM
except ImportError:
    from base_model import BaseLLM


class BioMistralModel(BaseLLM):

    def __init__(
        self,
        model_name="BioMistral/BioMistral-7B",
        max_length=8192
    ):
        super().__init__(model_name, max_length)
        self.load_model()

    def load_model(self):
        print(f"🔄 Loading BioMistral on {self.device}...")

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

        print(f"✅ BioMistral loaded")

    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        if context:
            context_text = self._format_context(context)
            instruction = f"""
Use the following biomedical literature to answer.

{context_text}

Question: {question}
"""
        else:
            instruction = question

        return f"<s>[INST] {instruction} [/INST]"
