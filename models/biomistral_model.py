"""
BioMistral 7B
Colab-safe, GPU + CPU + Disk offloading
NO pipeline, deterministic inference
"""

from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .base_model import BaseLLM
except ImportError:
    from base_model import BaseLLM


class BioMistralModel(BaseLLM):

    def __init__(
        self,
        model_name: str = "BioMistral/BioMistral-7B",
        max_length: int = 2048   # 🔑 cap context for stability
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            offload_folder="./offload",   # 🔑 REQUIRED
            offload_buffers=True          # 🔑 STRONGLY RECOMMENDED
        )

        self.model.eval()

        print("✅ BioMistral loaded successfully")

    def generate(
        self,
        question: str,
        context: List[Dict] = None,
        max_new_tokens: int = 128
    ) -> str:
        prompt = self._format_prompt(question, context or [])

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # deterministic
                use_cache=False,        # 🔑 prevents slow hangs
                eos_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

    def _format_prompt(self, question: str, context: List[Dict]) -> str:
        if context:
            context_text = self._format_context(context)
            instruction = f"""
Use the following biomedical literature to answer the question.

{context_text}

Question: {question}
"""
        else:
            instruction = question

        return f"<s>[INST] {instruction} [/INST]"
