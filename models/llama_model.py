"""
Llama 3.2 Model Implementation
"""

from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .base_model import BaseLLM
except ImportError:
    from base_model import BaseLLM


class LlamaModel(BaseLLM):
    """
    Llama 3.2 Instruct model
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_length: int = 4096
    ):
        super().__init__(model_name, max_length)
        self.load_model()

    def load_model(self):
        print(f"🔄 Loading {self.model_name} on {self.device}...")

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

        print(f"✅ Llama loaded successfully")
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _generate_text(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int
    ) -> str:
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
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()