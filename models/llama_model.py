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

        # fp16 is only numerically stable on CUDA. Apple's MPS backend has a
        # known bug where fp16 attention/softmax can overflow into inf/NaN
        # during sampling, which crashes generation. CPU also doesn't
        # benefit from fp16.