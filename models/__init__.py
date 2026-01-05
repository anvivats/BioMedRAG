"""
Models package for BioMed-RAG
Provides unified interface for all LLM implementations
"""

from .base_model import BaseLLM
from .phi3_model import Phi3Model
from .llama_model import LlamaModel
from .biomistral_model import BioMistralModel
from .tinyllama_model import TinyLlamaModel

__all__ = [
    'BaseLLM',
    'Phi3Model',
    'LlamaModel',
    'BioMistralModel',
    'TinyLlamaModel',
]


def get_model(model_name: str, **kwargs) -> BaseLLM:
    """
    Factory function to load models by name.
    
    Args:
        model_name: One of 'phi3', 'llama', 'biomistral', 'tinyllama'
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Initialized model instance
        
    Example:
        >>> model = get_model('tinyllama')
        >>> model = get_model('phi3')
        >>> model = get_model('biomistral', max_length=4096)
    """
    model_map = {
        'phi3': Phi3Model,
        'llama': LlamaModel,
        'biomistral': BioMistralModel,
        'tinyllama': TinyLlamaModel,
    }
    
    model_name_lower = model_name.lower()
    
    if model_name_lower not in model_map:
        available = ', '.join(model_map.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available}"
        )
    
    return model_map[model_name_lower](**kwargs)


# Model metadata for experiments
MODEL_INFO = {
    'tinyllama': {
        'name': 'TinyLlama 1.1B Chat',
        'parameters': '1.1B',
        'context_length': '2k',
        'organization': 'TinyLlama',
        'optimized_for': 'Fast CPU inference, resource-constrained environments'
    },
    'phi3': {
        'name': 'Phi-3 Mini',
        'parameters': '3.8B',
        'context_length': '4k',
        'organization': 'Microsoft',
        'optimized_for': 'Instruction following, reasoning'
    },
    'llama': {
        'name': 'Llama 3.2 3B',
        'parameters': '3B',
        'context_length': '8k',
        'organization': 'Meta',
        'optimized_for': 'General instruction following'
    },
    'biomistral': {
        'name': 'BioMistral 7B',
        'parameters': '7B',
        'context_length': '8k',
        'organization': 'BioMistral',
        'optimized_for': 'Biomedical literature, PubMed'
    }
}


def list_models():
    """Print available models and their specifications."""
    print("Available Models:")
    print("=" * 70)
    for key, info in MODEL_INFO.items():
        print(f"\n{key.upper()}:")
        for k, v in info.items():
            print(f"  {k}: {v}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    list_models()