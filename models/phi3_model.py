"""
Phi-3 Mini Model Implementation
Microsoft's 3.8B parameter lightweight model
OPTIMIZED FOR CPU - Uses pipeline for stable generation
STANDALONE VERSION - No base class dependency
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict

class Phi3Model:
    """Microsoft Phi-3 Mini 4K Instruct model."""
    
    def __init__(self):
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipe = None  # Use pipeline for CPU stability
        self.load_model()
    
    def load_model(self):
        """Load the Phi-3 model and tokenizer."""
        print(f"Loading {self.model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # CPU needs float32
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create pipeline for stable generation on CPU
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device if self.device == "cuda" else -1  # -1 for CPU in pipeline
        )
        
        # Get model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ {self.model_name} loaded successfully")
        print(f"  Parameters: {total_params/1e9:.2f}B")
        print(f"  Device: {self.device}")
    
    def format_prompt(self, question: str, context: List[str] = None) -> str:
        """Format prompt using Phi-3 chat template."""
        if context:
            context_text = "\n\n".join([
                f"Document {i+1}:\n{doc}" 
                for i, doc in enumerate(context)
            ])
            user_message = f"""Using the following scientific documents, provide a detailed answer to the question.

Documents:
{context_text}

Question: {question}

Answer the question based on the provided documents. Be specific and cite relevant information."""
        else:
            user_message = question
        
        # Phi-3 chat format
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        # Use tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate(
        self,
        question: str,
        context: List[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        **kwargs
    ) -> Dict:
        """Generate answer using pipeline (CPU-stable)."""
        
        # Format prompt
        prompt = self.format_prompt(question, context)
        
        print(f"🔄 Generating answer (this may take 1-3 minutes on CPU)...")
        
        # Generate using pipeline (much more stable on CPU)
        with torch.no_grad():
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                repetition_penalty=1.1,
                return_full_text=False,  # Only return generated text
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract answer
        answer = outputs[0]['generated_text'].strip()
        
        print(f"✓ Generation complete!")
        
        return {
            'answer': answer,
            'model': 'phi3',
            'prompt_length': len(self.tokenizer.encode(prompt)),
            'answer_length': len(self.tokenizer.encode(answer))
        }
    
    def batch_generate(
        self,
        questions: List[str],
        contexts: List[List[str]] = None,
        **kwargs
    ) -> List[Dict]:
        """Generate answers for multiple questions."""
        if contexts is None:
            contexts = [None] * len(questions)
        
        results = []
        for question, context in zip(questions, contexts):
            result = self.generate(question, context, **kwargs)
            results.append(result)
        
        return results