"""
MedCPT Embedding Wrapper
Simplified from original medical_RAG_system code
"""

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Union


class MedCPTEmbedder:
    """
    Wrapper for MedCPT Query Encoder and Cross-Encoder.
    Handles both initial embedding and reranking.
    """
    
    def __init__(
        self, 
        query_encoder: str = 'ncbi/MedCPT-Query-Encoder',
        cross_encoder: str = 'ncbi/MedCPT-Cross-Encoder',
        max_length: int = 512
    ):
        """
        Initialize MedCPT encoders.
        
        Args:
            query_encoder: HuggingFace model name for query encoding
            cross_encoder: HuggingFace model name for reranking
            max_length: Maximum sequence length
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        
        # Load query encoder (for FAISS search)
        print(f"Loading MedCPT Query Encoder on {self.device}...")
        self.query_model = AutoModel.from_pretrained(query_encoder).to(self.device)
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_encoder)
        
        # Load cross-encoder (for reranking)
        print(f"Loading MedCPT Cross-Encoder on {self.device}...")
        self.cross_model = AutoModelForSequenceClassification.from_pretrained(
            cross_encoder
        ).to(self.device)
        self.cross_tokenizer = AutoTokenizer.from_pretrained(cross_encoder)
        
        print("✓ MedCPT models loaded successfully")
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query to vector for FAISS search.
        
        Args:
            query: Question text
            
        Returns:
            numpy array of shape (1, 768)
        """
        with torch.no_grad():
            inputs = self.query_tokenizer(
                query,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.max_length
            ).to(self.device)
            
            outputs = self.query_model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embedding
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple documents for FAISS indexing.
        
        Args:
            documents: List of document texts
            batch_size: Number of documents to process at once
            
        Returns:
            numpy array of shape (num_docs, 768)
        """
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.query_tokenizer(
                    batch,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=self.max_length
                ).to(self.device)
                
                outputs = self.query_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
            
            if (i + batch_size) % 1000 == 0:
                print(f"Encoded {i + batch_size}/{len(documents)} documents")
        
        return np.vstack(all_embeddings)
    
    def rerank(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Rerank documents using cross-encoder for better relevance.
        
        Args:
            query: Question text
            documents: List of candidate document texts
            
        Returns:
            numpy array of relevance scores
        """
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        with torch.no_grad():
            encoded = self.cross_tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            
            logits = self.cross_model(**encoded).logits.squeeze(dim=1)
            scores = logits.cpu().numpy()
        
        return scores


# Example usage
if __name__ == "__main__":
    embedder = MedCPTEmbedder()
    
    # Test query encoding
    query = "What is the role of TP53 in cancer?"
    query_vec = embedder.encode_query(query)
    print(f"Query vector shape: {query_vec.shape}")
    
    # Test reranking
    documents = [
        "TP53 is a tumor suppressor gene that regulates cell cycle.",
        "Cancer treatment involves chemotherapy and radiation.",
        "TP53 mutations are found in over 50% of human cancers."
    ]
    scores = embedder.rerank(query, documents)
    print(f"\nReranking scores:")
    for doc, score in zip(documents, scores):
        print(f"  {score:.4f}: {doc[:50]}...")