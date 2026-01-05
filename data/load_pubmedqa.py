"""
Load PubMedQA dataset into SQLite database
Downloads from HuggingFace datasets and populates the database
"""

import json
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
from tqdm import tqdm
import sys

# Handle imports
try:
    from .create_database import DatabaseManager
except ImportError:
    from create_database import DatabaseManager


class PubMedQALoader:
    """Loads PubMedQA dataset into SQLite database."""
    
    def __init__(self, db_path: str = "data/pubmed.db"):
        """
        Initialize loader.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db = None
    
    def download_dataset(self, split: str = "train"):
        """
        Download PubMedQA dataset from HuggingFace.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            
        Returns:
            Dataset object
        """
        print(f"📥 Downloading PubMedQA dataset (split: {split})...")
        
        try:
            # PubMedQA dataset from HuggingFace
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
            print(f"✓ Downloaded {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"⚠ Error downloading dataset: {e}")
            print("  Trying alternative source...")
            
            # Alternative: Load from local JSON if download fails
            return self._load_from_local()
    
    def _load_from_local(self):
        """Fallback: Load from local JSON file if available."""
        local_path = Path("data/pubmedqa_raw.json")
        if local_path.exists():
            print(f"  Loading from local file: {local_path}")
            with open(local_path) as f:
                data = json.load(f)
            return data
        else:
            raise FileNotFoundError(
                "Could not download dataset and no local file found. "
                "Please check your internet connection."
            )
    
    def extract_documents_from_dataset(self, dataset) -> List[Dict]:
        """
        Extract unique documents from PubMedQA dataset.
        
        PubMedQA format:
        {
            'pubid': int,
            'question': str,
            'context': {
                'contexts': [str, str, ...],  # List of abstracts
                'labels': [str, str, ...],
                'meshes': [str, str, ...]
            },
            'long_answer': str,
            'final_decision': str
        }
        
        Args:
            dataset: HuggingFace dataset or list of examples
            
        Returns:
            List of document dictionaries
        """
        print("📝 Extracting documents from dataset...")
        
        documents = {}  # Use dict to avoid duplicates (key = pmid)
        
        for example in tqdm(dataset, desc="Processing examples"):
            pubid = example.get('pubid', None)
            
            # PubMedQA has context split into multiple sentences
            contexts = example.get('context', {}).get('contexts', [])
            meshes = example.get('context', {}).get('meshes', [])
            
            if pubid and contexts:
                # Combine context sentences into single abstract
                abstract = ' '.join(contexts)
                
                # Create document entry
                documents[pubid] = {
                    'pmid': pubid,
                    'title': f"PubMed Article {pubid}",  # PubMedQA doesn't include titles
                    'abstract': abstract,
                    'journal': None,
                    'published_date': None,
                    'authors': None,
                    'mesh_terms': ','.join(meshes) if meshes else None
                }
        
        print(f"✓ Extracted {len(documents)} unique documents")
        return list(documents.values())
    
    def load_into_database(self, documents: List[Dict], batch_size: int = 1000):
        """
        Load documents into SQLite database.
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to insert at once
        """
        print(f"💾 Loading {len(documents)} documents into database...")
        
        self.db = DatabaseManager(self.db_path)
        self.db.connect()
        
        # Convert to tuples for batch insert
        doc_tuples = [
            (
                doc['pmid'],
                doc['title'],
                doc['abstract'],
                doc['journal'],
                doc['published_date'],
                doc['authors'],
                doc['mesh_terms']
            )
            for doc in documents
        ]
        
        # Insert in batches
        for i in tqdm(range(0, len(doc_tuples), batch_size), desc="Inserting batches"):
            batch = doc_tuples[i:i + batch_size]
            self.db.insert_documents_batch(batch)
        
        print("✓ All documents loaded successfully")
        
        # Print stats
        stats = self.db.get_stats()
        print("\nDatabase Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        self.db.close()
    
    def load_full_pipeline(self, splits: List[str] = ["train"]):
        """
        Complete pipeline: download, extract, and load.
        
        Args:
            splits: List of dataset splits to load
        """
        all_documents = []
        
        for split in splits:
            print(f"\n{'=' * 60}")
            print(f"Processing split: {split}")
            print('=' * 60)
            
            # Download dataset
            dataset = self.download_dataset(split)
            
            # Extract documents
            documents = self.extract_documents_from_dataset(dataset)
            all_documents.extend(documents)
        
        # Remove duplicates across splits
        unique_docs = {doc['pmid']: doc for doc in all_documents}
        print(f"\n✓ Total unique documents across all splits: {len(unique_docs)}")
        
        # Load into database
        print(f"\n{'=' * 60}")
        print("Loading into Database")
        print('=' * 60)
        self.load_into_database(list(unique_docs.values()))


def main():
    """Main execution function."""
    print("=" * 60)
    print("PubMedQA Dataset Loader")
    print("=" * 60)
    print()
    
    # Initialize database first
    print("Step 1: Initializing database schema...")
    db = DatabaseManager("data/pubmed.db")
    db.connect()
    db.create_schema()
    db.close()
    print("✓ Database schema ready\n")
    
    # Load PubMedQA data
    print("Step 2: Loading PubMedQA dataset...")
    loader = PubMedQALoader("data/pubmed.db")
    
    # Load train split (you can add 'validation', 'test' too)
    loader.load_full_pipeline(splits=["train"])
    
    print("\n" + "=" * 60)
    print("✅ PUBMEDQA LOADING COMPLETE")
    print("=" * 60)
    print(f"\nDatabase location: {Path('data/pubmed.db').absolute()}")
    print("\nNext steps:")
    print("  1. Build FAISS index: python data/build_faiss_index.py")
    print("  2. Start FAISS server: docker-compose up")
    print("  3. Run experiments: python experiments/run_comparison.py")


if __name__ == "__main__":
    main()