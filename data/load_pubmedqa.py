"""
Load PubMedQA dataset into SQLite database
Downloads from HuggingFace datasets and populates the database
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------------------------------
# Project-safe imports (NO try/except, NO relative imports)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.create_database import DatabaseManager


class PubMedQALoader:
    """Loads PubMedQA dataset into SQLite database."""

    def __init__(self, db_path: str = "data/pubmed.db"):
        self.db_path = Path(db_path)
        self.db = None

    # ------------------------------------------------------------------
    # Download dataset
    # ------------------------------------------------------------------
    def download_dataset(self, split: str = "train"):
        print(f"📥 Downloading PubMedQA dataset (split: {split})...")

        dataset = load_dataset(
            "qiaojin/PubMedQA",
            "pqa_labeled",
            split=split,
        )

        print(f"✓ Downloaded {len(dataset)} examples")
        return dataset

    # ------------------------------------------------------------------
    # Extract documents
    # ------------------------------------------------------------------
    def extract_documents_from_dataset(self, dataset) -> List[Dict]:
        """
        Extract unique documents from PubMedQA dataset.
        """
        print("📝 Extracting documents from dataset...")

        documents = {}

        for example in tqdm(dataset, desc="Processing examples"):
            pubid = example.get("pubid")
            context = example.get("context", {})

            contexts = context.get("contexts", [])
            meshes = context.get("meshes", [])

            if not pubid or not contexts:
                continue

            abstract = " ".join(contexts)

            documents[pubid] = {
                "pmid": pubid,
                "title": f"PubMed Article {pubid}",
                "abstract": abstract,
                "journal": None,
                "published_date": None,
                "authors": None,
                "mesh_terms": ",".join(meshes) if meshes else None,
            }

        print(f"✓ Extracted {len(documents)} unique documents")
        return list(documents.values())

    # ------------------------------------------------------------------
    # Load into SQLite
    # ------------------------------------------------------------------
    def load_into_database(self, documents: List[Dict], batch_size: int = 1000):
        print(f"💾 Loading {len(documents)} documents into database...")

        self.db = DatabaseManager(self.db_path)
        self.db.connect()

        doc_tuples = [
            (
                doc["pmid"],
                doc["title"],
                doc["abstract"],
                doc["journal"],
                doc["published_date"],
                doc["authors"],
                doc["mesh_terms"],
            )
            for doc in documents
        ]

        for i in tqdm(range(0, len(doc_tuples), batch_size), desc="Inserting"):
            self.db.insert_documents_batch(
                doc_tuples[i : i + batch_size]
            )

        stats = self.db.get_stats()
        print("\nDatabase statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        self.db.close()
        print("✓ Database load complete")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self, splits: List[str] = ["train"]):
        all_documents = []

        for split in splits:
            print("\n" + "=" * 60)
            print(f"Processing split: {split}")
            print("=" * 60)

            dataset = self.download_dataset(split)
            docs = self.extract_documents_from_dataset(dataset)
            all_documents.extend(docs)

        unique_docs = {doc["pmid"]: doc for doc in all_documents}
        print(f"\n✓ Total unique documents: {len(unique_docs)}")

        self.load_into_database(list(unique_docs.values()))


# ------------------------------------------------------------------
# Load dataset for evaluation (used in experiments)
# ------------------------------------------------------------------
def load_pubmedqa(split: str = "train", test_size: int = 200) -> List[Dict]:
    """
    Load PubMedQA dataset for evaluation/experiments.
    
    Args:
        split: "train", "test", or "all"
               Note: PubMedQA only has a 'train' split, so we create train/test
        test_size: Number of examples to reserve for test set
    
    Returns:
        List of dicts with keys: question, answer, pmid
    """
    print(f"📥 Loading PubMedQA dataset for evaluation...")
    
    # Load the full train split (only split available in pqa_labeled)
    raw_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    
    dataset = []
    for example in raw_dataset:
        dataset.append({
            "question": example.get("question", ""),
            "answer": example.get("final_decision", ""),  # "yes", "no", or "maybe"
            "pmid": example.get("pubid", None)
        })
    
    # Split into train/test based on request
    if split == "test":
        dataset = dataset[-test_size:]  # Last N examples as test
        print(f"✓ Loaded {len(dataset)} examples (test set)")
    elif split == "train":
        dataset = dataset[:-test_size]  # Rest as train
        print(f"✓ Loaded {len(dataset)} examples (train set)")
    else:  # "all" or any other value
        print(f"✓ Loaded {len(dataset)} examples (full dataset)")
    
    return dataset


# ------------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PubMedQA Dataset Loader")
    print("=" * 60)

    db_path = "data/pubmed.db"

    print("\nStep 1: Creating database schema...")
    db = DatabaseManager(db_path)
    db.connect()
    db.create_schema()
    db.close()
    print("✓ Schema ready")

    print("\nStep 2: Loading PubMedQA dataset...")
    loader = PubMedQALoader(db_path)
    loader.run(splits=["train"])

    print("\n" + "=" * 60)
    print("✅ PUBMEDQA LOADING COMPLETE")
    print("=" * 60)
    print(f"Database: {Path(db_path).absolute()}")


if __name__ == "__main__":
    main()