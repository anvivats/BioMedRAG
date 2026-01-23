import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import json
import csv
import random
from typing import List, Dict

import numpy as np
import torch

from models import get_model, MODEL_INFO
from docker.faiss_server import search as retrieve_documents
from data.load_pubmedqa import load_pubmedqa


# =========================
# Experiment configuration
# =========================
SEED = 42
TOP_K = 5
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_NEW_TOKENS = 128

MODELS = ["tinyllama", "phi3", "llama", "biomistral"]

OUTPUT_DIR = Path("experiments/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "model_comparison.csv"
JSON_PATH = OUTPUT_DIR / "detailed_outputs.json"


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Metrics
# =========================
def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()

    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = len(common) / len(gold_tokens) if len(gold_tokens) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    # simple LCS-based ROUGE-L
    def lcs(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                if a[i] == b[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        return dp[-1][-1]

    pred_tokens = pred.split()
    gold_tokens = gold.split()

    lcs_len = lcs(pred_tokens, gold_tokens)
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs_len / len(gold_tokens) if len(gold_tokens) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


# =========================
# Main experiment
# =========================
def run():
    set_seed(SEED)

    print("📥 Loading PubMedQA...")
    dataset = load_pubmedqa(split="test")

    results = []
    detailed_logs = []

    for model_key in MODELS:
        print(f"\n🚀 Evaluating {model_key.upper()}")
        model = get_model(model_key)

        em_scores = []
        f1_scores = []
        rouge_scores = []
        retrieval_hits = []
        latencies = []

        for idx, sample in enumerate(dataset):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(dataset)}")
            
            question = sample["question"]
            gold_answer = sample["answer"]
            gold_pmid = sample.get("pmid")

            # ---------- Retrieval ----------
            retrieved_docs = retrieve_documents(question, top_k=TOP_K)

            if gold_pmid is not None:
                hit = int(any(doc["pmid"] == gold_pmid for doc in retrieved_docs))
                retrieval_hits.append(hit)

            # ---------- Generation ----------
            start = time.time()
            prediction = model.generate(
                question=question,
                context=retrieved_docs,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS
            )
            latency = time.time() - start

            latencies.append(latency)

            # ---------- Metrics ----------
            em_scores.append(exact_match(prediction, gold_answer))
            f1_scores.append(token_f1(prediction, gold_answer))
            rouge_scores.append(rouge_l(prediction, gold_answer))

            detailed_logs.append({
                "model": model_key,
                "question": question,
                "gold_answer": gold_answer,
                "prediction": prediction,
                "latency": latency,
                "retrieved_pmids": [d["pmid"] for d in retrieved_docs]
            })

        result = {
            "model": MODEL_INFO[model_key]["name"],
            "parameters": MODEL_INFO[model_key]["parameters"],
            "EM": np.mean(em_scores),
            "F1": np.mean(f1_scores),
            "ROUGE-L": np.mean(rouge_scores),
            "Recall@5": np.mean(retrieval_hits) if retrieval_hits else None,
            "Latency (s)": np.mean(latencies)
        }

        results.append(result)
        print(f"✓ {model_key}: EM={result['EM']:.3f}, F1={result['F1']:.3f}, Latency={result['Latency (s)']:.2f}s")

    # =========================
    # Save outputs
    # =========================
    print("\n💾 Saving results...")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(JSON_PATH, "w") as f:
        json.dump(detailed_logs, f, indent=2)

    print(f"✅ CSV saved to {CSV_PATH}")
    print(f"✅ JSON saved to {JSON_PATH}")
    print("\n📊 Summary Table:")
    for r in results:
        print(f"  {r['model']:20s} | EM: {r['EM']:.3f} | F1: {r['F1']:.3f} | ROUGE-L: {r['ROUGE-L']:.3f}")


if __name__ == "__main__":
    run()