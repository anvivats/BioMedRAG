"""
Evaluation metrics for biomedical question answering
Implements accuracy, F1, and other metrics for PubMedQA evaluation
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


class BiomedEvaluator:
    """
    Evaluator for biomedical QA tasks.
    Supports both exact match and relaxed matching metrics.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = []
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """
        Normalize answer text for comparison.
        
        Args:
            answer: Raw answer text
            
        Returns:
            Normalized answer (lowercase, no punctuation)
        """
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer
    
    @staticmethod
    def extract_yes_no_maybe(answer: str) -> str:
        """
        Extract yes/no/maybe from answer text.
        Used for PubMedQA evaluation.
        
        Args:
            answer: Generated answer text
            
        Returns:
            'yes', 'no', 'maybe', or 'unknown'
        """
        answer_lower = answer.lower()
        
        # Check for explicit yes/no/maybe
        if 'yes' in answer_lower[:50]:  # Check first 50 chars
            return 'yes'
        elif 'no' in answer_lower[:50]:
            return 'no'
        elif 'maybe' in answer_lower[:50] or 'unclear' in answer_lower[:50]:
            return 'maybe'
        
        # Check for affirmative/negative patterns
        affirmative_patterns = [
            'is true', 'correct', 'does occur', 'is associated',
            'can cause', 'has been shown'
        ]
        negative_patterns = [
            'is not', 'does not', 'no evidence', 'cannot',
            'has not been shown', 'is false'
        ]
        
        for pattern in affirmative_patterns:
            if pattern in answer_lower[:100]:
                return 'yes'
        
        for pattern in negative_patterns:
            if pattern in answer_lower[:100]:
                return 'no'
        
        return 'unknown'
    
    def exact_match(self, prediction: str, ground_truth: str) -> bool:
        """
        Exact match after normalization.
        
        Args:
            prediction: Model's answer
            ground_truth: Correct answer
            
        Returns:
            True if exact match
        """
        pred_norm = self.normalize_answer(prediction)
        truth_norm = self.normalize_answer(ground_truth)
        return pred_norm == truth_norm
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        Token-level F1 score.
        
        Args:
            prediction: Model's answer
            ground_truth: Correct answer
            
        Returns:
            F1 score (0-1)
        """
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0
        
        # Count overlapping tokens
        pred_counter = Counter(pred_tokens)
        truth_counter = Counter(truth_tokens)
        
        overlap = sum((pred_counter & truth_counter).values())
        
        if overlap == 0:
            return 0.0
        
        precision = overlap / len(pred_tokens)
        recall = overlap / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate_yes_no_question(
        self,
        prediction: str,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate yes/no/maybe questions (PubMedQA format).
        
        Args:
            prediction: Model's answer
            ground_truth: Correct answer ('yes', 'no', or 'maybe')
            
        Returns:
            Dict with accuracy and extracted label
        """
        extracted = self.extract_yes_no_maybe(prediction)
        ground_truth_norm = ground_truth.lower().strip()
        
        # Exact match
        exact = 1.0 if extracted == ground_truth_norm else 0.0
        
        # Partial credit for 'maybe' (if unsure)
        partial = 0.0
        if extracted == 'maybe' and ground_truth_norm != 'maybe':
            partial = 0.3  # Give partial credit for uncertainty
        
        return {
            'exact_match': exact,
            'partial_credit': partial,
            'extracted_label': extracted,
            'ground_truth': ground_truth_norm
        }
    
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        question_type: str = "yes_no"
    ) -> Dict:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of model answers
            ground_truths: List of correct answers
            question_type: 'yes_no' or 'free_form'
            
        Returns:
            Dict with aggregate metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        results = []
        
        for pred, truth in zip(predictions, ground_truths):
            if question_type == "yes_no":
                result = self.evaluate_yes_no_question(pred, truth)
            else:
                # Free-form questions
                result = {
                    'exact_match': float(self.exact_match(pred, truth)),
                    'f1_score': self.f1_score(pred, truth)
                }
            
            results.append(result)
        
        # Aggregate metrics
        if question_type == "yes_no":
            accuracy = np.mean([r['exact_match'] for r in results])
            partial_accuracy = np.mean([
                r['exact_match'] + r['partial_credit'] for r in results
            ])
            
            return {
                'accuracy': accuracy,
                'partial_accuracy': partial_accuracy,
                'num_samples': len(predictions),
                'results': results
            }
        else:
            exact_match_score = np.mean([r['exact_match'] for r in results])
            f1_score = np.mean([r['f1_score'] for r in results])
            
            return {
                'exact_match': exact_match_score,
                'f1_score': f1_score,
                'num_samples': len(predictions),
                'results': results
            }
    
    def confusion_matrix(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict:
        """
        Create confusion matrix for yes/no/maybe questions.
        
        Args:
            predictions: List of model answers
            ground_truths: List of correct answers
            
        Returns:
            Confusion matrix as dict
        """
        labels = ['yes', 'no', 'maybe', 'unknown']
        matrix = {label: {other: 0 for other in labels} for label in labels}
        
        for pred, truth in zip(predictions, ground_truths):
            extracted = self.extract_yes_no_maybe(pred)
            truth_norm = truth.lower().strip()
            
            if truth_norm not in labels:
                truth_norm = 'unknown'
            
            matrix[truth_norm][extracted] += 1
        
        return matrix
    
    def print_confusion_matrix(self, matrix: Dict):
        """Pretty print confusion matrix."""
        labels = ['yes', 'no', 'maybe', 'unknown']
        
        print("\nConfusion Matrix:")
        print("=" * 60)
        print(f"{'True \\ Pred':<15} {'yes':<10} {'no':<10} {'maybe':<10} {'unknown':<10}")
        print("-" * 60)
        
        for true_label in labels:
            row = f"{true_label:<15}"
            for pred_label in labels:
                count = matrix.get(true_label, {}).get(pred_label, 0)
                row += f"{count:<10}"
            print(row)
        
        print("=" * 60)
    
    def compute_metrics_by_model(
        self,
        results: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Compute metrics for multiple models.
        
        Args:
            results: Dict mapping model_name -> list of result dicts
            
        Returns:
            Comparison table
        """
        comparison = {}
        
        for model_name, model_results in results.items():
            predictions = [r['answer'] for r in model_results]
            ground_truths = [r['ground_truth'] for r in model_results]
            
            # Compute metrics
            metrics = self.evaluate_batch(predictions, ground_truths)
            
            # Add timing info
            avg_time = np.mean([r['total_time'] for r in model_results])
            avg_retrieval_time = np.mean([
                r.get('retrieval_time', 0) for r in model_results
            ])
            avg_generation_time = np.mean([
                r.get('generation_time', 0) for r in model_results
            ])
            
            comparison[model_name] = {
                'accuracy': metrics['accuracy'],
                'avg_time': avg_time,
                'avg_retrieval_time': avg_retrieval_time,
                'avg_generation_time': avg_generation_time,
                'num_samples': len(model_results)
            }
        
        return comparison
    
    def print_comparison_table(self, comparison: Dict):
        """Pretty print model comparison table."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        header = f"{'Model':<15} {'Accuracy':<12} {'Total (s)':<12} {'Retrieval':<12} {'Generation':<12}"
        print(header)
        print("-" * 80)
        
        for model_name, metrics in comparison.items():
            row = (
                f"{model_name.upper():<15} "
                f"{metrics['accuracy']:<12.2%} "
                f"{metrics['avg_time']:<12.2f} "
                f"{metrics['avg_retrieval_time']:<12.2f} "
                f"{metrics['avg_generation_time']:<12.2f}"
            )
            print(row)
        
        print("=" * 80)


# Example usage
if __name__ == "__main__":
    evaluator = BiomedEvaluator()
    
    # Test yes/no questions
    predictions = [
        "Yes, TP53 mutations are found in over 50% of cancers.",
        "No, there is no evidence linking TP53 to diabetes.",
        "The relationship is unclear and requires more research."
    ]
    
    ground_truths = ["yes", "no", "maybe"]
    
    print("Testing Yes/No Evaluation:")
    print("=" * 60)
    
    results = evaluator.evaluate_batch(predictions, ground_truths, "yes_no")
    
    print(f"\nAccuracy: {results['accuracy']:.2%}")
    print(f"Partial Accuracy: {results['partial_accuracy']:.2%}")
    print(f"Samples: {results['num_samples']}")
    
    # Confusion matrix
    matrix = evaluator.confusion_matrix(predictions, ground_truths)
    evaluator.print_confusion_matrix(matrix)