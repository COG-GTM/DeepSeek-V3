"""
Metrics calculation for the GPQA diamond benchmark.

This module provides functionality to calculate Pass@1 and other metrics
for evaluating model performance on the GPQA diamond benchmark.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

class GPQAMetrics:
    """
    Metrics calculator for the GPQA diamond benchmark.
    
    This class provides methods to calculate Pass@1 and other metrics
    for evaluating model performance on the GPQA diamond benchmark.
    """
    
    @staticmethod
    def calculate_pass_at_1(
        predictions: List[str],
        references: List[str],
        extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    ) -> Dict[str, float]:
        """
        Calculate Pass@1 metric for GPQA diamond benchmark.
        
        Args:
            predictions: List of model predictions (raw text responses).
            references: List of reference answers (A, B, C, D).
            extract_answer_fn: Optional function to extract answer choice from prediction.
                If None, uses a default regex-based extractor.
                
        Returns:
            Dictionary with Pass@1 score and other metrics.
        """
        if len(predictions) != len(references):
            raise ValueError(f"Number of predictions ({len(predictions)}) does not match "
                             f"number of references ({len(references)})")
        
        if extract_answer_fn is None:
            def default_extractor(text: str) -> Optional[str]:
                patterns = [
                    r'(?:answer|option)(?:\s+is)?\s*(?::|=|\()?\s*([A-D])',  # "answer is A" or "option: A"
                    r'([A-D])(?:\s+is\s+(?:the\s+)?(?:correct|right))',      # "A is correct"
                    r'(?:select|choose|pick)\s+(?:option|answer)?\s*(?::|=|\()?\s*([A-D])',  # "select A"
                    r'(?:^|\s|\.)([A-D])(?:\.|\s|$)',                        # "A." or " A "
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        return match.group(1).upper()
                return None
            
            extract_answer_fn = default_extractor
        
        extracted_answers = []
        for pred in predictions:
            answer = extract_answer_fn(pred)
            extracted_answers.append(answer)
        
        correct = 0
        invalid = 0
        
        for extracted, reference in zip(extracted_answers, references):
            if extracted is None:
                invalid += 1
            elif extracted.upper() == reference.upper():
                correct += 1
        
        total = len(predictions)
        valid = total - invalid
        
        pass_at_1 = correct / total if total > 0 else 0
        
        pass_at_1_valid = correct / valid if valid > 0 else 0
        
        metrics = {
            "pass@1": pass_at_1,
            "pass@1_valid": pass_at_1_valid,
            "accuracy": pass_at_1,  # Alias for pass@1
            "total": total,
            "correct": correct,
            "invalid": invalid,
            "valid": valid,
            "invalid_rate": invalid / total if total > 0 else 0,
        }
        
        return metrics
    
    @staticmethod
    def calculate_pass_at_1_by_category(
        predictions: List[str],
        references: List[str],
        categories: List[str],
        extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate Pass@1 metric broken down by category.
        
        Args:
            predictions: List of model predictions (raw text responses).
            references: List of reference answers (A, B, C, D).
            categories: List of category labels for each question.
            extract_answer_fn: Optional function to extract answer choice from prediction.
                
        Returns:
            Dictionary with Pass@1 scores by category and overall.
        """
        if len(predictions) != len(references) or len(predictions) != len(categories):
            raise ValueError("Number of predictions, references, and categories must match")
        
        category_indices = defaultdict(list)
        for i, category in enumerate(categories):
            category_indices[category].append(i)
        
        overall_metrics = GPQAMetrics.calculate_pass_at_1(
            predictions, references, extract_answer_fn
        )
        
        category_metrics = {}
        for category, indices in category_indices.items():
            category_preds = [predictions[i] for i in indices]
            category_refs = [references[i] for i in indices]
            
            metrics = GPQAMetrics.calculate_pass_at_1(
                category_preds, category_refs, extract_answer_fn
            )
            
            category_metrics[category] = metrics
        
        results = {
            "overall": overall_metrics,
            "by_category": category_metrics,
        }
        
        return results
    
    @staticmethod
    def format_metrics_report(metrics: Dict[str, Any]) -> str:
        """
        Format metrics as a human-readable report.
        
        Args:
            metrics: Dictionary of metrics from calculate_pass_at_1 or calculate_pass_at_1_by_category.
            
        Returns:
            Formatted report string.
        """
        report = []
        
        overall = metrics.get("overall", metrics)
        report.append("=== GPQA Diamond Benchmark Results ===")
        report.append(f"Pass@1: {overall['pass@1']:.4f} ({overall['correct']}/{overall['total']})")
        report.append(f"Pass@1 (valid answers only): {overall['pass@1_valid']:.4f} ({overall['correct']}/{overall['valid']})")
        report.append(f"Invalid answers: {overall['invalid']} ({overall['invalid_rate']:.2%})")
        report.append("")
        
        if "by_category" in metrics:
            report.append("=== Results by Category ===")
            for category, cat_metrics in metrics["by_category"].items():
                report.append(f"{category}:")
                report.append(f"  Pass@1: {cat_metrics['pass@1']:.4f} ({cat_metrics['correct']}/{cat_metrics['total']})")
                report.append(f"  Invalid answers: {cat_metrics['invalid']} ({cat_metrics['invalid_rate']:.2%})")
            report.append("")
        
        return "\n".join(report)


def calculate_pass_at_1(
    predictions: List[str],
    references: List[str],
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, float]:
    """
    Convenience function to calculate Pass@1 metric.
    
    Args:
        predictions: List of model predictions (raw text responses).
        references: List of reference answers (A, B, C, D).
        extract_answer_fn: Optional function to extract answer choice from prediction.
            
    Returns:
        Dictionary with Pass@1 score and other metrics.
    """
    return GPQAMetrics.calculate_pass_at_1(predictions, references, extract_answer_fn)


def calculate_pass_at_1_by_category(
    predictions: List[str],
    references: List[str],
    categories: List[str],
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to calculate Pass@1 metric by category.
    
    Args:
        predictions: List of model predictions (raw text responses).
        references: List of reference answers (A, B, C, D).
        categories: List of category labels for each question.
        extract_answer_fn: Optional function to extract answer choice from prediction.
            
    Returns:
        Dictionary with Pass@1 scores by category and overall.
    """
    return GPQAMetrics.calculate_pass_at_1_by_category(
        predictions, references, categories, extract_answer_fn
    )


if __name__ == "__main__":
    predictions = [
        "I think the answer is A because...",
        "The correct option is B due to...",
        "This is a difficult question, but I believe C is right...",
        "The answer should be D based on...",
        "I'm not sure about this one...",
    ]
    
    references = ["A", "B", "D", "D", "C"]
    
    metrics = calculate_pass_at_1(predictions, references)
    print(GPQAMetrics.format_metrics_report(metrics))
