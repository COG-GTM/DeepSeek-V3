"""
Data loader for the GPQA diamond benchmark dataset.

This module provides functionality to load and preprocess the GPQA diamond dataset
for evaluation with the DeepSeek-V3 model.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import datasets
import pandas as pd

logger = logging.getLogger(__name__)

class GPQADiamondDataLoader:
    """
    Data loader for the GPQA diamond benchmark dataset.
    
    This class handles loading the GPQA diamond dataset from Hugging Face,
    preprocessing it into a standardized format, and providing batched access
    for efficient evaluation.
    """
    
    def __init__(
        self,
        dataset_name: str = "spawn99/GPQA-diamond-ClaudeR1",
        cache_dir: Optional[str] = None,
        split: str = "train",
    ):
        """
        Initialize the GPQA diamond data loader.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face.
            cache_dir: Directory to cache the dataset.
            split: Dataset split to use (default: "train").
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.split = split
        self.dataset = None
        self.df = None
        
    def load(self) -> "GPQADiamondDataLoader":
        """
        Load the GPQA diamond dataset from Hugging Face.
        
        Returns:
            Self for method chaining.
        """
        logger.info(f"Loading GPQA diamond dataset: {self.dataset_name}")
        try:
            self.dataset = datasets.load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
            )
            
            self.df = pd.DataFrame(self.dataset[self.split])
            logger.info(f"Loaded {len(self.df)} questions from GPQA diamond dataset")
            
            return self
        except Exception as e:
            logger.error(f"Failed to load GPQA diamond dataset: {e}")
            raise
    
    def get_questions(self, indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Get questions from the dataset.
        
        Args:
            indices: Optional list of indices to retrieve. If None, returns all questions.
            
        Returns:
            List of question dictionaries.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        if indices is not None:
            questions_df = self.df.iloc[indices]
        else:
            questions_df = self.df
            
        return [dict(row) for _, row in questions_df.iterrows()]
    
    def get_batches(self, batch_size: int = 16) -> List[List[Dict[str, Any]]]:
        """
        Get batched questions for efficient processing.
        
        Args:
            batch_size: Number of questions per batch.
            
        Returns:
            List of batches, where each batch is a list of question dictionaries.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        questions = self.get_questions()
        batches = []
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batches.append(batch)
            
        logger.info(f"Created {len(batches)} batches of size {batch_size}")
        return batches
    
    def format_question_for_model(self, question: Dict[str, Any], system_prompt: Optional[str] = None) -> str:
        """
        Format a question for input to the model.
        
        Args:
            question: Question dictionary from the dataset.
            system_prompt: Optional system prompt to prepend to the question.
            
        Returns:
            Formatted question string.
        """
        question_text = question["question"]
        
        if re.search(r'\([A-D]\)', question_text):
            formatted_question = question_text
        else:
            formatted_question = question_text
            
        if system_prompt:
            formatted_question = f"{system_prompt}\n\n{formatted_question}"
            
        return formatted_question
    
    def extract_answer_from_response(self, response: str) -> Optional[str]:
        """
        Extract the answer choice (A, B, C, D) from a model response.
        
        Args:
            response: Model's response text.
            
        Returns:
            Extracted answer choice or None if no valid answer found.
        """
        patterns = [
            r'(?:answer|option)(?:\s+is)?\s*(?::|=|\()?\s*([A-D])',  # "answer is A" or "option: A"
            r'([A-D])(?:\s+is\s+(?:the\s+)?(?:correct|right))',      # "A is correct"
            r'(?:select|choose|pick)\s+(?:option|answer)?\s*(?::|=|\()?\s*([A-D])',  # "select A"
            r'(?:^|\s|\.)([A-D])(?:\.|\s|$)',                        # "A." or " A "
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    def get_correct_answer(self, question: Dict[str, Any]) -> str:
        """
        Get the correct answer for a question.
        
        Args:
            question: Question dictionary from the dataset.
            
        Returns:
            Correct answer choice (A, B, C, D).
        """
        return question["correct_answer"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary of dataset statistics.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        stats = {
            "total_questions": len(self.df),
        }
        
        return stats


def load_gpqa_diamond_dataset(
    dataset_name: str = "spawn99/GPQA-diamond-ClaudeR1",
    cache_dir: Optional[str] = None,
    split: str = "train",
) -> GPQADiamondDataLoader:
    """
    Convenience function to load the GPQA diamond dataset.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face.
        cache_dir: Directory to cache the dataset.
        split: Dataset split to use (default: "train").
        
    Returns:
        Loaded GPQADiamondDataLoader instance.
    """
    loader = GPQADiamondDataLoader(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        split=split,
    )
    return loader.load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = load_gpqa_diamond_dataset()
    questions = loader.get_questions(indices=[0, 1, 2])
    
    for q in questions:
        print(f"Question: {q['question']}")
        print(f"Correct answer: {q['correct_answer']}")
        print(f"Explanation: {q['correct_explanation']}")
        print("-" * 80)
