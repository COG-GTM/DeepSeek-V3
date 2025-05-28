"""
Evaluator for the GPQA diamond benchmark.

This module provides the core evaluation logic for running the GPQA diamond
benchmark on the DeepSeek-V3 model.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.gpqa_diamond.data_loader import GPQADiamondDataLoader
from evaluation.gpqa_diamond.metrics import GPQAMetrics
from evaluation.utils.inference_wrapper import InferenceWrapper, create_inference_wrapper

logger = logging.getLogger(__name__)

class GPQAEvaluator:
    """
    Evaluator for the GPQA diamond benchmark.
    
    This class provides methods to evaluate the DeepSeek-V3 model
    on the GPQA diamond benchmark.
    """
    
    def __init__(
        self,
        inference_wrapper: InferenceWrapper,
        data_loader: GPQADiamondDataLoader,
        system_prompt: Optional[str] = None,
        batch_size: int = 16,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the GPQA evaluator.
        
        Args:
            inference_wrapper: Inference wrapper for the DeepSeek-V3 model.
            data_loader: Data loader for the GPQA diamond dataset.
            system_prompt: Optional system prompt to prepend to questions.
            batch_size: Batch size for inference.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            output_dir: Directory to save evaluation results.
        """
        self.inference_wrapper = inference_wrapper
        self.data_loader = data_loader
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.output_dir = output_dir
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(
        self,
        indices: Optional[List[int]] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the GPQA diamond benchmark.
        
        Args:
            indices: Optional list of question indices to evaluate.
                If None, evaluates all questions.
            save_results: Whether to save evaluation results to disk.
            
        Returns:
            Dictionary of evaluation results.
        """
        questions = self.data_loader.get_questions(indices)
        
        formatted_questions = []
        for question in questions:
            formatted = self.data_loader.format_question_for_model(
                question, self.system_prompt
            )
            formatted_questions.append(formatted)
        
        all_responses = []
        
        logger.info(f"Generating responses for {len(formatted_questions)} questions")
        
        for i in tqdm(range(0, len(formatted_questions), self.batch_size)):
            batch_questions = formatted_questions[i:i + self.batch_size]
            
            batch_responses = self.inference_wrapper.generate(
                prompts=batch_questions,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            
            all_responses.extend(batch_responses)
        
        correct_answers = [self.data_loader.get_correct_answer(q) for q in questions]
        
        metrics = GPQAMetrics.calculate_pass_at_1(
            predictions=all_responses,
            references=correct_answers,
            extract_answer_fn=self.data_loader.extract_answer_from_response,
        )
        
        results = {
            "metrics": metrics,
            "questions": questions,
            "responses": all_responses,
            "formatted_questions": formatted_questions,
            "correct_answers": correct_answers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        if save_results and self.output_dir is not None:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            results: Dictionary of evaluation results.
        """
        if self.output_dir is None:
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=2)
        
        details_path = os.path.join(self.output_dir, "details.json")
        details = {
            "timestamp": results["timestamp"],
            "questions": [],
        }
        
        for i, (question, response, formatted_question, correct_answer) in enumerate(zip(
            results["questions"],
            results["responses"],
            results["formatted_questions"],
            results["correct_answers"],
        )):
            extracted_answer = self.data_loader.extract_answer_from_response(response)
            
            is_correct = (
                extracted_answer is not None and
                extracted_answer.upper() == correct_answer.upper()
            )
            
            details["questions"].append({
                "index": i,
                "question": question["question"],
                "formatted_question": formatted_question,
                "response": response,
                "correct_answer": correct_answer,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
            })
        
        with open(details_path, "w") as f:
            json.dump(details, f, indent=2)
        
        report_path = os.path.join(self.output_dir, "report.txt")
        report = GPQAMetrics.format_metrics_report(results["metrics"])
        
        with open(report_path, "w") as f:
            f.write(report)
        
        logger.info(f"Saved evaluation results to {self.output_dir}")


def create_evaluator(
    model_path: str,
    dataset_name: str = "spawn99/GPQA-diamond-ClaudeR1",
    inference_framework: str = "sglang",
    config_path: Optional[str] = None,
    dtype: str = "fp8",
    system_prompt: Optional[str] = None,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> GPQAEvaluator:
    """
    Create a GPQA evaluator.
    
    Args:
        model_path: Path to the model weights.
        dataset_name: Name of the dataset on Hugging Face.
        inference_framework: Inference framework to use ("native" or "sglang").
        config_path: Path to the model configuration file.
        dtype: Data type for inference ("fp8" or "bf16").
        system_prompt: Optional system prompt to prepend to questions.
        batch_size: Batch size for inference.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        output_dir: Directory to save evaluation results.
        cache_dir: Directory to cache the dataset.
        
    Returns:
        Initialized GPQA evaluator.
    """
    inference_wrapper = create_inference_wrapper(
        framework=inference_framework,
        model_path=model_path,
        config_path=config_path,
        dtype=dtype,
        max_batch_size=batch_size,
    )
    
    data_loader = GPQADiamondDataLoader(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    ).load()
    
    evaluator = GPQAEvaluator(
        inference_wrapper=inference_wrapper,
        data_loader=data_loader,
        system_prompt=system_prompt,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_dir=output_dir,
    )
    
    return evaluator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    model_path = "/path/to/deepseek-v3-model"
    
    evaluator = create_evaluator(
        model_path=model_path,
        output_dir="./results",
    )
    
    results = evaluator.evaluate(indices=[0, 1, 2])
    
    print(GPQAMetrics.format_metrics_report(results["metrics"]))
