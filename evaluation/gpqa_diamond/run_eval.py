#!/usr/bin/env python3
"""
Main script to run GPQA diamond benchmark evaluation.

This script provides a command-line interface to run the GPQA diamond
benchmark evaluation on the DeepSeek-V3 model.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any

_project_root = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.gpqa_diamond.evaluator import create_evaluator
from evaluation.utils.result_processor import process_results

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GPQA diamond benchmark evaluation on DeepSeek-V3 model."
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model weights",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp8", "bf16"],
        default="fp8",
        help="Data type for inference (fp8 or bf16)",
    )
    
    parser.add_argument(
        "--inference_framework",
        type=str,
        choices=["native", "sglang", "mock"],
        default="sglang",
        help="Inference framework to use (native, sglang, or mock)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="spawn99/GPQA-diamond-ClaudeR1",
        help="Name of the dataset on Hugging Face",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache the dataset",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt to prepend to questions",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default=None,
        help="Comma-separated list of question indices to evaluate (e.g., '0,1,2')",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save evaluation results",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main():
    """Run GPQA diamond benchmark evaluation."""
    args = parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    sample_indices = None
    if args.sample_indices is not None:
        sample_indices = [int(idx) for idx in args.sample_indices.split(",")]
    
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Running GPQA diamond benchmark evaluation with the following configuration:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Config path: {args.config_path}")
    logger.info(f"  Data type: {args.dtype}")
    logger.info(f"  Inference framework: {args.inference_framework}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max new tokens: {args.max_new_tokens}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Dataset name: {args.dataset_name}")
    logger.info(f"  Sample indices: {sample_indices}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    logger.info("Creating evaluator...")
    evaluator = create_evaluator(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        inference_framework=args.inference_framework,
        config_path=args.config_path,
        dtype=args.dtype,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_dir=None if args.no_save else args.output_dir,
        cache_dir=args.cache_dir,
    )
    
    logger.info("Running evaluation...")
    results = evaluator.evaluate(
        indices=sample_indices,
        save_results=not args.no_save,
    )
    
    if not args.no_save:
        logger.info("Processing results...")
        processed = process_results(args.output_dir)
    
    from evaluation.gpqa_diamond.metrics import GPQAMetrics
    print("\n" + "=" * 80)
    print(GPQAMetrics.format_metrics_report(results["metrics"]))
    print("=" * 80 + "\n")
    
    baseline = 0.591  # 59.1% as mentioned in the README
    pass_at_1 = results["metrics"]["pass@1"]
    diff = pass_at_1 - baseline
    print(f"Comparison with baseline:")
    print(f"  Baseline: {baseline:.4f}")
    print(f"  Current: {pass_at_1:.4f}")
    print(f"  Difference: {diff:.4f} ({diff/baseline:.2%})")
    print("\n")
    
    if not args.no_save:
        print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
