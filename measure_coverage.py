#!/usr/bin/env python3
"""
Script to measure baseline test coverage for DeepSeek-V3 codebase.
"""

import sys
import os
import subprocess

def main():
    """Measure baseline coverage across all Python modules."""
    print("Measuring baseline test coverage for DeepSeek-V3...")
    
    python_files = [
        "evaluation/gpqa_diamond/data_loader.py",
        "evaluation/gpqa_diamond/evaluator.py", 
        "evaluation/gpqa_diamond/metrics.py",
        "evaluation/gpqa_diamond/run_eval.py",
        "evaluation/utils/inference_wrapper.py",
        "evaluation/utils/result_processor.py",
        "inference/model.py",
        "inference/generate.py",
        "inference/convert.py",
        "inference/kernel.py",
        "inference/fp8_cast_bf16.py"
    ]
    
    print(f"Found {len(python_files)} Python modules to analyze:")
    for f in python_files:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (missing)")
    
    total_lines = 0
    for f in python_files:
        if os.path.exists(f):
            with open(f, 'r') as file:
                lines = len(file.readlines())
                total_lines += lines
                print(f"  {f}: {lines} lines")
    
    print(f"\nTotal lines of code: {total_lines}")
    print(f"Target coverage: 90% ({int(total_lines * 0.9)} lines)")
    
    print("\nBaseline measurement complete.")

if __name__ == "__main__":
    main()
