# GPQA Diamond Benchmark

This directory contains the evaluation framework for the GPQA Diamond benchmark, a challenging dataset of graduate-level physics questions.

## Overview

The GPQA Diamond benchmark consists of 198 hard undergraduate-level physics questions, with 87% focused on quantum mechanics. The benchmark is designed to test the reasoning capabilities of large language models on complex physics problems.

## Usage

To run the GPQA Diamond benchmark evaluation:

```bash
python run_eval.py \
    --model_path /path/to/deepseek-v3-model \
    --config_path /path/to/config.json \
    --inference_framework sglang \
    --dtype fp8 \
    --output_dir ./results
```

For a full list of command-line arguments, see the [main README](../README.md#command-line-arguments).

## Components

- **data_loader.py**: Loads and preprocesses the GPQA Diamond dataset from Hugging Face
- **evaluator.py**: Provides the core evaluation logic for running the benchmark
- **metrics.py**: Implements the Pass@1 scoring methodology
- **run_eval.py**: Main script to execute the evaluation from the command line

## Example

Here's a minimal example of how to use the evaluation framework programmatically:

```python
from evaluation.gpqa_diamond.evaluator import create_evaluator

# Create evaluator
evaluator = create_evaluator(
    model_path="/path/to/deepseek-v3-model",
    config_path="/path/to/config.json",
    inference_framework="sglang",
    dtype="fp8",
    output_dir="./results",
)

# Run evaluation
results = evaluator.evaluate()

# Print report
from evaluation.gpqa_diamond.metrics import GPQAMetrics
print(GPQAMetrics.format_metrics_report(results["metrics"]))
```

## Expected Performance

DeepSeek-V3 achieves 59.1% Pass@1 on the GPQA Diamond benchmark, as mentioned in the repository README. This evaluation framework is designed to reproduce this result.

## References

- [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022)
- [GPQA Diamond Dataset on Hugging Face](https://huggingface.co/datasets/spawn99/GPQA-diamond-ClaudeR1)
