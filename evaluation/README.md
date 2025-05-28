# DeepSeek-V3 Evaluation Framework

This directory contains the evaluation framework for benchmarking the DeepSeek-V3 model on various tasks.

## Available Benchmarks

- [GPQA Diamond](#gpqa-diamond-benchmark): Graduate-level physics Q&A benchmark

## GPQA Diamond Benchmark

The GPQA Diamond benchmark is a challenging dataset of 198 hard undergraduate-level physics questions, with 87% focused on quantum mechanics. The benchmark is designed to test the reasoning capabilities of large language models on complex physics problems.

### Usage

To run the GPQA Diamond benchmark evaluation:

```bash
python -m evaluation.gpqa_diamond.run_eval \
    --model_path /path/to/deepseek-v3-model \
    --config_path /path/to/config.json \
    --inference_framework sglang \
    --dtype fp8 \
    --output_dir ./results
```

### Command-line Arguments

#### Model Arguments
- `--model_path`: Path to the model weights (required)
- `--config_path`: Path to the model configuration file
- `--dtype`: Data type for inference (`fp8` or `bf16`, default: `fp8`)

#### Inference Arguments
- `--inference_framework`: Inference framework to use (`native` or `sglang`, default: `sglang`)
- `--batch_size`: Batch size for inference (default: 16)
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.0)

#### Dataset Arguments
- `--dataset_name`: Name of the dataset on Hugging Face (default: `spawn99/GPQA-diamond-ClaudeR1`)
- `--cache_dir`: Directory to cache the dataset
- `--system_prompt`: System prompt to prepend to questions
- `--sample_indices`: Comma-separated list of question indices to evaluate (e.g., '0,1,2')

#### Output Arguments
- `--output_dir`: Directory to save evaluation results (default: `./results`)
- `--no_save`: Do not save evaluation results

#### Logging Arguments
- `--verbose`: Enable verbose logging

### Example Results

Running the GPQA Diamond benchmark on DeepSeek-V3 should produce results similar to:

```
=== GPQA Diamond Benchmark Results ===
Pass@1: 0.5910 (117/198)
Pass@1 (valid answers only): 0.6053 (117/193)
Invalid answers: 5 (2.53%)

Comparison with baseline:
  Baseline: 0.5910
  Current: 0.5910
  Difference: 0.0000 (0.00%)
```

### Output Files

The evaluation framework generates the following output files:

- `metrics.json`: JSON file containing evaluation metrics
- `details.json`: JSON file containing detailed evaluation results
- `report.txt`: Text file containing a human-readable report
- `report.md`: Markdown file containing a comprehensive report
- `performance.png`: Plot of performance metrics
- `answer_distribution.png`: Plot of answer distribution

## Framework Structure

The evaluation framework is organized as follows:

```
evaluation/
  |- gpqa_diamond/
     |- data_loader.py     # Dataset loading and preprocessing
     |- evaluator.py       # Core evaluation logic
     |- metrics.py         # Pass@1 calculation
     |- run_eval.py        # Main script to execute evaluation
  |- utils/
     |- inference_wrapper.py  # Interface to inference frameworks
     |- result_processor.py   # Analyze and visualize results
```

### Components

- **data_loader.py**: Loads and preprocesses the GPQA Diamond dataset from Hugging Face
- **evaluator.py**: Provides the core evaluation logic for running the benchmark
- **metrics.py**: Implements the Pass@1 scoring methodology
- **run_eval.py**: Main script to execute the evaluation from the command line
- **inference_wrapper.py**: Provides a wrapper around different inference frameworks
- **result_processor.py**: Processes and visualizes evaluation results

## Adding New Benchmarks

To add a new benchmark, create a new directory under `evaluation/` with a similar structure to `gpqa_diamond/`. The new benchmark should implement at least:

- A data loader for loading and preprocessing the benchmark dataset
- An evaluator for running the benchmark
- Metrics calculation for the benchmark
- A main script to execute the benchmark from the command line

## References

- [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022)
- [GPQA Diamond Dataset on Hugging Face](https://huggingface.co/datasets/spawn99/GPQA-diamond-ClaudeR1)
