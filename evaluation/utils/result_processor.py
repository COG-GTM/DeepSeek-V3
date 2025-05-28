"""
Result processor for benchmark evaluations.

This module provides functionality to process and visualize
evaluation results for benchmarks.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

class ResultProcessor:
    """
    Result processor for benchmark evaluations.
    
    This class provides methods to process and visualize
    evaluation results for benchmarks.
    """
    
    def __init__(
        self,
        results_dir: str,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the result processor.
        
        Args:
            results_dir: Directory containing evaluation results.
            output_dir: Directory to save processed results and visualizations.
                If None, uses results_dir.
        """
        self.results_dir = results_dir
        self.output_dir = output_dir or results_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.metrics = self._load_metrics()
        self.details = self._load_details()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """
        Load metrics from results directory.
        
        Returns:
            Dictionary of metrics.
        """
        metrics_path = os.path.join(self.results_dir, "metrics.json")
        
        if not os.path.exists(metrics_path):
            logger.warning(f"Metrics file not found: {metrics_path}")
            return {}
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        return metrics
    
    def _load_details(self) -> Dict[str, Any]:
        """
        Load details from results directory.
        
        Returns:
            Dictionary of details.
        """
        details_path = os.path.join(self.results_dir, "details.json")
        
        if not os.path.exists(details_path):
            logger.warning(f"Details file not found: {details_path}")
            return {}
        
        with open(details_path, "r") as f:
            details = json.load(f)
        
        return details
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of evaluation results.
        
        Returns:
            Dictionary of summary statistics.
        """
        if not self.metrics:
            logger.warning("No metrics available for summary")
            return {}
        
        summary = {
            "pass@1": self.metrics.get("pass@1", 0),
            "pass@1_valid": self.metrics.get("pass@1_valid", 0),
            "total_questions": self.metrics.get("total", 0),
            "correct_answers": self.metrics.get("correct", 0),
            "invalid_answers": self.metrics.get("invalid", 0),
            "invalid_rate": self.metrics.get("invalid_rate", 0),
        }
        
        if self.details and "timestamp" in self.details:
            summary["timestamp"] = self.details["timestamp"]
        
        return summary
    
    def plot_performance(self, save_path: Optional[str] = None) -> Figure:
        """
        Plot performance metrics.
        
        Args:
            save_path: Path to save the plot. If None, uses default path.
            
        Returns:
            Matplotlib figure.
        """
        if not self.metrics:
            logger.warning("No metrics available for plotting")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pass_at_1 = self.metrics.get("pass@1", 0)
        pass_at_1_valid = self.metrics.get("pass@1_valid", 0)
        
        bars = ax.bar(
            ["Pass@1", "Pass@1 (valid answers only)"],
            [pass_at_1, pass_at_1_valid],
            color=["#3498db", "#2ecc71"],
        )
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )
        
        baseline = 0.591  # 59.1% as mentioned in the README
        ax.axhline(y=baseline, color="r", linestyle="--", label=f"Baseline: {baseline:.4f}")
        
        ax.set_ylabel("Score")
        ax.set_title("GPQA Diamond Benchmark Performance")
        ax.set_ylim(0, 1.0)
        ax.legend()
        
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            default_path = os.path.join(self.output_dir, "performance.png")
            plt.savefig(default_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_answer_distribution(self, save_path: Optional[str] = None) -> Figure:
        """
        Plot distribution of correct, incorrect, and invalid answers.
        
        Args:
            save_path: Path to save the plot. If None, uses default path.
            
        Returns:
            Matplotlib figure.
        """
        if not self.metrics:
            logger.warning("No metrics available for plotting")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
            return fig
        
        correct = self.metrics.get("correct", 0)
        invalid = self.metrics.get("invalid", 0)
        total = self.metrics.get("total", 0)
        incorrect = total - correct - invalid
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = ["Correct", "Incorrect", "Invalid"]
        sizes = [correct, incorrect, invalid]
        colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
        explode = (0.1, 0, 0)  # Explode the 1st slice (Correct)
        
        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        
        ax.axis("equal")
        
        ax.set_title("Distribution of Answers")
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            default_path = os.path.join(self.output_dir, "answer_distribution.png")
            plt.savefig(default_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report of evaluation results.
        
        Args:
            save_path: Path to save the report. If None, uses default path.
            
        Returns:
            Report string.
        """
        if not self.metrics:
            logger.warning("No metrics available for report")
            return "No metrics available for report"
        
        summary = self.generate_summary()
        
        report = []
        report.append("# GPQA Diamond Benchmark Evaluation Report")
        report.append("")
        
        if "timestamp" in summary:
            report.append(f"Evaluation timestamp: {summary['timestamp']}")
            report.append("")
        
        report.append("## Summary")
        report.append("")
        report.append(f"- Pass@1: {summary['pass@1']:.4f} ({summary['correct_answers']}/{summary['total_questions']})")
        report.append(f"- Pass@1 (valid answers only): {summary['pass@1_valid']:.4f}")
        report.append(f"- Invalid answers: {summary['invalid_answers']} ({summary['invalid_rate']:.2%})")
        report.append("")
        
        baseline = 0.591  # 59.1% as mentioned in the README
        diff = summary['pass@1'] - baseline
        report.append("## Comparison with Baseline")
        report.append("")
        report.append(f"- Baseline: {baseline:.4f}")
        report.append(f"- Difference: {diff:.4f} ({diff/baseline:.2%})")
        report.append("")
        
        report.append("## Visualizations")
        report.append("")
        report.append("See the following files for visualizations:")
        report.append("- performance.png: Performance metrics")
        report.append("- answer_distribution.png: Distribution of answers")
        report.append("")
        
        if self.details and "questions" in self.details:
            report.append("## Sample Questions and Responses")
            report.append("")
            
            num_samples = min(5, len(self.details["questions"]))
            for i in range(num_samples):
                question = self.details["questions"][i]
                report.append(f"### Question {i+1}")
                report.append("")
                report.append(f"**Question:** {question['question']}")
                report.append("")
                report.append(f"**Model Response:** {question['response']}")
                report.append("")
                report.append(f"**Correct Answer:** {question['correct_answer']}")
                report.append(f"**Extracted Answer:** {question['extracted_answer'] or 'None'}")
                report.append(f"**Correct:** {question['is_correct']}")
                report.append("")
        
        report_str = "\n".join(report)
        
        if save_path is not None:
            with open(save_path, "w") as f:
                f.write(report_str)
        else:
            default_path = os.path.join(self.output_dir, "report.md")
            with open(default_path, "w") as f:
                f.write(report_str)
        
        return report_str
    
    def process_results(self) -> Dict[str, Any]:
        """
        Process evaluation results and generate all outputs.
        
        Returns:
            Dictionary of processed results.
        """
        summary = self.generate_summary()
        
        performance_fig = self.plot_performance()
        distribution_fig = self.plot_answer_distribution()
        
        report = self.generate_report()
        
        processed = {
            "summary": summary,
            "report": report,
        }
        
        return processed


def process_results(
    results_dir: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to process evaluation results.
    
    Args:
        results_dir: Directory containing evaluation results.
        output_dir: Directory to save processed results and visualizations.
            If None, uses results_dir.
            
    Returns:
        Dictionary of processed results.
    """
    processor = ResultProcessor(results_dir, output_dir)
    return processor.process_results()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    results_dir = "./results"
    
    processed = process_results(results_dir)
    
    print("Summary:")
    for key, value in processed["summary"].items():
        print(f"  {key}: {value}")
