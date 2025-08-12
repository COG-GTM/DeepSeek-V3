import pytest
from evaluation.gpqa_diamond.metrics import GPQAMetrics

class TestGPQAMetrics:
    
    def test_calculate_pass_at_1_perfect_score(self):
        """Test pass@1 calculation with perfect score."""
        predictions = ["A", "B", "C", "D"]
        ground_truth = ["A", "B", "C", "D"]
        
        metrics = GPQAMetrics.calculate_pass_at_1(predictions, ground_truth)
        
        assert metrics["pass_at_1"] == 1.0
        assert metrics["total_questions"] == 4
        assert metrics["correct_answers"] == 4
    
    def test_calculate_pass_at_1_zero_score(self):
        """Test pass@1 calculation with zero score."""
        predictions = ["B", "A", "D", "C"]
        ground_truth = ["A", "B", "C", "D"]
        
        metrics = GPQAMetrics.calculate_pass_at_1(predictions, ground_truth)
        
        assert metrics["pass_at_1"] == 0.0
        assert metrics["total_questions"] == 4
        assert metrics["correct_answers"] == 0
    
    def test_calculate_pass_at_1_partial_score(self):
        """Test pass@1 calculation with partial score."""
        predictions = ["A", "A", "C", "C"]
        ground_truth = ["A", "B", "C", "D"]
        
        metrics = GPQAMetrics.calculate_pass_at_1(predictions, ground_truth)
        
        assert metrics["pass_at_1"] == 0.5
        assert metrics["total_questions"] == 4
        assert metrics["correct_answers"] == 2
    
    def test_calculate_pass_at_1_empty_lists(self):
        """Test pass@1 calculation with empty lists."""
        predictions = []
        ground_truth = []
        
        metrics = GPQAMetrics.calculate_pass_at_1(predictions, ground_truth)
        
        assert metrics["pass_at_1"] == 0.0
        assert metrics["total_questions"] == 0
        assert metrics["correct_answers"] == 0
    
    def test_calculate_pass_at_1_mismatched_lengths(self):
        """Test pass@1 calculation with mismatched list lengths."""
        predictions = ["A", "B"]
        ground_truth = ["A", "B", "C"]
        
        with pytest.raises(ValueError, match="Predictions and ground truth must have the same length"):
            GPQAMetrics.calculate_pass_at_1(predictions, ground_truth)
    
    def test_calculate_pass_at_1_by_category(self):
        """Test pass@1 calculation by category."""
        predictions = ["A", "B", "C", "D"]
        ground_truth = ["A", "A", "C", "C"]
        categories = ["math", "math", "science", "science"]
        
        metrics = GPQAMetrics.calculate_pass_at_1_by_category(predictions, ground_truth, categories)
        
        assert "math" in metrics
        assert "science" in metrics
        assert metrics["math"]["pass_at_1"] == 0.5
        assert metrics["science"]["pass_at_1"] == 0.5
        assert metrics["math"]["total_questions"] == 2
        assert metrics["science"]["total_questions"] == 2
    
    def test_calculate_pass_at_1_by_category_single_category(self):
        """Test pass@1 calculation by category with single category."""
        predictions = ["A", "B"]
        ground_truth = ["A", "B"]
        categories = ["math", "math"]
        
        metrics = GPQAMetrics.calculate_pass_at_1_by_category(predictions, ground_truth, categories)
        
        assert "math" in metrics
        assert metrics["math"]["pass_at_1"] == 1.0
        assert metrics["math"]["total_questions"] == 2
    
    def test_format_metrics_report(self):
        """Test formatting metrics report."""
        metrics = {
            "pass_at_1": 0.75,
            "total_questions": 100,
            "correct_answers": 75
        }
        
        report = GPQAMetrics.format_metrics_report(metrics)
        
        assert "Pass@1: 75.00%" in report
        assert "Total Questions: 100" in report
        assert "Correct Answers: 75" in report
    
    def test_format_metrics_report_with_categories(self):
        """Test formatting metrics report with categories."""
        metrics = {
            "pass_at_1": 0.8,
            "total_questions": 50,
            "correct_answers": 40
        }
        category_metrics = {
            "math": {"pass_at_1": 0.9, "total_questions": 25, "correct_answers": 22},
            "science": {"pass_at_1": 0.7, "total_questions": 25, "correct_answers": 18}
        }
        
        report = GPQAMetrics.format_metrics_report(metrics)
        
        assert "Overall Pass@1: 80.00%" in report
        assert "math: 90.00%" in report
        assert "science: 70.00%" in report
