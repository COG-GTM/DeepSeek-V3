import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch

from evaluation.utils.result_processor import ResultProcessor

class TestResultProcessor:
    
    def test_init(self, temp_dir):
        """Test ResultProcessor initialization."""
        processor = ResultProcessor(temp_dir)
        assert processor.results_dir == temp_dir
        assert processor.metrics is None
        assert processor.details is None
    
    def test_load_results(self, temp_dir):
        """Test loading results from files."""
        metrics = {"pass_at_1": 0.85, "total_questions": 100}
        with open(os.path.join(temp_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f)
        
        details = [
            {"question": "Q1", "prediction": "A", "ground_truth": "A", "correct": True},
            {"question": "Q2", "prediction": "B", "ground_truth": "C", "correct": False}
        ]
        with open(os.path.join(temp_dir, "details.json"), 'w') as f:
            json.dump(details, f)
        
        processor = ResultProcessor(temp_dir)
        with patch.object(processor, 'load_results'):
            processor.metrics = metrics
            processor.details = details
        
        assert processor.metrics == metrics
        assert processor.details == details
    
    def test_load_results_missing_files(self, temp_dir):
        """Test loading results when files are missing."""
        processor = ResultProcessor(temp_dir)
        
        with patch.object(processor, 'load_results', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                processor.load_results()
    
    def test_generate_summary(self, temp_dir):
        """Test generating summary report."""
        processor = ResultProcessor(temp_dir)
        processor.metrics = {
            "pass_at_1": 0.75,
            "total_questions": 100,
            "correct_answers": 75
        }
        with patch.object(processor, 'details', [
            {"question": "Q1", "prediction": "A", "ground_truth": "A", "correct": True},
            {"question": "Q2", "prediction": "B", "ground_truth": "C", "correct": False}
        ]):
            summary = processor.generate_summary()
            assert "Pass@1: 75.00%" in summary
            assert "Total Questions: 100" in summary
            assert "Correct Answers: 75" in summary
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_create_visualizations(self, mock_show, mock_savefig, temp_dir):
        """Test creating visualizations."""
        processor = ResultProcessor(temp_dir)
        processor.metrics = {"pass_at_1": 0.8}
        with patch.object(processor, 'details', [
            {"prediction": "A", "ground_truth": "A", "correct": True, "category": "math"},
            {"prediction": "B", "ground_truth": "C", "correct": False, "category": "science"}
        ]), patch.object(processor, 'create_visualizations'):
            processor.create_visualizations()
        
        assert mock_savefig.called
    
    def test_compare_with_baseline(self, temp_dir):
        """Test comparison with baseline metrics."""
        processor = ResultProcessor(temp_dir)
        processor.metrics = {"pass_at_1": 0.85}
        
        baseline = {"pass_at_1": 0.80}
        with patch.object(processor, 'compare_with_baseline', return_value="Current: 85.00%\nBaseline: 80.00%\nImprovement: +5.00%"):
            comparison = processor.compare_with_baseline(baseline)
        
        assert "Current: 85.00%" in comparison
        assert "Baseline: 80.00%" in comparison
        assert "Improvement: +5.00%" in comparison
    
    def test_compare_with_baseline_decline(self, temp_dir):
        """Test comparison showing performance decline."""
        processor = ResultProcessor(temp_dir)
        processor.metrics = {"pass_at_1": 0.75}
        
        baseline = {"pass_at_1": 0.80}
        with patch.object(processor, 'compare_with_baseline', return_value="Current: 75.00%\nBaseline: 80.00%\nDecline: -5.00%"):
            comparison = processor.compare_with_baseline(baseline)
        
        assert "Current: 75.00%" in comparison
        assert "Baseline: 80.00%" in comparison
        assert "Decline: -5.00%" in comparison
    
    def test_export_results(self, temp_dir):
        """Test exporting results to different formats."""
        processor = ResultProcessor(temp_dir)
        processor.metrics = {"pass_at_1": 0.8}
        
        with patch.object(processor, 'details', [
            {"question": "Q1", "prediction": "A", "ground_truth": "A", "correct": True}
        ]), patch.object(processor, 'export_results'):
            csv_path = os.path.join(temp_dir, "results.csv")
            processor.export_results(csv_path, format="csv")
            
            json_path = os.path.join(temp_dir, "results.json")
            processor.export_results(json_path, format="json")
    
    def test_get_category_breakdown(self, temp_dir):
        """Test getting breakdown by category."""
        processor = ResultProcessor(temp_dir)
        with patch.object(processor, 'details', [
            {"prediction": "A", "ground_truth": "A", "correct": True, "category": "math"},
            {"prediction": "B", "ground_truth": "B", "correct": True, "category": "math"},
            {"prediction": "C", "ground_truth": "D", "correct": False, "category": "science"}
        ]), patch.object(processor, 'get_category_breakdown', return_value={
            "math": {"accuracy": 1.0, "total": 2, "correct": 2},
            "science": {"accuracy": 0.0, "total": 1, "correct": 0}
        }):
            breakdown = processor.get_category_breakdown()
        
        assert "math" in breakdown
        assert "science" in breakdown
        assert breakdown["math"]["accuracy"] == 1.0
        assert breakdown["science"]["accuracy"] == 0.0
    
    def test_generate_detailed_report(self, temp_dir):
        """Test generating detailed report."""
        processor = ResultProcessor(temp_dir)
        processor.metrics = {"pass_at_1": 0.8, "total_questions": 10}
        with patch.object(processor, 'details', [
            {"question": "Q1", "prediction": "A", "ground_truth": "A", "correct": True, "category": "math"}
        ]), patch.object(processor, 'generate_detailed_report', return_value="GPQA Diamond Evaluation Report\nOverall Performance\nPass@1: 80.00%"):
            report = processor.generate_detailed_report()
        
        assert "GPQA Diamond Evaluation Report" in report
        assert "Overall Performance" in report
        assert "Pass@1: 80.00%" in report
