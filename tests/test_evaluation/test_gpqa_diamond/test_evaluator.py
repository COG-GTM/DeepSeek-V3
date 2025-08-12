import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from evaluation.gpqa_diamond.evaluator import GPQAEvaluator, create_evaluator

class TestGPQAEvaluator:
    
    def test_init(self, mock_gpqa_data, temp_dir):
        """Test GPQAEvaluator initialization."""
        data_path = os.path.join(temp_dir, "test_data.jsonl")
        with open(data_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        inference_wrapper = Mock()
        data_loader = Mock()
        evaluator = GPQAEvaluator(inference_wrapper, data_loader)
        
        assert hasattr(evaluator, 'inference_wrapper')
        assert hasattr(evaluator, 'data_loader')
    
    @patch('evaluation.gpqa_diamond.evaluator.GPQAMetrics')
    def test_evaluate(self, mock_metrics, mock_gpqa_data, temp_dir):
        """Test evaluation process."""
        data_path = os.path.join(temp_dir, "test_data.jsonl")
        with open(data_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        inference_wrapper = Mock()
        inference_wrapper.batch_generate.return_value = ["B", "B"]
        
        mock_metrics.calculate_pass_at_1.return_value = {
            "pass_at_1": 1.0,
            "total_questions": 2,
            "correct_answers": 2
        }
        mock_metrics.calculate_pass_at_1_by_category.return_value = {
            "geography": {"pass_at_1": 1.0, "total_questions": 1, "correct_answers": 1},
            "math": {"pass_at_1": 1.0, "total_questions": 1, "correct_answers": 1}
        }
        
        data_loader = Mock()
        data_loader.get_questions.return_value = mock_gpqa_data
        data_loader.format_question_for_model.return_value = "formatted question"
        data_loader.get_correct_answer.return_value = "B"
        
        evaluator = GPQAEvaluator(inference_wrapper, data_loader)
        results = evaluator.evaluate()
        
        assert "metrics" in results
        assert "category_metrics" in results
        assert "details" in results
        assert len(results["details"]) == 2
        
        inference_wrapper.batch_generate.assert_called_once()
    
    def test_save_results(self, mock_gpqa_data, temp_dir):
        """Test saving evaluation results."""
        data_path = os.path.join(temp_dir, "test_data.jsonl")
        with open(data_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        inference_wrapper = Mock()
        data_loader = Mock()
        evaluator = GPQAEvaluator(inference_wrapper, data_loader, output_dir=temp_dir)
        
        results = {
            "metrics": {"pass_at_1": 0.8},
            "questions": [],
            "responses": [],
            "formatted_questions": [],
            "correct_answers": [],
            "timestamp": "2023-01-01 00:00:00"
        }
        
        evaluator._save_results(results)
        
        assert os.path.exists(os.path.join(temp_dir, "metrics.json"))
        assert os.path.exists(os.path.join(temp_dir, "details.json"))
        
        with open(os.path.join(temp_dir, "metrics.json"), 'r') as f:
            saved_metrics = json.load(f)
            assert saved_metrics["pass_at_1"] == 0.8

class TestCreateEvaluator:
    
    def test_create_evaluator_mock(self, mock_gpqa_data, temp_dir):
        """Test creating evaluator with mock inference."""
        data_path = os.path.join(temp_dir, "test_data.jsonl")
        with open(data_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        with patch('evaluation.gpqa_diamond.evaluator.create_inference_wrapper') as mock_create_wrapper, \
             patch('evaluation.gpqa_diamond.evaluator.GPQADiamondDataLoader') as mock_loader_class:
            
            mock_wrapper = Mock()
            mock_create_wrapper.return_value = mock_wrapper
            
            mock_loader = Mock()
            mock_loader_class.return_value.load.return_value = mock_loader
            
            evaluator = create_evaluator(data_path)
            
            assert evaluator is not None
    
    def test_create_evaluator_invalid_type(self, temp_dir):
        """Test creating evaluator with invalid inference type."""
        data_path = os.path.join(temp_dir, "test_data.jsonl")
        
        with patch('evaluation.gpqa_diamond.evaluator.create_inference_wrapper') as mock_create_wrapper:
            mock_create_wrapper.side_effect = ValueError("Unsupported inference type")
            
            with pytest.raises(ValueError, match="Unsupported inference type"):
                create_evaluator(data_path, inference_framework="invalid")
    
    @patch('evaluation.gpqa_diamond.evaluator.NativeInferenceWrapper')
    def test_create_evaluator_native(self, mock_native, mock_gpqa_data, temp_dir):
        """Test creating evaluator with native inference."""
        data_path = os.path.join(temp_dir, "test_data.jsonl")
        with open(data_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        mock_native.return_value = Mock()
        
        with patch('evaluation.gpqa_diamond.evaluator.create_inference_wrapper') as mock_create_wrapper, \
             patch('evaluation.gpqa_diamond.evaluator.GPQADiamondDataLoader') as mock_loader_class:
            
            mock_wrapper = Mock()
            mock_create_wrapper.return_value = mock_wrapper
            
            mock_loader = Mock()
            mock_loader_class.return_value.load.return_value = mock_loader
            
            evaluator = create_evaluator(data_path, inference_framework="native", config_path="/fake/config")
            
            assert evaluator is not None
