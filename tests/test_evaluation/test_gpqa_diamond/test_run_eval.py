import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace

from evaluation.gpqa_diamond.run_eval import main, parse_args

class TestRunEval:
    
    def test_parse_args_minimal(self):
        """Test parsing minimal required arguments."""
        with patch('sys.argv', ['run_eval.py', "--data-path", "/path/to/data", "--model-path", "/path/to/model", "--config-path", "/path/to/config"]):
            args = parse_args()
        
        assert args.data_path == "/path/to/data"
        assert args.model_path == "/path/to/model"
        assert args.config_path == "/path/to/config"
        assert args.inference_type == "native"
        assert args.batch_size == 8
        assert args.output_dir == "./results"
    
    def test_parse_args_all_options(self):
        """Test parsing all available arguments."""
        with patch('sys.argv', ['run_eval.py', "--data-path", "/path/to/data", "--model-path", "/path/to/model", "--config-path", "/path/to/config", "--inference-type", "sglang", "--batch-size", "16", "--output-dir", "/custom/output", "--dtype", "fp16"]):
            args = parse_args()
        
        assert args.data_path == "/path/to/data"
        assert args.model_path == "/path/to/model"
        assert args.config_path == "/path/to/config"
        assert args.inference_type == "sglang"
        assert args.batch_size == 16
        assert args.output_dir == "/custom/output"
        assert args.dtype == "fp16"
    
    @patch('evaluation.gpqa_diamond.run_eval.create_evaluator')
    @patch('evaluation.gpqa_diamond.run_eval.ResultProcessor')
    def test_main_successful_evaluation(self, mock_processor, mock_create_evaluator, temp_dir):
        """Test successful evaluation run."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {
            "metrics": {"pass_at_1": 0.85},
            "category_metrics": {},
            "details": []
        }
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        args = Namespace(
            data_path="/fake/data",
            model_path="/fake/model",
            config_path="/fake/config",
            inference_type="mock",
            batch_size=8,
            output_dir=temp_dir,
            dtype="bf16"
        )
        
        main(args)
        
        mock_create_evaluator.assert_called_once()
        mock_evaluator.evaluate.assert_called_once()
        mock_processor.assert_called_once_with(temp_dir)
    
    @patch('evaluation.gpqa_diamond.run_eval.create_evaluator')
    def test_main_evaluation_failure(self, mock_create_evaluator, temp_dir):
        """Test handling of evaluation failure."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate.side_effect = Exception("Evaluation failed")
        mock_create_evaluator.return_value = mock_evaluator
        
        args = Namespace(
            data_path="/fake/data",
            model_path="/fake/model",
            config_path="/fake/config",
            inference_type="mock",
            batch_size=8,
            output_dir=temp_dir,
            dtype="bf16"
        )
        
        with pytest.raises(Exception, match="Evaluation failed"):
            with patch('evaluation.gpqa_diamond.run_eval.parse_args', return_value=args):
                main()
    
    @patch('evaluation.gpqa_diamond.run_eval.create_evaluator')
    @patch('evaluation.gpqa_diamond.run_eval.ResultProcessor')
    @patch('builtins.print')
    def test_main_baseline_comparison(self, mock_print, mock_processor, mock_create_evaluator, temp_dir):
        """Test baseline comparison functionality."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {
            "metrics": {"pass_at_1": 0.85},
            "category_metrics": {},
            "details": []
        }
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        args = Namespace(
            data_path="/fake/data",
            model_path="/fake/model",
            config_path="/fake/config",
            inference_type="mock",
            batch_size=8,
            output_dir=temp_dir,
            dtype="bf16"
        )
        
        main(args)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Baseline comparison" in call for call in print_calls)
        assert any("Current Pass@1: 85.00%" in call for call in print_calls)
    
    @patch('sys.argv', ['run_eval.py', '--data-path', '/fake/data', '--model-path', '/fake/model', '--config-path', '/fake/config'])
    @patch('evaluation.gpqa_diamond.run_eval.main')
    def test_script_entry_point(self, mock_main):
        """Test script entry point."""
        from evaluation.gpqa_diamond.run_eval import __name__
        
        assert callable(main)
