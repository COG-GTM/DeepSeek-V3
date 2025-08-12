import pytest
import os
import tempfile
import json
from unittest.mock import patch, Mock

from evaluation.gpqa_diamond.data_loader import GPQADiamondDataLoader

class TestGPQADiamondDataLoader:
    
    def test_init(self, temp_dir):
        """Test GPQADiamondDataLoader initialization."""
        loader = GPQADiamondDataLoader()
        assert hasattr(loader, 'dataset_name')
        assert hasattr(loader, 'dataset')
        assert hasattr(loader, 'df')
    
    @patch('datasets.load_dataset')
    def test_load(self, mock_load_dataset, mock_gpqa_data):
        """Test loading GPQA data."""
        mock_dataset = {"train": mock_gpqa_data}
        mock_load_dataset.return_value = mock_dataset
        
        loader = GPQADiamondDataLoader()
        loader.load()
        
        questions = loader.get_questions()
        assert len(questions) == 2
        assert questions[0]["question"] == "What is the capital of France?"
        assert questions[1]["answer"] == "B"
    
    @patch('datasets.load_dataset')
    def test_get_questions(self, mock_load_dataset, mock_gpqa_data):
        """Test getting questions from loaded data."""
        mock_dataset = {"train": mock_gpqa_data}
        mock_load_dataset.return_value = mock_dataset
        
        loader = GPQADiamondDataLoader()
        loader.load()
        questions = loader.get_questions()
        
        assert len(questions) == 2
        assert questions[0]["question"] == "What is the capital of France?"
    
    @patch('datasets.load_dataset')
    def test_get_batches(self, mock_load_dataset, mock_gpqa_data):
        """Test getting batches from data."""
        mock_dataset = {"train": mock_gpqa_data}
        mock_load_dataset.return_value = mock_dataset
        
        loader = GPQADiamondDataLoader()
        loader.load()
        batches = loader.get_batches(batch_size=1)
        
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert batches[0][0]["question"] == "What is the capital of France?"
    
    def test_format_question_for_model(self):
        """Test formatting question for model input."""
        loader = GPQADiamondDataLoader()
        question_data = {
            "question": "What is 2+2?",
            "choices": ["A) 3", "B) 4", "C) 5", "D) 6"]
        }
        
        formatted = loader.format_question_for_model(question_data)
        
        assert "What is 2+2?" in formatted
    
    def test_extract_answer_from_response(self):
        """Test extracting answer from model response."""
        loader = GPQADiamondDataLoader()
        
        assert loader.extract_answer_from_response("The answer is B") == "B"
        assert loader.extract_answer_from_response("B) is correct") == "B"
        assert loader.extract_answer_from_response("I choose A") == "A"
        assert loader.extract_answer_from_response("No clear answer") is None
    
    @patch('datasets.load_dataset')
    def test_load_dataset_error(self, mock_load_dataset):
        """Test loading dataset with error."""
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        loader = GPQADiamondDataLoader()
        
        with pytest.raises(Exception):
            loader.load()
